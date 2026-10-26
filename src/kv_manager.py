"""
Slot pool and KV-cache policies for agents that pause for tool calls.

The scheduler, not llama-server, decides which slot each request uses
(via "id_slot"), so it knows whose KV sits in which slot. When an agent
pauses for a tool call, a policy chooses what happens to its KV:

  preserve  keep the slot reserved for the agent until it returns
            (optionally only for a time-to-live, as in Continuum)
  release   free the slot; the KV survives only if no one else uses
            the slot first (llama-server's default behavior)
  offload   save the slot's KV to disk, then free the slot; when the
            agent returns, restore it unless it is still in a slot
"""

import asyncio
import itertools
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Slot:
    slot_id: int
    state: str = "free"  # free | busy | reserved
    holder: Optional[str] = None  # agent whose KV is in this slot
    last_used: float = 0.0
    expiry: Optional[asyncio.TimerHandle] = None


@dataclass
class AgentKV:
    agent_id: str
    reserved_slot: Optional[Slot] = None
    disk_file: Optional[str] = None
    tool_history: list = field(default_factory=list)


@dataclass
class Decision:
    action: str  # preserve | release | offload
    ttl: Optional[float] = None
    reason: str = ""


class SlotPool:
    """
    Hands out server slots to agents. Waiters are served FIFO, or, with
    ready_priority, agents returning from a tool call go first unless a
    new task has waited longer than aging_s (so long tasks do not starve
    new ones and vice versa).
    """

    def __init__(self, n_slots, ready_priority=False, aging_s=10.0):
        self.slots = [Slot(i) for i in range(n_slots)]
        self.ready_priority = ready_priority
        self.aging_s = aging_s
        self.waiters = []
        self.sequence = itertools.count()

    def free_slots(self):
        return [s for s in self.slots if s.state == "free"]

    def waiting(self):
        return len(self.waiters)

    def slot_holding(self, agent):
        for slot in self.slots:
            if slot.holder == agent.agent_id:
                return slot
        return None

    async def acquire(self, agent, returning):
        if agent.reserved_slot is not None:
            slot = agent.reserved_slot
            agent.reserved_slot = None
            if slot.expiry:
                slot.expiry.cancel()
                slot.expiry = None
            slot.state = "busy"
            return slot

        future = asyncio.get_running_loop().create_future()
        self.waiters.append(
            (returning, time.perf_counter(), next(self.sequence), agent, future)
        )
        self._dispatch()
        return await future

    def _next_waiter(self):
        if not self.ready_priority:
            return min(self.waiters, key=lambda w: w[2])

        now = time.perf_counter()
        starving = [
            w for w in self.waiters
            if not w[0] and now - w[1] > self.aging_s
        ]
        if starving:
            return min(starving, key=lambda w: w[2])

        return min(self.waiters, key=lambda w: (not w[0], w[2]))

    def _dispatch(self):
        while self.waiters and self.free_slots():
            waiter = self._next_waiter()
            self.waiters.remove(waiter)
            _, _, _, agent, future = waiter

            free = self.free_slots()
            # Prefer the slot that still holds this agent's KV, then an
            # empty slot, then the least recently used one.
            own = [s for s in free if s.holder == agent.agent_id]
            empty = [s for s in free if s.holder is None]
            slot = (own or empty or sorted(free, key=lambda s: s.last_used))[0]

            slot.state = "busy"
            future.set_result(slot)

    def release(self, slot):
        slot.state = "free"
        slot.last_used = time.perf_counter()
        self._dispatch()

    def reserve(self, agent, slot, ttl=None):
        slot.state = "reserved"
        agent.reserved_slot = slot

        if ttl is not None:
            def expire():
                if agent.reserved_slot is slot:
                    agent.reserved_slot = None
                    slot.expiry = None
                    self.release(slot)

            slot.expiry = asyncio.get_running_loop().call_later(ttl, expire)


class CostModel:
    """
    Per-token costs (milliseconds) measured by experiments/kv_cost_probe.py
    on the target machine. Defaults are the RTX 5070 laptop numbers.
    """

    def __init__(self, recompute_ms=0.14, save_ms=0.018, restore_ms=0.013):
        self.recompute_ms = recompute_ms
        self.save_ms = save_ms
        self.restore_ms = restore_ms

    def recompute_s(self, tokens):
        return tokens * self.recompute_ms / 1000

    def offload_s(self, tokens):
        return tokens * (self.save_ms + self.restore_ms) / 1000


class ToolTimePredictor:
    """
    Predicts the next tool time from the agent's own recent tool times
    (exponential moving average), falling back to the median of all tool
    times seen so far. The real tool time is never used for the decision.
    """

    def __init__(self, alpha=0.5, prior_s=1.0):
        self.alpha = alpha
        self.prior_s = prior_s
        self.observed = []

    def predict(self, agent):
        if agent.tool_history:
            estimate = agent.tool_history[0]
            for value in agent.tool_history[1:]:
                estimate = self.alpha * value + (1 - self.alpha) * estimate
            return estimate

        if self.observed:
            ordered = sorted(self.observed)
            return ordered[len(ordered) // 2]

        return self.prior_s

    def record(self, agent, tool_s):
        agent.tool_history.append(tool_s)
        self.observed.append(tool_s)


class Policy:
    name = "policy"

    def decide(self, agent, context_tokens, pool, predicted_tool_s):
        raise NotImplementedError


class PreservePolicy(Policy):
    name = "preserve"

    def decide(self, agent, context_tokens, pool, predicted_tool_s):
        return Decision("preserve")


class ReleasePolicy(Policy):
    name = "release"

    def decide(self, agent, context_tokens, pool, predicted_tool_s):
        return Decision("release")


class OffloadPolicy(Policy):
    name = "offload"

    def decide(self, agent, context_tokens, pool, predicted_tool_s):
        return Decision("offload")


class TTLPolicy(Policy):
    """Continuum-style: keep the KV for a fixed time, then release it."""

    name = "ttl"

    def __init__(self, ttl_s):
        self.ttl_s = ttl_s

    def decide(self, agent, context_tokens, pool, predicted_tool_s):
        return Decision("preserve", ttl=self.ttl_s)


class MinWastePolicy(Policy):
    """
    InferCept-style min-waste choice, in slot-seconds that other agents
    lose. Holding a slot only wastes time if someone needs it; offloading
    costs the save and restore I/O; releasing costs a recompute when the
    agent returns, if its slot was taken in the meantime.
    """

    name = "min_waste"

    def __init__(self, cost_model):
        self.cost = cost_model

    def decide(self, agent, context_tokens, pool, predicted_tool_s):
        # Other agents need a slot now or soon: this slot is the one
        # being released, so pressure exists if anyone is waiting or no
        # other slot is free.
        pressure = pool.waiting() > 0 or len(pool.free_slots()) == 0

        if not pressure:
            return Decision("preserve", reason="no pressure")

        costs = {
            "preserve": predicted_tool_s,
            "offload": self.cost.offload_s(context_tokens),
            "release": self.cost.recompute_s(context_tokens),
        }
        action = min(costs, key=costs.get)
        reason = ", ".join(f"{k}={v:.3f}s" for k, v in costs.items())

        if action == "preserve":
            # Bound the bet: if the tool runs long, stop holding the slot.
            return Decision(
                "preserve",
                ttl=max(costs["offload"], 2 * predicted_tool_s),
                reason=reason,
            )

        return Decision(action, reason=reason)


def make_policy(name, cost_model, ttl_s=2.0):
    if name == "preserve":
        return PreservePolicy()
    if name == "release":
        return ReleasePolicy()
    if name == "offload":
        return OffloadPolicy()
    if name == "ttl":
        return TTLPolicy(ttl_s)
    if name == "min_waste":
        return MinWastePolicy(cost_model)
    raise ValueError(f"Unknown policy: {name}")

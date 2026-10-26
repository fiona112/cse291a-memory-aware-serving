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
    state: str = "free"  # free | busy | reserved | reclaiming
    holder: Optional[str] = None  # agent whose KV is in this slot
    last_used: float = 0.0
    expiry: Optional[asyncio.TimerHandle] = None
    reserved_for: Optional["AgentKV"] = None


@dataclass
class AgentKV:
    agent_id: str
    reserved_slot: Optional[Slot] = None
    disk_file: Optional[str] = None
    tool_history: list = field(default_factory=list)
    predicted_return: Optional[float] = None
    context_tokens: int = 0


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
        # Set by enable_reclaim(): on-demand eviction of idle agents.
        self.reclaimer = None
        self.cost_model = None

    def enable_reclaim(self, reclaimer, cost_model):
        """
        reclaimer(agent, slot, action) is a coroutine that saves or drops
        the agent's KV and then calls release(slot).
        """
        self.reclaimer = reclaimer
        self.cost_model = cost_model

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
            slot.reserved_for = None
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

        self._reclaim()

    def _reclaim(self):
        """
        If agents are waiting and no slot is free, evict idle agents whose
        tools are predicted to run longest, as long as their remaining
        wait is longer than what offloading costs.
        """
        if not self.reclaimer or self.free_slots():
            return

        in_progress = sum(s.state == "reclaiming" for s in self.slots)
        needed = len(self.waiters) - in_progress
        if needed <= 0:
            return

        now = time.perf_counter()
        candidates = []
        for slot in self.slots:
            agent = slot.reserved_for
            if slot.state != "reserved" or agent is None:
                continue
            remaining = (agent.predicted_return or now) - now
            offload_s = self.cost_model.offload_s(agent.context_tokens)
            if remaining > offload_s:
                candidates.append((remaining, slot, agent))

        candidates.sort(key=lambda c: c[0], reverse=True)
        for _, slot, agent in candidates[:needed]:
            slot.state = "reclaiming"
            slot.reserved_for = None
            agent.reserved_slot = None
            if slot.expiry:
                slot.expiry.cancel()
                slot.expiry = None

            tokens = agent.context_tokens
            action = (
                "offload"
                if self.cost_model.offload_s(tokens) < self.cost_model.recompute_s(tokens)
                else "release"
            )
            asyncio.get_running_loop().create_task(
                self.reclaimer(agent, slot, action)
            )

    def release(self, slot):
        slot.state = "free"
        slot.reserved_for = None
        slot.last_used = time.perf_counter()
        self._dispatch()

    def reserve(self, agent, slot, ttl=None):
        slot.state = "reserved"
        slot.reserved_for = agent
        agent.reserved_slot = slot

        if ttl is not None:
            def expire():
                if agent.reserved_slot is slot:
                    agent.reserved_slot = None
                    slot.expiry = None
                    self.release(slot)

            slot.expiry = asyncio.get_running_loop().call_later(ttl, expire)

        self._reclaim()


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
    InferCept-style min-waste, applied on demand. Holding an idle agent's
    KV wastes nothing until another agent needs the slot, so every pause
    starts as preserve. When a slot is needed and none is free, the pool
    (SlotPool._reclaim) evicts the idle agent with the longest predicted
    remaining tool time, if that time exceeds the offload cost, and uses
    whichever of offload or release costs less.
    """

    name = "min_waste"

    def __init__(self, cost_model):
        self.cost = cost_model

    def decide(self, agent, context_tokens, pool, predicted_tool_s):
        agent.context_tokens = context_tokens
        agent.predicted_return = time.perf_counter() + predicted_tool_s
        return Decision(
            "preserve",
            reason=f"predicted tool {predicted_tool_s:.2f}s",
        )


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

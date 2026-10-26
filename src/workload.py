from dataclasses import dataclass


@dataclass
class Task:
    task_id: str
    prompt: str
    max_tokens: int
    workload_type: str
    tool_delay: float = 0.0
    # Multistep only: resend the whole conversation in step 2,
    # as a real agent does, instead of a fresh prompt.
    agent_history: bool = False


def make_background(task_id: str, context_repeats: int) -> str:
    # Each repeat is ~32 tokens, so 120 repeats is
    # ~3.9k prompt tokens with Qwen2.5's tokenizer.
    return (
        f"This is background information for task {task_id}. "
        "The fictional city has parks, schools, stores, public transportation, "
        "restaurants, libraries, and residential neighborhoods. "
        * context_repeats
    )


def make_short_task(task_id: str) -> Task:
    return Task(
        task_id=task_id,
        prompt=f"Task {task_id}: Explain why the sky is blue in one sentence.",
        max_tokens=50,
        workload_type="short",
    )


def make_long_context_task(
    task_id: str,
    context_repeats: int = 120,
) -> Task:
    prompt = (
        make_background(task_id, context_repeats)
        + f"\nFor task {task_id}, summarize the city in one sentence."
    )

    return Task(
        task_id=task_id,
        prompt=prompt,
        max_tokens=50,
        workload_type="long_context",
    )


def make_long_output_task(task_id: str) -> Task:
    return Task(
        task_id=task_id,
        prompt=(
            f"Task {task_id}: Explain how photosynthesis works in detail. "
            "Include the main inputs, outputs, and stages."
        ),
        max_tokens=400,
        workload_type="long_output",
    )


def make_multistep_task(
    task_id: str,
    tool_delay: float = 2.0,
    agent_history: bool = False,
    context_repeats: int = 120,
) -> Task:
    if agent_history:
        # An agent that carries a large context across the tool
        # call, so whether its KV survives the pause matters.
        prompt = (
            make_background(task_id, context_repeats)
            + f"\nFor task {task_id}, give me three facts about this city."
        )
    else:
        prompt = f"Task {task_id}: Give me three facts about penguins."

    return Task(
        task_id=task_id,
        prompt=prompt,
        max_tokens=100,
        workload_type="multistep",
        tool_delay=tool_delay,
        agent_history=agent_history,
    )


def make_test_workload(
    tasks_per_type: int = 12,
    context_repeats: int = 120,
    tool_delay: float = 2.0,
    agent_history: bool = False,
):
    tasks = []

    for i in range(tasks_per_type):
        tasks.append(make_short_task(f"short_{i}"))
        tasks.append(
            make_long_context_task(
                f"long_context_{i}",
                context_repeats=context_repeats,
            )
        )
        tasks.append(make_long_output_task(f"long_output_{i}"))
        tasks.append(
            make_multistep_task(
                f"multistep_{i}",
                tool_delay=tool_delay,
                agent_history=agent_history,
                context_repeats=context_repeats,
            )
        )

    return tasks
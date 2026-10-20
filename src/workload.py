from dataclasses import dataclass


@dataclass
class Task:
    task_id: str
    prompt: str
    max_tokens: int
    workload_type: str
    tool_delay: float = 0.0


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
    # Each repeat is ~32 tokens, so 120 repeats is
    # ~3.9k prompt tokens with Qwen2.5's tokenizer.
    long_context = (
        f"This is background information for task {task_id}. "
        "The fictional city has parks, schools, stores, public transportation, "
        "restaurants, libraries, and residential neighborhoods. "
        * context_repeats
    )

    prompt = (
        long_context
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


def make_multistep_task(task_id: str) -> Task:
    return Task(
        task_id=task_id,
        prompt=f"Task {task_id}: Give me three facts about penguins.",
        max_tokens=100,
        workload_type="multistep",
        tool_delay=2.0,
    )


def make_test_workload(
    tasks_per_type: int = 12,
    context_repeats: int = 120,
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
        tasks.append(make_multistep_task(f"multistep_{i}"))

    return tasks
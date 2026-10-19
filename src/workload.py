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


def make_long_context_task(task_id: str) -> Task:
    long_context = (
        f"This is background information for task {task_id}. "
        "The fictional city has parks, schools, stores, public transportation, "
        "restaurants, libraries, and residential neighborhoods. "
        * 120
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


def make_test_workload():
    tasks = []

    for i in range(3):
        tasks.append(make_short_task(f"short_{i}"))
        tasks.append(make_long_context_task(f"long_context_{i}"))
        tasks.append(make_long_output_task(f"long_output_{i}"))
        tasks.append(make_multistep_task(f"multistep_{i}"))

    return tasks
import asyncio
import time


class FIFOScheduler:
    def __init__(self, concurrency: int):
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)

    async def run_task(self, task, task_runner, arrival_time):
        async with self.semaphore:
            start_time = time.perf_counter()

            queue_time = start_time - arrival_time

            return await task_runner(
                task,
                queue_time=queue_time,
            )

    async def run(self, tasks, task_runner):
        jobs = []

        for task in tasks:
            arrival_time = time.perf_counter()

            job = asyncio.create_task(
                self.run_task(
                    task,
                    task_runner,
                    arrival_time,
                )
            )

            jobs.append(job)

        return await asyncio.gather(*jobs)
import time

import aiohttp

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

# Per-request timeout in seconds. A request that takes
# longer than this raises asyncio.TimeoutError and the
# task is recorded as timed out.
REQUEST_TIMEOUT = 120.0


async def generate_with_stats(
    prompt: str,
    max_tokens: int = 100,
    timeout: float = None,
):
    """
    Sends one chat request and returns (content, stats).

    prompt is either a single user message or a full list
    of chat messages (for agents that resend their history).

    stats holds per-request measurements: wall-clock
    latency plus the token counts and prefill/decode
    timings that llama-server reports in its response.
    """
    if isinstance(prompt, str):
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    else:
        messages = prompt

    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    client_timeout = aiohttp.ClientTimeout(
        total=timeout or REQUEST_TIMEOUT
    )

    start_time = time.perf_counter()

    async with aiohttp.ClientSession(
        timeout=client_timeout
    ) as session:
        async with session.post(LLM_URL, json=payload) as response:
            response.raise_for_status()
            data = await response.json()

    latency = time.perf_counter() - start_time

    usage = data.get("usage", {})
    timings = data.get("timings", {})

    stats = {
        "latency": latency,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": timings.get("cache_n"),
        "prefill_ms": timings.get("prompt_ms"),
        "decode_ms": timings.get("predicted_ms"),
        "finish_reason": data["choices"][0].get("finish_reason"),
    }

    return data["choices"][0]["message"]["content"], stats


async def generate(prompt: str, max_tokens: int = 100) -> str:
    content, _ = await generate_with_stats(
        prompt,
        max_tokens=max_tokens,
    )

    return content

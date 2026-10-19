import aiohttp

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"


async def generate(prompt: str, max_tokens: int = 100) -> str:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(LLM_URL, json=payload) as response:
            response.raise_for_status()
            data = await response.json()

    return data["choices"][0]["message"]["content"]
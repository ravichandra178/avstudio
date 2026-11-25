import os

import httpx
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROK_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is missing; set it in .env or export it in your shell.")

endpoint = os.getenv(
    "GROQ_ENDPOINT", "https://api.groq.com/openai/v1/chat/completions"
)
model = os.getenv("GROK_MODEL", "openai/gpt-oss-20b")

response = httpx.post(
    endpoint,
    headers={"Authorization": f"Bearer {groq_api_key}"},
    json={
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful question generator."},
            {"role": "user", "content": "Generate 5 multiple-choice questions on B-Cells."},
        ],
        "max_tokens": 800,
    },
    timeout=60,
)
response.raise_for_status()

print(response.json()["choices"][0]["message"]["content"])
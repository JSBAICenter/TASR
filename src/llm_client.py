import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CACHE_DIR = Path("data/llm_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = os.getenv("LLM_BASE_URL", "https://your-openai-compatible-endpoint.example/v1")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "qwen3.6-27b")
API_KEY = os.getenv("LLM_API_KEY", "REPLACE_ME")
NO_THINKING_KWARG = os.getenv("LLM_NO_THINKING_KWARG", "").lower() in ("1", "true", "yes")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def _cache_key(prompt: str, model: str, temperature: float, thinking: bool) -> str:
    raw = json.dumps(
        {"prompt": prompt, "model": model, "temperature": temperature, "thinking": thinking},
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def call_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 256,
    thinking: bool = False,
) -> str:
    key = _cache_key(prompt, model, temperature, thinking)
    cache_file = CACHE_DIR / f"{key}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)["response"]

    kwargs = dict(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if not NO_THINKING_KWARG:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": thinking}}
    response = client.chat.completions.create(**kwargs)
    text = (response.choices[0].message.content or "").strip()

    with open(cache_file, "w") as f:
        json.dump({"prompt": prompt, "model": model, "response": text}, f)

    return text


if __name__ == "__main__":
    r1 = call_llm("What is 2+2? Answer with just the number.")
    print(f"First call: {r1}")
    r2 = call_llm("What is 2+2? Answer with just the number.")
    print(f"Second call (cached): {r2}")
    print(f"Cache files: {len(list(CACHE_DIR.iterdir()))}")

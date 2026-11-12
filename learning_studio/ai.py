import json

import httpx

from .config import Settings
from .prompts import EXTRACTION_SYSTEM, JSON_SUFFIX, TUTOR_SYSTEM, extraction_prompt


class AIExtractionError(RuntimeError):
    pass


class AIProvider:
    def __init__(self, settings: Settings):
        self.settings = settings

    def extract(self, text: str, document_id: str) -> list[dict]:
        errors: list[str] = []
        for provider in self.settings.providers:
            try:
                return self._request(provider, text, document_id)
            except (httpx.HTTPError, ValueError, KeyError, AIExtractionError) as exc:
                errors.append(f"{provider}: {exc}")
        raise AIExtractionError("All AI providers failed: " + "; ".join(errors))

    def _request(self, provider: str, text: str, document_id: str) -> list[dict]:
        config = {
            "github_models": (self.settings.github_token, self.settings.github_models_endpoint, self.settings.github_models_name),
            "groq": (self.settings.groq_api_key, self.settings.groq_endpoint, self.settings.groq_model),
            "openai": (self.settings.openai_api_key, self.settings.openai_endpoint, self.settings.openai_model),
        }.get(provider)
        if not config or not config[0]:
            raise AIExtractionError("provider is not configured")
        prompt = extraction_prompt(document_id, text)
        response = httpx.post(
            config[1],
            headers={"Authorization": f"Bearer {config[0]}"},
            json={
                "model": config[2],
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)

    def answer(
        self, instruction: str, context: str, timeout_seconds: float | None = None
    ) -> str:
        errors: list[str] = []
        providers = self.settings.providers
        if timeout_seconds is not None:
            providers = providers[:1]
        for provider in providers:
            try:
                return self._chat(provider, instruction, context, timeout_seconds)
            except (httpx.HTTPError, ValueError, KeyError, AIExtractionError) as exc:
                errors.append(f"{provider}: {exc}")
        raise AIExtractionError("All AI providers failed: " + "; ".join(errors))

    def gemini_answer(self, instruction: str, context: str) -> str:
        if not self.settings.gemini_api_key:
            raise AIExtractionError("GEMINI_API_KEY is not configured")
        model = self.settings.gemini_model.removeprefix("models/")
        response = httpx.post(
            f"{self.settings.gemini_endpoint.rstrip('/')}/{model}:generateContent",
            params={"key": self.settings.gemini_api_key},
            json={
                "contents": [{"parts": [{"text": f"{instruction}\n\nStudy context:\n{context}"}]}],
                "generationConfig": {"temperature": 0.2},
            },
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        response.raise_for_status()
        try:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise AIExtractionError("Gemini returned no text") from exc

    def gemini_search_answer(
        self, instruction: str, context: str, timeout_seconds: float | None = None
    ) -> str:
        if not self.settings.gemini_api_key:
            raise AIExtractionError("GEMINI_API_KEY is not configured")
        model = self.settings.gemini_model.removeprefix("models/")
        endpoint = (
            self.settings.gemini_endpoint.rstrip("/").removesuffix("/models")
            + "/interactions"
        )
        timeout = timeout_seconds or 120.0
        response = httpx.post(
            endpoint,
            headers={
                "x-goog-api-key": self.settings.gemini_api_key,
                "Api-Revision": "2026-05-20",
            },
            json={
                "model": model,
                "input": f"{instruction}\n\nStudy context:\n{context}",
                "tools": [{"type": "google_search"}],
            },
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
        )
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as exc:
            raise AIExtractionError("Gemini returned invalid JSON") from exc
        answer = "\n".join(
            block["text"]
            for step in data.get("steps", [])
            if step.get("type") == "model_output"
            for block in step.get("content", [])
            if block.get("type") == "text" and isinstance(block.get("text"), str)
        ).strip()
        if not answer:
            raise AIExtractionError("Gemini returned no text")
        return answer

    def structured_answer(self, instruction: str, context: str) -> dict:
        response = self.answer(
            instruction + JSON_SUFFIX,
            context,
        )
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(cleaned)
        except (json.JSONDecodeError, IndexError) as exc:
            raise AIExtractionError("AI returned invalid JSON") from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.openai_api_key:
            raise AIExtractionError("OPENAI_API_KEY is required for vector search")
        response = httpx.post(
            self.settings.embedding_endpoint,
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
            json={"model": self.settings.embedding_model, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]

    def _chat(
        self,
        provider: str,
        instruction: str,
        context: str,
        timeout_seconds: float | None = None,
    ) -> str:
        if provider == "gemini":
            return self._gemini_chat(instruction, context, timeout_seconds)
        config = {
            "github_models": (self.settings.github_token, self.settings.github_models_endpoint, self.settings.github_models_name),
            "groq": (self.settings.groq_api_key, self.settings.groq_endpoint, self.settings.groq_model),
            "openai": (self.settings.openai_api_key, self.settings.openai_endpoint, self.settings.openai_model),
        }.get(provider)
        if provider == "ollama":
            payload = {
                "model": self.settings.ollama_model,
                "keep_alive": self.settings.ollama_keep_alive,
                "stream": False,
                "options": {
                    "num_ctx": self.settings.ollama_context_tokens,
                    "num_predict": self.settings.ollama_max_output_tokens,
                },
                "messages": [
                    {"role": "system", "content": TUTOR_SYSTEM},
                    {"role": "user", "content": f"{instruction}\n\nStudy context:\n{context}"},
                ],
            }
            if "JSON only" in instruction:
                payload["format"] = "json"
            response = httpx.post(
                f"{self.settings.ollama_endpoint.rstrip('/')}/api/chat",
                json=payload,
                timeout=min(self.settings.ollama_timeout_seconds, timeout_seconds)
                if timeout_seconds is not None
                else self.settings.ollama_timeout_seconds,
            )
            response.raise_for_status()
            try:
                return response.json()["message"]["content"].strip()
            except (KeyError, TypeError) as exc:
                raise AIExtractionError("Ollama returned no text") from exc
        if not config or not config[0]:
            raise AIExtractionError("provider is not configured")
        response = httpx.post(
            config[1],
            headers={"Authorization": f"Bearer {config[0]}"},
            json={
                "model": config[2],
                "temperature": 0.2,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            TUTOR_SYSTEM
                        ),
                    },
                    {"role": "user", "content": f"{instruction}\n\nStudy context:\n{context}"},
                ],
            },
            timeout=timeout_seconds or 120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def _gemini_chat(
        self, instruction: str, context: str, timeout_seconds: float | None = None
    ) -> str:
        if not self.settings.gemini_api_key:
            raise AIExtractionError("GEMINI_API_KEY is not configured")
        model = self.settings.gemini_model.removeprefix("models/")
        generation_config = {"temperature": 0.2}
        if "JSON only" in instruction:
            generation_config["responseMimeType"] = "application/json"
        response = httpx.post(
            f"{self.settings.gemini_endpoint.rstrip('/')}/{model}:generateContent",
            params={"key": self.settings.gemini_api_key},
            json={
                "contents": [
                    {"parts": [{"text": f"{instruction}\n\nStudy context:\n{context}"}]}
                ],
                "generationConfig": generation_config,
            },
            timeout=httpx.Timeout(
                timeout_seconds or 120.0,
                connect=min(timeout_seconds or 10.0, 10.0),
            ),
        )
        response.raise_for_status()
        try:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise AIExtractionError("Gemini returned no text") from exc

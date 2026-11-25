import json

import httpx

from .config import Settings
from .prompts import EXTRACTION_SYSTEM, JSON_SUFFIX, TUTOR_SYSTEM, extraction_prompt
from .tracing import (
    token_usage,
    trace_generation,
    trace_operation,
    update_generation,
    update_operation,
)


class AIExtractionError(RuntimeError):
    pass


class AIProvider:
    def __init__(
        self,
        settings: Settings,
        *,
        session_id: str | None = None,
        feature: str = "ai",
    ):
        self.settings = settings
        self.session_id = session_id
        self.feature = feature

    def _redact_credentials(self, message: str) -> str:
        for credential in (
            self.settings.gemini_api_key,
            self.settings.github_token,
            self.settings.groq_api_key,
            self.settings.grok_api_key,
            self.settings.openai_api_key,
        ):
            if credential:
                message = message.replace(credential, "[REDACTED]")
        return message

    def extract(self, text: str, document_id: str) -> list[dict]:
        with trace_operation(
            "extract-concepts",
            feature=self.feature,
            input={"document_id": document_id, "text": text},
            metadata={"document_id": document_id, "provider_order": self.settings.providers},
            session_id=self.session_id,
        ) as observation:
            errors: list[str] = []
            for provider in self.settings.providers:
                try:
                    concepts = self._request(provider, text, document_id)
                    update_operation(
                        observation,
                        output={"concept_count": len(concepts)},
                        summary={"concept_count": len(concepts)},
                    )
                    return concepts
                except (httpx.HTTPError, ValueError, KeyError, AIExtractionError) as exc:
                    errors.append(f"{provider}: {self._redact_credentials(str(exc))}")
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
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        with trace_generation(
            "extract-concepts",
            provider=provider,
            model=config[2],
            input=messages,
            metadata={"document_id": document_id},
        ) as generation:
            response = httpx.post(
                config[1],
                headers={"Authorization": f"Bearer {config[0]}"},
                json={
                    "model": config[2],
                    "temperature": 0,
                    "messages": messages,
                },
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            concepts = json.loads(content)
            update_generation(
                generation,
                output=concepts,
                usage_details=token_usage(payload, provider),
            )
            return concepts

    def answer(
        self, instruction: str, context: str, timeout_seconds: float | None = None
    ) -> str:
        providers = self.settings.providers
        if timeout_seconds is not None:
            providers = providers[:1]
        with trace_operation(
            "generate-answer",
            feature=self.feature,
            input={"instruction": instruction, "context": context},
            metadata={"provider_order": providers},
            session_id=self.session_id,
        ) as observation:
            errors: list[str] = []
            for provider in providers:
                try:
                    answer = self._chat(provider, instruction, context, timeout_seconds)
                    update_operation(observation, output=answer)
                    return answer
                except (httpx.HTTPError, ValueError, KeyError, AIExtractionError) as exc:
                    errors.append(f"{provider}: {self._redact_credentials(str(exc))}")
            raise AIExtractionError("All AI providers failed: " + "; ".join(errors))

    def gemini_answer(self, instruction: str, context: str) -> str:
        if not self.settings.gemini_api_key:
            raise AIExtractionError("GEMINI_API_KEY is not configured")
        model = self.settings.gemini_model.removeprefix("models/")
        request = {
                "contents": [{"parts": [{"text": f"{instruction}\n\nStudy context:\n{context}"}]}],
                "generationConfig": {"temperature": 0.2},
            }
        with trace_operation(
            "generate-answer",
            feature=self.feature,
            input=request,
            metadata={"provider": "gemini", "model": model},
            session_id=self.session_id,
        ) as observation, trace_generation(
            "generate-answer",
            provider="gemini",
            model=model,
            input=request,
        ) as generation:
            response = httpx.post(
                f"{self.settings.gemini_endpoint.rstrip('/')}/{model}:generateContent",
                params={"key": self.settings.gemini_api_key},
                json=request,
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
            response.raise_for_status()
            payload = response.json()
            try:
                answer = payload["candidates"][0]["content"]["parts"][0]["text"].strip()
            except (KeyError, IndexError, TypeError) as exc:
                raise AIExtractionError("Gemini returned no text") from exc
            update_generation(
                generation,
                output=answer,
                usage_details=token_usage(payload, "gemini"),
            )
            update_operation(observation, output=answer)
            return answer

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
        request = {
                "model": model,
                "input": f"{instruction}\n\nStudy context:\n{context}",
                "tools": [{"type": "google_search"}],
            }
        with trace_operation(
            "search-grounded-answer",
            feature=self.feature,
            input=request,
            metadata={"provider": "gemini", "model": model},
            session_id=self.session_id,
        ) as observation, trace_generation(
            "search-grounded-answer",
            provider="gemini",
            model=model,
            input=request,
            metadata={"tools": ["google_search"]},
        ) as generation:
            response = httpx.post(
                endpoint,
                headers={
                    "x-goog-api-key": self.settings.gemini_api_key,
                    "Api-Revision": "2026-05-20",
                },
                json=request,
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
            update_generation(
                generation,
                output=answer,
                usage_details=token_usage(data, "gemini"),
            )
            update_operation(observation, output=answer)
            return answer

    def structured_answer(self, instruction: str, context: str) -> dict:
        with trace_operation(
            "generate-structured-answer",
            feature=self.feature,
            input={"instruction": instruction, "context": context},
            session_id=self.session_id,
        ) as observation:
            response = self.answer(
                instruction + JSON_SUFFIX,
                context,
            )
            try:
                cleaned = response.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
                result = json.loads(cleaned)
            except (json.JSONDecodeError, IndexError) as exc:
                raise AIExtractionError("AI returned invalid JSON") from exc
            update_operation(
                observation,
                output=result,
                summary={"status": "completed"},
            )
            return result

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.openai_api_key:
            raise AIExtractionError("OPENAI_API_KEY is required for vector search")
        request = {"model": self.settings.embedding_model, "input": texts}
        with trace_operation(
            "create-embeddings",
            feature=self.feature,
            input=texts,
            metadata={"model": self.settings.embedding_model, "text_count": len(texts)},
            session_id=self.session_id,
        ) as observation, trace_generation(
            "create-embeddings",
            as_type="embedding",
            provider="openai",
            model=self.settings.embedding_model,
            input=request,
            metadata={"text_count": len(texts)},
        ) as generation:
            response = httpx.post(
                self.settings.embedding_endpoint,
                headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                json=request,
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload["data"]
            embeddings = [
                item["embedding"] for item in sorted(data, key=lambda item: item["index"])
            ]
            update_generation(
                generation,
                output={"embedding_count": len(embeddings)},
                usage_details=token_usage(payload, "openai"),
            )
            update_operation(
                observation,
                output={"embedding_count": len(embeddings)},
                summary={"embedding_count": len(embeddings)},
            )
            return embeddings

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
            with trace_generation(
                "generate-answer",
                provider=provider,
                model=self.settings.ollama_model,
                input=payload["messages"],
            ) as generation:
                response = httpx.post(
                    f"{self.settings.ollama_endpoint.rstrip('/')}/api/chat",
                    json=payload,
                    timeout=min(self.settings.ollama_timeout_seconds, timeout_seconds)
                    if timeout_seconds is not None
                    else self.settings.ollama_timeout_seconds,
                )
                response.raise_for_status()
                result = response.json()
                try:
                    answer = result["message"]["content"].strip()
                except (KeyError, TypeError) as exc:
                    raise AIExtractionError("Ollama returned no text") from exc
                update_generation(
                    generation,
                    output=answer,
                    usage_details=token_usage(result, provider),
                )
                return answer
        if not config or not config[0]:
            raise AIExtractionError("provider is not configured")
        messages = [
            {"role": "system", "content": TUTOR_SYSTEM},
            {"role": "user", "content": f"{instruction}\n\nStudy context:\n{context}"},
        ]
        request = {
                "model": config[2],
                "temperature": 0.2,
                "messages": messages,
            }
        with trace_generation(
            "generate-answer",
            provider=provider,
            model=config[2],
            input=messages,
        ) as generation:
            response = httpx.post(
                config[1],
                headers={"Authorization": f"Bearer {config[0]}"},
                json=request,
                timeout=timeout_seconds or 120,
            )
            response.raise_for_status()
            payload = response.json()
            answer = payload["choices"][0]["message"]["content"].strip()
            update_generation(
                generation,
                output=answer,
                usage_details=token_usage(payload, provider),
            )
            return answer

    def _gemini_chat(
        self, instruction: str, context: str, timeout_seconds: float | None = None
    ) -> str:
        if not self.settings.gemini_api_key:
            raise AIExtractionError("GEMINI_API_KEY is not configured")
        model = self.settings.gemini_model.removeprefix("models/")
        generation_config = {"temperature": 0.2}
        if "JSON only" in instruction:
            generation_config["responseMimeType"] = "application/json"
        request = {
                "contents": [
                    {"parts": [{"text": f"{instruction}\n\nStudy context:\n{context}"}]}
                ],
                "generationConfig": generation_config,
            }
        with trace_generation(
            "generate-answer",
            provider="gemini",
            model=model,
            input=request,
        ) as generation:
            response = httpx.post(
                f"{self.settings.gemini_endpoint.rstrip('/')}/{model}:generateContent",
                params={"key": self.settings.gemini_api_key},
                json=request,
                timeout=httpx.Timeout(
                    timeout_seconds or 120.0,
                    connect=min(timeout_seconds or 10.0, 10.0),
                ),
            )
            response.raise_for_status()
            payload = response.json()
            try:
                answer = payload["candidates"][0]["content"]["parts"][0]["text"].strip()
            except (KeyError, IndexError, TypeError) as exc:
                raise AIExtractionError("Gemini returned no text") from exc
            update_generation(
                generation,
                output=answer,
                usage_details=token_usage(payload, "gemini"),
            )
            return answer

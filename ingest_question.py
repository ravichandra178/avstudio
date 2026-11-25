"""Generate textbook-grounded MCQs with the provider selected by USE_MODEL."""

import argparse
import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from threading import BoundedSemaphore, Event, Lock
from time import monotonic, perf_counter, sleep

import httpx
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError

from learning_studio.language import is_language_text
from learning_studio.prompts import entity_question_ingestion_prompt
from learning_studio.tracing import (
    flush_traces,
    token_usage,
    trace_generation,
    trace_operation,
    update_generation,
    update_operation,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest_question")
_grok_request_lock = Lock()
_grok_last_request = 0.0
_grok_quota_exhausted = Event()


class GrokDailyQuotaExceeded(RuntimeError):
    """Raised when the Groq daily token quota has ended this ingestion run."""

STRUCTURAL_LABELS = {
    "StudyGroup",
    "Textbook",
    "Document",
    "Chunk",
    "StudySession",
    "StudyMessage",
}

OLLAMA_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "options": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 4,
            "maxItems": 4,
        },
        "correct_option": {"type": "integer", "minimum": 0, "maximum": 3},
        "explanation": {"type": "string"},
        "question_te": {"type": "string"},
        "options_te": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 4,
            "maxItems": 4,
        },
        "explanation_te": {"type": "string"},
    },
    "required": [
        "question",
        "options",
        "correct_option",
        "explanation",
        "question_te",
        "options_te",
        "explanation_te",
    ],
    "additionalProperties": False,
}

OLLAMA_MODEL_ALIASES = {
    "qwen2.5-1.5b-instruct:local": "hf.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF:latest",
}


class IngestModel(str, Enum):
    GEMINI = "GEMINI"
    GROK = "GROK"
    OLLAMA = "OLLAMA"


def parse_ingest_model(value: str) -> IngestModel:
    try:
        return IngestModel(value.strip().upper())
    except ValueError as exc:
        supported = ", ".join(model.value for model in IngestModel)
        raise ValueError(f"USE_MODEL must be one of: {supported}") from exc


def parse_json(content: str) -> dict:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise TypeError("AI returned a non-object JSON value")
    options = value.get("options")
    options_te = value.get("options_te")
    correct_option = value.get("correct_option")
    if (
        not isinstance(value.get("question"), str)
        or not value["question"].strip()
        or not is_language_text(value["question"], "en")
        or not isinstance(options, list)
        or len(options) != 4
        or not all(
            isinstance(option, str)
            and option.strip()
            and is_language_text(option, "en")
            for option in options
        )
        or not isinstance(correct_option, int)
        or not 0 <= correct_option < 4
        or not isinstance(value.get("explanation", ""), str)
        or not is_language_text(value.get("explanation", ""), "en")
        or not isinstance(value.get("question_te"), str)
        or not value["question_te"].strip()
        or not isinstance(options_te, list)
        or len(options_te) != 4
        or not all(isinstance(option, str) and option.strip() for option in options_te)
        or not isinstance(value.get("explanation_te"), str)
    ):
        raise ValueError("AI returned an invalid four-option question")
    telugu_fields = [
        value["question_te"].strip(),
        *(option.strip() for option in options_te),
        value["explanation_te"].strip(),
    ]
    if not all(is_language_text(text, "te") for text in telugu_fields) or not any(
        "\u0c00" <= char <= "\u0c7f" for text in telugu_fields for char in text
    ):
        raise ValueError("AI returned an invalid Telugu question translation")
    return {
        "question": value["question"].strip(),
        "options": [option.strip() for option in options],
        "correct_option": correct_option,
        "explanation": str(value.get("explanation", "")).strip(),
        "question_te": value["question_te"].strip(),
        "options_te": [option.strip() for option in options_te],
        "explanation_te": value["explanation_te"].strip(),
    }


def textbook_name(properties: dict) -> str | None:
    for key in ("filename", "name", "source", "id"):
        value = properties.get(key)
        if isinstance(value, str) and value.lower().endswith(".pdf"):
            return value
    return None


def parse_toc_chapters(text: str) -> list[tuple[str, int]]:
    lines = text.splitlines()
    contents_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.search(r"\bcontents\b", line, re.IGNORECASE)
        ),
        None,
    )
    if contents_index is None:
        return []

    chapters = []
    expected_number = 1
    title_parts: list[str] = []
    active = False

    def save_chapter(page_number: int) -> None:
        title = " ".join(" ".join(title_parts).split())
        if title:
            chapters.append((title, page_number))

    for line in lines[contents_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue

        chapter_match = re.match(r"^(\d{1,2})(?:\s+|\t+)?(.*)$", stripped)
        if chapter_match and int(chapter_match.group(1)) == expected_number:
            if active:
                title_parts = []
            remainder = chapter_match.group(2).strip()
            title_parts = [remainder] if remainder else []
            active = True
            continue

        if not active:
            continue

        page_match = re.fullmatch(r"\d{1,3}", stripped)
        if page_match:
            save_chapter(int(stripped))
            title_parts = []
            expected_number += 1
            active = False
            continue

        title_parts.append(stripped)

    return chapters


def page_bounds(page: object) -> tuple[int, int] | None:
    match = re.match(r"^\s*(\d+)(?:\s*[-–]\s*(\d+))?", str(page or ""))
    if not match:
        return None
    start = int(match.group(1))
    return start, int(match.group(2) or start)


def normalized_chapter_heading(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.casefold()).split())


def chapter_source_text(
    documents: list[dict], chapter_name: str, next_chapter_name: str | None
) -> str:
    chapter_heading = normalized_chapter_heading(chapter_name)
    next_heading = (
        normalized_chapter_heading(next_chapter_name)
        if next_chapter_name
        else None
    )
    ordered_documents = []
    for document in documents:
        bounds = page_bounds(document.get("page"))
        text = document.get("text")
        if bounds and isinstance(text, str) and text.strip() and bounds[0] > 4:
            ordered_documents.append((bounds, text.strip()))
    ordered_documents.sort(key=lambda item: item[0][0])

    start_index = next(
        (
            index
            for index, (_, text) in enumerate(ordered_documents)
            if chapter_heading in normalized_chapter_heading(text)
        ),
        None,
    )
    if start_index is None:
        return ""
    end_index = len(ordered_documents)
    if next_heading:
        end_index = next(
            (
                index
                for index in range(start_index + 1, len(ordered_documents))
                if next_heading
                in normalized_chapter_heading(ordered_documents[index][1])
            ),
            end_index,
        )

    passages = []
    remaining = 6000
    for (page_start, page_end), text in ordered_documents[start_index:end_index]:
        page_label = f"[Pages {page_start}-{page_end}]\n"
        excerpt = text[: max(0, remaining - len(page_label))]
        passages.append(f"{page_label}{excerpt}")
        remaining -= len(page_label) + len(excerpt)
        if remaining <= 0:
            break
    return "\n\n".join(passages)


def load_entities(driver, database: str, selected_textbook: str | None) -> list[dict]:
    started = perf_counter()
    logger.info(
        "QUESTION_INGEST_ENTITY_LOAD_START database=%s textbook=%s",
        database,
        selected_textbook or "all",
    )
    with driver.session(database=database) as session:
        result = session.run(
            """
            MATCH (d:Document)-[:MENTIONS]->(entity)
            WITH d, entity, properties(d) AS document_properties,
                 properties(entity) AS entity_properties
            WITH d, entity, document_properties,
                 coalesce(
                     entity_properties.name,
                     entity_properties.title,
                     entity_properties.text,
                     entity_properties.id
                 ) AS entity_name
            WHERE (
                document_properties.filename = $textbook
                OR document_properties.name = $textbook
                OR document_properties.source = $textbook
                OR document_properties.id = $textbook
                OR $textbook IS NULL
            )
              AND entity_name IS NOT NULL
            OPTIONAL MATCH (entity)-[relation]-(related)
            WITH d, entity, entity_name, labels(entity)[0] AS entity_type,
                 collect(DISTINCT {
                     relationship: type(relation),
                     name: coalesce(
                         properties(related).name,
                         properties(related).title,
                         properties(related).text,
                         properties(related).id
                     )
                 })[..12] AS relationships
            RETURN properties(d) AS document_properties,
                   entity_type, entity_name, relationships
            ORDER BY document_properties.name, entity_type, entity_name
            """,
            textbook=selected_textbook,
        )
        entities = []
        seen: set[tuple[str, str, str]] = set()
        for record in result:
            data = record.data()
            name = data["entity_name"]
            entity_type = data["entity_type"] or "Entity"
            if entity_type in STRUCTURAL_LABELS or re.fullmatch(r"[0-9a-fA-F]{32,}", name):
                continue
            document = textbook_name(data["document_properties"])
            if not document:
                continue
            key = (document, entity_type, name)
            if key not in seen:
                seen.add(key)
                entities.append(
                    {
                        "textbook": document,
                        "entity_type": entity_type,
                        "entity_name": name,
                        "relationships": [
                            item for item in data["relationships"] if item.get("name")
                        ],
                    }
                )

        document_result = session.run(
            """
            MATCH (d:Document)
            WITH properties(d) AS document_properties
            WHERE (
                document_properties.filename = $textbook
                OR document_properties.name = $textbook
                OR document_properties.source = $textbook
                OR document_properties.id = $textbook
                OR $textbook IS NULL
            )
            RETURN document_properties
            ORDER BY document_properties.page
            """,
            textbook=selected_textbook,
        )
        documents_by_textbook: dict[str, list[dict]] = {}
        for record in document_result:
            properties = record.data()["document_properties"]
            document = textbook_name(properties)
            if document:
                documents_by_textbook.setdefault(document, []).append(properties)

        entities_by_key = {
            (entity["textbook"], entity["entity_type"], entity["entity_name"]): entity
            for entity in entities
        }
        chapter_count = 0
        discovered_chapter_count = 0
        missing_context_count = 0
        for document, page_documents in documents_by_textbook.items():
            toc_documents = [
                item
                for item in page_documents
                if (bounds := page_bounds(item.get("page")))
                and bounds[0] <= 4
                and isinstance(item.get("text"), str)
                and re.search(r"\bcontents\b", item["text"], re.IGNORECASE)
            ]
            chapters = [
                chapter
                for item in toc_documents
                for chapter in parse_toc_chapters(item["text"])
            ]
            discovered_chapter_count += len(chapters)
            for index, (chapter_name, start_page) in enumerate(chapters):
                next_chapter_name = (
                    chapters[index + 1][0] if index + 1 < len(chapters) else None
                )
                source_text = chapter_source_text(
                    page_documents, chapter_name, next_chapter_name
                )
                if not source_text:
                    logger.warning(
                        "QUESTION_INGEST_CHAPTER_CONTEXT_MISSING textbook=%s "
                        "chapter=%s page=%d",
                        document,
                        chapter_name,
                        start_page,
                    )
                    missing_context_count += 1
                    continue
                key = (document, "Topic", chapter_name)
                entity = entities_by_key.get(key)
                if entity is None:
                    entity = {
                        "textbook": document,
                        "entity_type": "Topic",
                        "entity_name": chapter_name,
                        "relationships": [],
                    }
                    entities_by_key[key] = entity
                    entities.append(entity)
                entity["source_text"] = source_text
                entity["chapter_page"] = start_page
                chapter_count += 1
        logger.info(
            "QUESTION_INGEST_CHAPTERS_LOADED discovered=%d with_source_text=%d "
            "missing_source_text=%d textbooks=%d",
            discovered_chapter_count,
            chapter_count,
            missing_context_count,
            len(documents_by_textbook),
        )
        logger.info(
            "QUESTION_INGEST_ENTITY_LOAD_FINISHED count=%d duration_seconds=%.3f",
            len(entities),
            perf_counter() - started,
        )
        return entities


def question_prompt(settings: dict, entity: dict) -> str:
    relationship_text = "; ".join(
        f"{item['relationship']} -> {item['name']}"
        for item in entity["relationships"]
    ) or "No additional direct relationship facts."
    prompt = entity_question_ingestion_prompt(
        entity["entity_type"],
        entity["entity_name"],
        relationship_text,
        settings["group"],
        settings["level"],
    )
    source_text = entity.get("source_text")
    if source_text:
        prompt += (
            f"\n\nTextbook source passage (pages {entity.get('chapter_page', 'unknown')} onward):\n"
            f"{source_text}\n"
            "Ground the question and its correct answer specifically in this passage."
        )
    previous_questions = entity.get("previous_questions", [])
    if previous_questions:
        prompt += (
            "\n\nGenerate a new, non-duplicate question. Do not reuse or paraphrase "
            "these existing questions:\n"
            + "\n".join(f"- {question}" for question in previous_questions)
        )
    question_number = entity.get("question_number")
    if question_number:
        prompt += (
            f"\nThis is question {question_number} of {entity.get('questions_target', question_number)} "
            "in a set. Use a distinct "
            "textbook-supported angle or question format from the others."
        )
    variation_attempt = entity.get("variation_attempt")
    if variation_attempt and variation_attempt > 1:
        prompt += (
            f"\nThis is variation attempt {variation_attempt}; choose a different "
            "question angle and wording than the prior response."
        )
    return prompt


def grok_error_detail(exc: Exception, api_key: str) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        response_body = " ".join(exc.response.text.split())
        if api_key:
            response_body = response_body.replace(api_key, "[REDACTED]")
        return (
            f"http_status={exc.response.status_code} "
            f"response_body={response_body[:1000]!r}"
        )
    return f"error_type={type(exc).__name__} error={str(exc)[:1000]!r}"


def grok_rate_limit_delay(exc: httpx.HTTPStatusError, attempt: int) -> float | None:
    if exc.response.status_code != 429:
        return None

    try:
        payload = exc.response.json()
    except ValueError:
        payload = {}
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    message = error.get("message", "") if isinstance(error, dict) else ""
    message = str(message)
    if re.search(r"tokens?\s+per\s+day|\bTPD\b|daily\s+tokens?", message, re.IGNORECASE):
        return None

    retry_after = exc.response.headers.get("Retry-After", "").strip()
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass

    match = re.search(
        r"try again in\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b",
        message,
        re.IGNORECASE,
    )
    if match:
        return float(match.group(1))
    return float(15 * attempt)


def gemini_retry_delay(exc: httpx.HTTPStatusError, attempt: int) -> float | None:
    if exc.response.status_code != 429:
        return None

    body = exc.response.text
    if re.search(r"per.?day|daily|tokens per day|requests per day", body, re.IGNORECASE):
        return None

    retry_after = exc.response.headers.get("Retry-After", "").strip()
    if retry_after:
        try:
            return min(30.0, max(0.0, float(retry_after)))
        except ValueError:
            pass

    try:
        payload = exc.response.json()
    except ValueError:
        payload = {}
    details = payload.get("error", {}).get("details", [])
    for detail in details if isinstance(details, list) else []:
        retry_delay = detail.get("retryDelay") if isinstance(detail, dict) else None
        if isinstance(retry_delay, str):
            match = re.fullmatch(r"(\d+(?:\.\d+)?)s", retry_delay)
            if match:
                return min(30.0, float(match.group(1)))
    return min(15.0, float(2**attempt))


def wait_for_grok_request(settings: dict) -> None:
    global _grok_last_request
    with _grok_request_lock:
        if _grok_quota_exhausted.is_set():
            raise GrokDailyQuotaExceeded(
                "Groq daily token quota exhausted for this ingestion run"
            )
        interval = max(0.0, float(settings.get("grok_request_interval", 0)))
        remaining = interval - (monotonic() - _grok_last_request)
        if remaining > 0:
            logger.info("QUESTION_INGEST_GROK_PACING wait_seconds=%.1f", remaining)
            sleep(remaining)
        if _grok_quota_exhausted.is_set():
            raise GrokDailyQuotaExceeded(
                "Groq daily token quota exhausted for this ingestion run"
            )
        _grok_last_request = monotonic()


def check_grok_credentials(settings: dict, client: httpx.Client) -> None:
    api_key = settings["grok_api_key"]
    model = settings["grok_model"]
    if not api_key or not model:
        raise RuntimeError("GROK_API_KEY and GROK_MODEL are required for question ingestion")

    endpoint = settings["grok_endpoint"].rstrip("/")
    models_endpoint = (
        endpoint.removesuffix("/chat/completions") + "/models"
        if endpoint.endswith("/chat/completions")
        else endpoint + "/models"
    )
    started = perf_counter()
    logger.info(
        "QUESTION_INGEST_CREDENTIAL_CHECK_START provider=groq model=%s",
        model,
    )
    try:
        response = client.get(
            models_endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=settings["grok_timeout"],
        )
        response.raise_for_status()
        payload = response.json()
        model_entries = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(model_entries, list):
            raise TypeError("Grok models response did not contain a data list")
        available_models = {
            entry.get("id") for entry in model_entries if isinstance(entry, dict)
        }
        if model not in available_models:
            raise RuntimeError(
                f"Configured GROK_MODEL {model!r} is not available to this API key"
            )
    except (
        httpx.HTTPError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        logger.error(
            "QUESTION_INGEST_CREDENTIAL_CHECK_FAILED provider=groq model=%s "
            "duration_seconds=%.3f %s",
            model,
            perf_counter() - started,
            grok_error_detail(exc, api_key),
        )
        raise RuntimeError("Grok credentials or configured model failed preflight") from exc

    logger.info(
        "QUESTION_INGEST_CREDENTIAL_CHECK_SUCCESS provider=groq model=%s "
        "duration_seconds=%.3f",
        model,
        perf_counter() - started,
    )


def _generate_grok_question(
    settings: dict,
    entity: dict,
    client: httpx.Client,
    attempt: int = 1,
    retry_instruction: str = "",
) -> dict:
    api_key = settings["grok_api_key"]
    if not api_key:
        raise RuntimeError("GROK_API_KEY is required for question ingestion")

    wait_for_grok_request(settings)
    started = perf_counter()
    logger.info(
        "QUESTION_INGEST_PROVIDER_START name=%s provider=groq model=%s timeout_seconds=%s",
        entity["entity_name"],
        settings["grok_model"],
        settings["grok_timeout"],
    )
    try:
        request = {
                "model": settings["grok_model"],
                "temperature": 0,
                "max_tokens": settings["grok_max_output_tokens"],
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return one concise, valid JSON object only. Do not include "
                            "markdown or text outside the object."
                        ),
                    },
                    {
                        "role": "user",
                        "content": question_prompt(settings, entity)
                        + (f"\n\nCorrection for this retry: {retry_instruction}" if retry_instruction else ""),
                    },
                ],
            }
        with trace_generation(
            "generate-textbook-question",
            provider="groq",
            model=settings["grok_model"],
            input=request["messages"],
            metadata={"entity_type": entity["entity_type"]},
        ) as generation:
            response = client.post(
                settings["grok_endpoint"],
                headers={"Authorization": f"Bearer {api_key}"},
                json=request,
                timeout=settings["grok_timeout"],
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            question = {**parse_json(content), "provider": "groq"}
            update_generation(
                generation,
                output=question,
                usage_details=token_usage(payload, "groq"),
            )
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        if (
            isinstance(exc, httpx.HTTPStatusError)
            and exc.response.status_code == 429
            and re.search(
                r"tokens?\s+per\s+day|\bTPD\b|daily\s+tokens?",
                exc.response.text,
                re.IGNORECASE,
            )
        ):
            _grok_quota_exhausted.set()
        max_attempts = max(1, int(settings.get("grok_generation_attempts", 3)))
        if isinstance(exc, httpx.HTTPStatusError):
            delay = grok_rate_limit_delay(exc, attempt)
            if delay is not None and attempt < max_attempts:
                logger.warning(
                    "QUESTION_INGEST_GROK_RATE_LIMIT_RETRY name=%s attempt=%d "
                    "wait_seconds=%.1f",
                    entity["entity_name"],
                    attempt,
                    delay,
                )
                sleep(delay)
                return _generate_grok_question(
                    settings,
                    entity,
                    client,
                    attempt + 1,
                )
        elif isinstance(exc, ValueError) and attempt < max_attempts:
            logger.warning(
                "QUESTION_INGEST_GROK_INVALID_OUTPUT_RETRY name=%s attempt=%d",
                entity["entity_name"],
                attempt,
            )
            return _generate_grok_question(
                settings,
                entity,
                client,
                attempt + 1,
                (
                    "The prior response failed validation: "
                    f"{exc}. Return accurate Telugu-script translations in every "
                    "field ending in _te; do not put English sentences there."
                ),
            )
        logger.error(
            "QUESTION_INGEST_PROVIDER_FAILED name=%s provider=groq duration_seconds=%.3f %s",
            entity["entity_name"],
            perf_counter() - started,
            grok_error_detail(exc, api_key),
        )
        raise

    logger.info(
        "QUESTION_INGEST_PROVIDER_SUCCESS name=%s provider=groq duration_seconds=%.3f",
        entity["entity_name"],
        perf_counter() - started,
    )
    return question


def _generate_question(settings: dict, entity: dict, client: httpx.Client) -> dict:
    use_model = settings.get("use_model", IngestModel.OLLAMA)
    if not isinstance(use_model, IngestModel):
        use_model = parse_ingest_model(str(use_model))

    prompt = question_prompt(settings, entity)
    if use_model is IngestModel.GROK:
        if not settings.get("grok_api_key") or not settings.get("grok_model"):
            raise RuntimeError(
                "GROK_API_KEY and GROK_MODEL are required for question ingestion"
            )
        return _generate_grok_question(settings, entity, client)

    if use_model is IngestModel.GEMINI:
        api_key = settings.get("gemini_api_key", "")
        model = settings.get("gemini_model", "")
        if not api_key or not model:
            raise RuntimeError(
                "GEMINI_API_KEY and GEMINI_MODEL are required for question ingestion"
            )
        provider = "gemini"
        timeout = settings["gemini_timeout"]
        started = perf_counter()
        logger.info(
            "QUESTION_INGEST_PROVIDER_START name=%s provider=%s model=%s timeout_seconds=%s",
            entity["entity_name"],
            provider,
            model,
            timeout,
        )
        try:
            request = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": settings["gemini_max_output_tokens"],
                    "responseMimeType": "application/json",
                },
            }
            with trace_generation(
                "generate-textbook-question",
                provider=provider,
                model=model,
                input=request,
                metadata={"entity_type": entity["entity_type"]},
            ) as generation:
                for attempt in range(1, 3):
                    try:
                        response = client.post(
                            f"{settings['gemini_endpoint'].rstrip('/')}/"
                            f"{model.removeprefix('models/')}:generateContent",
                            headers={"x-goog-api-key": api_key},
                            json=request,
                            timeout=timeout,
                        )
                        response.raise_for_status()
                        break
                    except httpx.HTTPStatusError as exc:
                        delay = gemini_retry_delay(exc, attempt)
                        if delay is None or attempt == 2:
                            raise
                        logger.warning(
                            "QUESTION_INGEST_GEMINI_RETRY status=%d attempt=%d "
                            "wait_seconds=%.1f",
                            exc.response.status_code,
                            attempt,
                            delay,
                        )
                        sleep(delay)
                    except httpx.TimeoutException:
                        raise
                payload = response.json()
                content = payload["candidates"][0]["content"]["parts"][0]["text"]
                question = {**parse_json(content), "provider": provider}
                update_generation(
                    generation,
                    output=question,
                    usage_details=token_usage(payload, "gemini"),
                )
        except (
            httpx.HTTPError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:
            logger.error(
                "QUESTION_INGEST_PROVIDER_FAILED name=%s provider=%s "
                "duration_seconds=%.3f error_type=%s error=%s",
                entity["entity_name"],
                provider,
                perf_counter() - started,
                type(exc).__name__,
                str(exc)[:1000],
            )
            raise
        logger.info(
            "QUESTION_INGEST_PROVIDER_SUCCESS name=%s provider=%s duration_seconds=%.3f",
            entity["entity_name"],
            provider,
            perf_counter() - started,
        )
        return question

    model = settings.get("ollama_model", "")
    if not model:
        raise RuntimeError("OLLAMA_MODEL is required for question ingestion")
    ollama_model = OLLAMA_MODEL_ALIASES.get(model, model)
    if ollama_model != model:
        logger.info(
            "QUESTION_INGEST_OLLAMA_MODEL_RESOLVED configured_model=%s local_model=%s",
            model,
            ollama_model,
        )

    provider = "ollama"
    timeout = settings["timeout"]
    started = perf_counter()
    logger.info(
        "QUESTION_INGEST_PROVIDER_START name=%s provider=%s model=%s timeout_seconds=%s",
        entity["entity_name"],
        provider,
        ollama_model,
        timeout,
    )
    for attempt in range(2):
        output_tokens = settings["max_output_tokens"]
        attempt_prompt = prompt
        if attempt:
            output_tokens = max(1024, output_tokens * 2)
            attempt_prompt += (
                "\n\nRETRY: Your previous response was invalid. Return one short, complete "
                "JSON object only. Keep the question to one sentence, each option under "
                "10 words, and the explanation under 20 words. Do not use markdown or "
                "include text outside the JSON object."
            )
        result = None
        output_content = ""
        try:
            request = {
                    "model": ollama_model,
                    "stream": False,
                    "keep_alive": settings["keep_alive"],
                    "format": OLLAMA_QUESTION_SCHEMA,
                    "options": {
                        "temperature": 0,
                        "num_ctx": settings["context_tokens"],
                        "num_predict": output_tokens,
                    },
                    "messages": [
                        {"role": "system", "content": "Return only valid JSON."},
                        {"role": "user", "content": attempt_prompt},
                    ],
                }
            with trace_generation(
                "generate-textbook-question",
                provider=provider,
                model=ollama_model,
                input=request["messages"],
                metadata={
                    "entity_type": entity["entity_type"],
                    "attempt": attempt + 1,
                },
            ) as generation:
                response = client.post(
                    f"{settings['ollama_endpoint'].rstrip('/')}/api/chat",
                    json=request,
                    timeout=timeout,
                )
                response.raise_for_status()
                result = response.json()
                message = result.get("message", {})
                output_content = (
                    message.get("content", "") if isinstance(message, dict) else ""
                )
                question = {**parse_json(output_content), "provider": provider}
                update_generation(
                    generation,
                    output=question,
                    usage_details=token_usage(result, "ollama"),
                )
        except httpx.HTTPError:
            logger.exception(
                "QUESTION_INGEST_PROVIDER_FAILED name=%s provider=%s attempt=%d "
                "duration_seconds=%.3f",
                entity["entity_name"],
                provider,
                attempt + 1,
                perf_counter() - started,
            )
            raise
        except (KeyError, TypeError, ValueError) as exc:
            if attempt == 0:
                logger.warning(
                    "QUESTION_INGEST_OUTPUT_INVALID name=%s provider=%s attempt=1 "
                    "error_type=%s done_reason=%s eval_count=%s response_chars=%s",
                    entity["entity_name"],
                    provider,
                    type(exc).__name__,
                    (result or {}).get("done_reason", "unknown"),
                    (result or {}).get("eval_count", "unknown"),
                    len(output_content),
                )
                continue
            logger.error(
                "QUESTION_INGEST_PROVIDER_FAILED name=%s provider=%s attempt=2 "
                "duration_seconds=%.3f error_type=%s",
                entity["entity_name"],
                provider,
                perf_counter() - started,
                type(exc).__name__,
            )
            raise
        break
    logger.info(
        "QUESTION_INGEST_PROVIDER_SUCCESS name=%s provider=%s attempts=%d "
        "duration_seconds=%.3f",
        entity["entity_name"],
        provider,
        attempt + 1,
        perf_counter() - started,
    )
    return question


def generate_question(
    settings: dict,
    entity: dict,
    client: httpx.Client,
    gemini_unavailable: Event | None = None,
    ollama_semaphore: BoundedSemaphore | None = None,
) -> dict:
    if gemini_unavailable is None:
        gemini_unavailable = Event()

    with trace_operation(
        "generate-textbook-question",
        feature="question-ingestion",
        input=question_prompt(settings, entity),
        metadata={"entity_type": entity["entity_type"]},
    ) as observation:
        use_model = settings.get("use_model", IngestModel.GEMINI)
        if not isinstance(use_model, IngestModel):
            use_model = parse_ingest_model(str(use_model))

        def generate_ollama() -> dict:
            if ollama_semaphore is None:
                return _generate_question(
                    settings | {"use_model": IngestModel.OLLAMA}, entity, client
                )
            queued_at = monotonic()
            ollama_semaphore.acquire()
            queue_wait = monotonic() - queued_at
            request_started = perf_counter()
            if queue_wait > 0:
                logger.info(
                    "QUESTION_INGEST_OLLAMA_QUEUE name=%s queue_wait_seconds=%.3f",
                    entity["entity_name"],
                    queue_wait,
                )
            try:
                return _generate_question(
                    settings | {"use_model": IngestModel.OLLAMA}, entity, client
                )
            finally:
                logger.info(
                    "QUESTION_INGEST_OLLAMA_REQUEST name=%s request_duration_seconds=%.3f",
                    entity["entity_name"],
                    perf_counter() - request_started,
                )
                ollama_semaphore.release()

        question = None
        gemini_error = None
        if use_model is IngestModel.GEMINI and not gemini_unavailable.is_set():
            try:
                question = _generate_question(
                    settings | {"use_model": IngestModel.GEMINI}, entity, client
                )
            except (
                httpx.HTTPError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as exc:
                gemini_error = exc
                gemini_unavailable.set()
                logger.error(
                    "QUESTION_INGEST_GEMINI_FAILED name=%s error_type=%s; "
                    "disabling Gemini for this run",
                    entity["entity_name"],
                    type(exc).__name__,
                )
        elif use_model is IngestModel.GEMINI:
            logger.info(
                "QUESTION_INGEST_GEMINI_SKIPPED name=%s reason=provider_unavailable",
                entity["entity_name"],
            )

        if use_model is IngestModel.GROK:
            if _grok_quota_exhausted.is_set():
                raise GrokDailyQuotaExceeded(
                    "Groq daily token quota exhausted for this ingestion run"
                )
            question = _generate_question(
                settings | {"use_model": IngestModel.GROK}, entity, client
            )

        if question is None and use_model is IngestModel.GEMINI:
            if not settings.get("grok_api_key"):
                message = (
                    "Gemini generation failed and GROK_API_KEY is not configured; "
                    "Ollama fallback is disabled"
                )
                logger.info(
                    "QUESTION_INGEST_GROK_SKIPPED name=%s reason=GROK_API_KEY_missing; "
                    "Ollama fallback disabled",
                    entity["entity_name"],
                )
                raise RuntimeError(message) from gemini_error
            logger.info(
                "QUESTION_INGEST_FALLBACK name=%s provider=groq reason=gemini_failed",
                entity["entity_name"],
            )
            try:
                question = _generate_question(
                    settings | {"use_model": IngestModel.GROK}, entity, client
                )
            except GrokDailyQuotaExceeded:
                raise
            except (
                httpx.HTTPError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as exc:
                if _grok_quota_exhausted.is_set():
                    raise GrokDailyQuotaExceeded(
                        "Groq daily token quota exhausted for this ingestion run"
                    ) from exc
                logger.exception(
                    "QUESTION_INGEST_GROK_FAILED name=%s error_type=%s; "
                    "Ollama fallback disabled",
                    entity["entity_name"],
                    type(exc).__name__,
                )
                raise RuntimeError(
                    "Gemini and Groq generation failed; Ollama fallback is disabled"
                ) from exc

        if question is None:
            if use_model is IngestModel.OLLAMA:
                logger.info(
                    "QUESTION_INGEST_PROVIDER_SELECTED provider=ollama "
                    "reason=USE_MODEL_OLLAMA"
                )
                question = generate_ollama()
            else:
                question = _generate_question(
                    settings | {"use_model": use_model}, entity, client
                )
        update_operation(
            observation,
            output=question,
            summary={
                "provider": question["provider"],
                "question_characters": len(question["question"]),
            },
        )
        return question


def question_id_for_entity(entity: dict) -> str:
    source_id = hashlib.sha256(
        f"{entity['textbook']}|{entity['entity_type']}|{entity['entity_name']}".encode()
    ).hexdigest()
    return f"textbook-{source_id[:24]}"


def normalized_question(question: str) -> str:
    return " ".join(re.findall(r"\w+", question.casefold()))


def question_id_for_text(entity: dict, question: str) -> str:
    identity = (
        f"{entity['textbook']}|{entity['entity_type']}|{entity['entity_name']}|"
        f"{normalized_question(question)}"
    )
    return f"textbook-{hashlib.sha256(identity.encode()).hexdigest()[:32]}"


def generate_questions_for_entity(
    settings: dict,
    entity: dict,
    existing_questions: list[str],
    count: int,
    client: httpx.Client,
    gemini_unavailable: Event | None = None,
    ollama_semaphore: BoundedSemaphore | None = None,
) -> tuple[list[dict], int]:
    questions = []
    prior_questions = list(existing_questions)
    signatures = {normalized_question(question) for question in prior_questions}
    failed = 0

    for _ in range(count):
        question = None
        generation_failed = False
        for duplicate_attempt in range(1, 4):
            generation_entity = entity | {
                "previous_questions": prior_questions,
                "question_number": len(prior_questions) + 1,
                "questions_target": len(signatures) + count,
                "variation_attempt": duplicate_attempt,
            }
            try:
                if gemini_unavailable is None and ollama_semaphore is None:
                    candidate = generate_question(
                        settings, generation_entity, client
                    )
                else:
                    candidate = generate_question(
                        settings,
                        generation_entity,
                        client,
                        gemini_unavailable,
                        ollama_semaphore,
                    )
            except Exception as exc:
                if _grok_quota_exhausted.is_set():
                    remaining = count - len(questions) - failed
                    failed += remaining
                    logger.error(
                        "QUESTION_INGEST_DAILY_QUOTA_STOP entity=%s remaining_questions=%d",
                        entity["entity_name"],
                        remaining,
                    )
                    return questions, failed
                logger.exception(
                    "QUESTION_INGEST_QUESTION_FAILED entity=%s question_number=%d",
                    entity["entity_name"],
                    len(prior_questions) + 1,
                )
                failed += 1
                generation_failed = True
                if (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code == 429
                    and grok_rate_limit_delay(exc, 1) is None
                ):
                    _grok_quota_exhausted.set()
                    remaining = count - len(questions) - failed
                    failed += remaining
                    return questions, failed
                break

            signature = normalized_question(candidate["question"])
            if signature not in signatures:
                candidate["question_id"] = question_id_for_text(
                    entity,
                    candidate["question"],
                )
                signatures.add(signature)
                prior_questions.append(candidate["question"])
                questions.append(candidate)
                question = candidate
                break

            logger.warning(
                "QUESTION_INGEST_DUPLICATE_RETRY entity=%s question_number=%d "
                "attempt=%d",
                entity["entity_name"],
                len(prior_questions) + 1,
                duplicate_attempt,
            )
        if question is None and not generation_failed:
            failed += 1
            logger.error(
                "QUESTION_INGEST_DUPLICATE_EXHAUSTED entity=%s question_number=%d",
                entity["entity_name"],
                len(prior_questions) + 1,
            )
    return questions, failed


def question_id_candidates(question_id: str) -> list[str]:
    clean_id = question_id.removeprefix("textbook-")
    return list(
        {
            question_id,
            f"textbook-{clean_id}",
            clean_id,
            f"textbook-{clean_id[:24]}",
            clean_id[:24],
        }
    )


def existing_question_ids(driver, database: str, question_ids: list[str]) -> set[str]:
    started = perf_counter()
    candidates = sorted(
        {
            candidate
            for question_id in question_ids
            for candidate in question_id_candidates(question_id)
        }
    )
    if not candidates:
        logger.info(
            "QUESTION_INGEST_EXISTING_CHECK_FINISHED ids=0 existing=0 duration_seconds=%.3f",
            perf_counter() - started,
        )
        return set()

    def _check(tx):
        result = tx.run(
            """
            MATCH (q:StudyQuestion)
            WHERE q.id IN $candidate_ids
            RETURN q.id AS id
            """,
            candidate_ids=candidates,
        )
        return {record["id"] for record in result}

    with driver.session(database=database) as session:
        result = session.execute_read(_check)
    logger.info(
        "QUESTION_INGEST_EXISTING_CHECK_FINISHED ids=%d existing=%d duration_seconds=%.3f",
        len(question_ids),
        len(result),
        perf_counter() - started,
    )
    return result


def existing_questions_by_entity(
    driver,
    database: str,
    entities: list[dict],
) -> dict[tuple[str, str, str], list[str]]:
    keys = {
        (entity["textbook"], entity["entity_type"], entity["entity_name"])
        for entity in entities
    }
    if not keys:
        return {}
    entity_keys = [
        {"textbook": textbook, "entity_type": entity_type, "entity_name": entity_name}
        for textbook, entity_type, entity_name in sorted(keys)
    ]

    def _load(tx):
        result = tx.run(
            """
            UNWIND $entities AS entity
            MATCH (q:StudyQuestion)
            WHERE q.textbook = entity.textbook
              AND q.entity_type = entity.entity_type
              AND q.entity_name = entity.entity_name
            RETURN entity.textbook AS textbook,
                   entity.entity_type AS entity_type,
                   entity.entity_name AS entity_name,
                   coalesce(q.question_en, q.question) AS question
            """,
            entities=entity_keys,
        )
        return [record.data() for record in result]

    with driver.session(database=database) as session:
        records = session.execute_read(_load)
    questions: dict[tuple[str, str, str], list[str]] = {}
    for record in records:
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            continue
        key = (record["textbook"], record["entity_type"], record["entity_name"])
        questions.setdefault(key, []).append(question)
    return questions


def ensure_question_schema(driver, database: str) -> None:
    started = perf_counter()
    logger.info("QUESTION_INGEST_SCHEMA_START database=%s", database)
    def _create_index(tx):
        tx.run(
            "CREATE INDEX study_question_id_index IF NOT EXISTS "
            "FOR (q:StudyQuestion) ON (q.id)"
        ).consume()

    try:
        with driver.session(database=database) as session:
            session.execute_write(_create_index)
        logger.info(
            "QUESTION_INGEST_SCHEMA_FINISHED database=%s duration_seconds=%.3f",
            database,
            perf_counter() - started,
        )
    except (DriverError, Neo4jError) as exc:
        logger.warning(
            "QUESTION_INGEST_SCHEMA_FAILED database=%s duration_seconds=%.3f error_type=%s error=%s",
            database,
            perf_counter() - started,
            type(exc).__name__,
            exc,
        )


def store_question(driver, database: str, entity: dict, question: dict) -> str:
    return store_questions(driver, database, [(entity, question)])[0]


def store_entity_questions(
    driver, database: str, entity: dict, questions: list[dict]
) -> tuple[int, int]:
    if not questions:
        return 0, 0

    try:
        stored_ids = store_questions(
            driver, database, [(entity, question) for question in questions]
        )
    except (DriverError, Neo4jError):
        logger.exception(
            "QUESTION_INGEST_ENTITY_STORE_BATCH_FAILED name=%s count=%d",
            entity["entity_name"],
            len(questions),
        )
        stored_count = 0
        failed_count = 0
        for question in questions:
            try:
                store_question(driver, database, entity, question)
            except (DriverError, Neo4jError) as exc:
                failed_count += 1
                logger.exception(
                    "QUESTION_INGEST_STORE_FAILED name=%s error_type=%s",
                    entity["entity_name"],
                    type(exc).__name__,
                )
            else:
                stored_count += 1
                logger.info(
                    "QUESTION_INGEST_STORED id=%s provider=%s",
                    entity["entity_name"],
                    question["provider"],
                )
        return stored_count, failed_count

    for question in questions:
        logger.info(
            "QUESTION_INGEST_STORED id=%s provider=%s",
            entity["entity_name"],
            question["provider"],
        )
    return len(stored_ids), 0


def store_questions(
    driver, database: str, items: list[tuple[dict, dict]]
) -> list[str]:
    if not items:
        return []
    started = perf_counter()
    records = []
    for entity, question in items:
        question_id = question.get("question_id") or question_id_for_entity(entity)
        citations = json.dumps(
            [
                {
                    "concept": entity["entity_name"],
                    "entity_type": entity["entity_type"],
                    "source_document_id": entity["textbook"],
                    "related_concepts": entity["relationships"],
                }
            ]
        )
        answer = (
            f"{question['options'][question['correct_option']]}. "
            f"{question['explanation']}"
        ).strip()
        records.append(
            {
                "question_id": question_id,
                "question": question["question"],
                "question_te": question.get("question_te"),
                "options": question["options"],
                "options_te": question.get("options_te"),
                "correct_option": question["correct_option"],
                "answer": answer,
                "answer_te": (
                    f"{question['options_te'][question['correct_option']]}. "
                    f"{question.get('explanation_te', '')}"
                ).strip()
                if question.get("options_te")
                else None,
                "explanation": question["explanation"],
                "explanation_te": question.get("explanation_te"),
                "level": os.environ.get(
                    "QUESTION_LEVEL", "AP Telangana Group 1 screening exam"
                ),
                "textbook": entity["textbook"],
                "citations": citations,
                "entity_type": entity["entity_type"],
                "entity_name": entity["entity_name"],
                "ingestion_source": f"{question.get('provider', 'ollama')}_textbook",
            }
        )

    def _store(tx):
        tx.run(
            """
            UNWIND $records AS item
            MERGE (q:StudyQuestion {id: item.question_id})
            SET q.question = item.question, q.question_en = item.question,
                q.question_te = item.question_te,
                q.options = item.options, q.options_en = item.options,
                q.options_te = item.options_te,
                q.correct_option = item.correct_option,
                q.answer = item.answer, q.answer_en = item.answer,
                q.answer_te = item.answer_te,
                q.explanation = item.explanation,
                q.explanation_en = item.explanation,
                q.explanation_te = item.explanation_te,
                q.level = item.level,
                q.textbook = item.textbook, q.mode = 'textbook',
                q.document_id = item.textbook, q.citations = item.citations,
                q.ingestion_source = item.ingestion_source,
                q.entity_type = item.entity_type, q.entity_name = item.entity_name,
                q.created_at = coalesce(q.created_at, datetime()),
                q.asked_at = coalesce(q.asked_at, q.created_at),
                q.liked = coalesce(q.liked, false)
            """,
            records=records,
        ).consume()

    with driver.session(database=database) as session:
        session.execute_write(_store)
    question_ids = [record["question_id"] for record in records]
    logger.info(
        "QUESTION_INGEST_BATCH_STORE_FINISHED count=%d duration_seconds=%.3f",
        len(question_ids),
        perf_counter() - started,
    )
    return question_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textbook", help="Generate questions for one exact PDF name")
    parser.add_argument("--limit", type=int, help="Process at most this many entities")
    parser.add_argument(
        "--only-zero-questions",
        action="store_true",
        help="Generate questions only for entities that currently have none",
    )
    args = parser.parse_args()
    run_started = perf_counter()
    settings = {
        "group": os.getenv("STUDY_GROUP", "Group 1"),
        "level": os.getenv("QUESTION_LEVEL", "AP Telangana Group 1 screening exam"),
        "use_model": parse_ingest_model(os.getenv("USE_MODEL", "OLLAMA")),
        "gemini_api_key": os.getenv("GEMINI_API_KEY", "").strip(),
        "gemini_endpoint": os.getenv(
            "GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/models"
        ),
        "gemini_model": os.getenv("GEMINI_MODEL", "models/gemini-3.8-flash").strip(),
        "gemini_timeout": float(os.getenv("GEMINI_QUESTION_TIMEOUT_SECONDS", "60")),
        "gemini_max_output_tokens": int(
            os.getenv("GEMINI_QUESTION_MAX_OUTPUT_TOKENS", "768")
        ),
        "grok_api_key": os.getenv("GROK_API_KEY", "").strip(),
        "grok_endpoint": os.getenv(
            "GROK_ENDPOINT", "https://api.groq.com/openai/v1/chat/completions"
        ),
        "grok_model": os.getenv("GROK_MODEL", "qwen/qwen-qwq-32b").strip(),
        "grok_timeout": float(os.getenv("GROK_QUESTION_TIMEOUT_SECONDS", "60")),
        "grok_max_output_tokens": int(
            os.getenv("GROK_QUESTION_MAX_OUTPUT_TOKENS", "4096")
        ),
        "ollama_endpoint": os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"),
        "ollama_model": os.getenv(
            "OLLAMA_MODEL", "qwen2.5-1.5b-instruct:local"
        ).strip(),
        "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "10m"),
        "timeout": float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300")),
        "context_tokens": int(os.getenv("OLLAMA_CONTEXT_TOKENS", "8192")),
        "max_output_tokens": int(os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "512")),
        "workers": max(
            1,
            int(
                os.getenv(
                    "QUESTION_INGEST_WORKERS",
                    "1"
                    if os.getenv("USE_MODEL", "OLLAMA").strip().upper()
                    in {"GEMINI", "GROK"}
                    else "4",
                )
            ),
        ),
        "grok_request_interval": max(
            0.0, float(os.getenv("GROK_QUESTION_REQUEST_INTERVAL_SECONDS", "20"))
        ),
        "questions_per_entity": int(os.getenv("QUESTIONS_PER_ENTITY", "3")),
        "ollama_workers": max(
            1, int(os.getenv("QUESTION_INGEST_OLLAMA_WORKERS", "1"))
        ),
        "batch_size": max(1, int(os.getenv("QUESTION_INGEST_BATCH_SIZE", "25"))),
    }
    selected_requirements = {
        IngestModel.GEMINI: (
            ("gemini_api_key", "GEMINI_API_KEY"),
            ("gemini_model", "GEMINI_MODEL"),
        ),
        IngestModel.GROK: (
            ("grok_api_key", "GROK_API_KEY"),
            ("grok_model", "GROK_MODEL"),
        ),
        IngestModel.OLLAMA: (("ollama_model", "OLLAMA_MODEL"),),
    }[settings["use_model"]]
    missing_settings = [
        env_name for setting_name, env_name in selected_requirements
        if not settings[setting_name]
    ]
    if missing_settings:
        raise RuntimeError(
            f"USE_MODEL={settings['use_model'].value} requires "
            + " and ".join(missing_settings)
        )
    selected_provider = {
        IngestModel.GEMINI: ("gemini", settings["gemini_model"]),
        IngestModel.GROK: ("groq", settings["grok_model"]),
        IngestModel.OLLAMA: ("ollama", settings["ollama_model"]),
    }[settings["use_model"]]
    logger.info(
        "QUESTION_INGEST_CONFIGURATION database=%s textbook=%s limit=%s "
        "only_zero_questions=%s use_model=%s provider=%s model=%s workers=%s batch_size=%s "
        "grok_request_interval_seconds=%.1f questions_per_entity=%d",
        os.getenv("NEO4J_DATABASE", "neo4j"),
        args.textbook or "all",
        args.limit if args.limit is not None else "unlimited",
        args.only_zero_questions,
        settings["use_model"].value,
        *selected_provider,
        settings["workers"],
        settings["batch_size"],
        settings["grok_request_interval"],
        settings["questions_per_entity"],
    )
    if not 3 <= settings["questions_per_entity"] <= 5:
        raise ValueError("QUESTIONS_PER_ENTITY must be between 3 and 5")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    driver = None
    try:
        if settings["use_model"] is IngestModel.GROK:
            with httpx.Client() as client:
                check_grok_credentials(settings, client)
        if settings["use_model"] is IngestModel.GROK:
            logger.info(
                "QUESTION_INGEST_PREFLIGHT_PASSED provider=%s model=%s; starting ingestion",
                *selected_provider,
            )
        else:
            logger.info(
                "QUESTION_INGEST_PROVIDER_SELECTED provider=%s model=%s; starting ingestion",
                *selected_provider,
            )
        logger.info("QUESTION_INGEST_NEO4J_CONNECT_START database=%s", database)
        connect_started = perf_counter()
        driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
            max_connection_lifetime=300,
            liveness_check_timeout=30,
            keep_alive=True,
        )
        logger.info(
            "QUESTION_INGEST_NEO4J_CONNECT_FINISHED database=%s duration_seconds=%.3f",
            database,
            perf_counter() - connect_started,
        )
        ensure_question_schema(driver, database)
        entities = load_entities(
            driver,
            database,
            args.textbook,
        )
        if args.limit is not None:
            entities = entities[: max(0, args.limit)]
            logger.info(
                "QUESTION_INGEST_LIMIT_APPLIED limit=%d entities=%d",
                max(0, args.limit),
                len(entities),
            )
        existing_by_entity = existing_questions_by_entity(driver, database, entities)
        pending = []
        skipped_count = 0
        for index, entity in enumerate(entities, 1):
            entity_key = (
                entity["textbook"],
                entity["entity_type"],
                entity["entity_name"],
            )
            previous_questions = list(
                dict.fromkeys(existing_by_entity.get(entity_key, []))
            )
            unique_questions = {
                normalized_question(question) for question in previous_questions
            }
            missing_count = max(
                0,
                settings["questions_per_entity"] - len(unique_questions),
            )
            if args.only_zero_questions and unique_questions:
                missing_count = 0
            if missing_count == 0:
                skipped_count += 1
                logger.info(
                    "QUESTION_INGEST_SKIPPED %d/%d textbook=%s type=%s name=%s "
                    "existing=%d target=%d reason=%s",
                    index,
                    len(entities),
                    entity["textbook"],
                    entity["entity_type"],
                    entity["entity_name"],
                    len(unique_questions),
                    settings["questions_per_entity"],
                    "has_existing_questions" if args.only_zero_questions else "target_reached",
                )
            else:
                pending.append((index, entity, previous_questions, missing_count))
        logger.info(
            "QUESTION_INGEST_START entities=%d pending_entities=%d pending_questions=%d "
            "target_per_entity=%d provider=%s:%s workers=%d batch_size=%d skipped=%d",
            len(entities),
            len(pending),
            sum(item[3] for item in pending),
            settings["questions_per_entity"],
            *selected_provider,
            settings["workers"],
            settings["batch_size"],
            skipped_count,
        )
        generated_count = 0
        failed_count = 0
        gemini_unavailable = Event()
        ollama_semaphore = BoundedSemaphore(settings["ollama_workers"])
        batch_count = (
            len(pending) + settings["batch_size"] - 1
        ) // settings["batch_size"]
        with (
            httpx.Client() as client,
            ThreadPoolExecutor(max_workers=settings["workers"]) as executor,
        ):
            for batch_index, start in enumerate(
                range(0, len(pending), settings["batch_size"]), 1
            ):
                batch_started = perf_counter()
                batch = pending[start : start + settings["batch_size"]]
                logger.info(
                    "QUESTION_INGEST_BATCH_START batch=%d/%d size=%d pending_before=%d",
                    batch_index,
                    batch_count,
                    len(batch),
                    len(pending) - start,
                )
                futures = {}
                for index, entity, previous_questions, missing_count in batch:
                    logger.info(
                        "QUESTION_INGEST_ENTITY %d/%d textbook=%s type=%s name=%s "
                        "existing=%d requested=%d status=queued",
                        index,
                        len(entities),
                        entity["textbook"],
                        entity["entity_type"],
                        entity["entity_name"],
                        len(previous_questions),
                        missing_count,
                    )
                    futures[
                        executor.submit(
                            generate_questions_for_entity,
                            settings,
                            entity,
                            previous_questions,
                            missing_count,
                            client,
                            gemini_unavailable,
                            ollama_semaphore,
                        )
                    ] = (index, entity)
                batch_failed = 0
                generated_this_batch = 0
                stored_this_batch = 0
                provider_counts = {selected_provider[0]: 0}
                for future in as_completed(futures):
                    index, entity = futures[future]
                    try:
                        questions, question_failures = future.result()
                        batch_failed += question_failures
                        failed_count += question_failures
                        generated_this_batch += len(questions)
                        for question in questions:
                            provider_counts[question["provider"]] = (
                                provider_counts.get(question["provider"], 0) + 1
                            )
                            logger.info(
                                "QUESTION_INGEST_GENERATED %d/%d name=%s provider=%s",
                                index,
                                len(entities),
                                entity["entity_name"],
                                question["provider"],
                            )
                        stored_count, store_failures = store_entity_questions(
                            driver, database, entity, questions
                        )
                        stored_this_batch += stored_count
                        generated_count += stored_count
                        batch_failed += store_failures
                        failed_count += store_failures
                    except Exception as exc:
                        failed_count += 1
                        batch_failed += 1
                        logger.exception(
                            "QUESTION_INGEST_FAILED %d/%d name=%s error_type=%s",
                            index,
                            len(entities),
                            entity["entity_name"],
                            type(exc).__name__,
                        )
                logger.info(
                    "QUESTION_INGEST_BATCH_FINISHED batch=%d/%d generated=%d "
                    "stored=%d failed=%d provider=%s provider_count=%d "
                    "duration_seconds=%.3f",
                    batch_index,
                    batch_count,
                    generated_this_batch,
                    stored_this_batch,
                    batch_failed,
                    selected_provider[0],
                    provider_counts[selected_provider[0]],
                    perf_counter() - batch_started,
                )
        logger.info(
            "QUESTION_INGEST_FINISHED provider=%s stored=%d skipped=%d failed=%d "
            "duration_seconds=%.3f",
            selected_provider[0],
            generated_count,
            skipped_count,
            failed_count,
            perf_counter() - run_started,
        )
    except Exception:
        logger.exception(
            "QUESTION_INGEST_ABORTED duration_seconds=%.3f",
            perf_counter() - run_started,
        )
        raise
    finally:
        if driver is not None:
            close_started = perf_counter()
            try:
                driver.close()
            except (DriverError, Neo4jError):
                logger.exception("QUESTION_INGEST_NEO4J_CLOSE_FAILED")
            else:
                logger.info(
                    "QUESTION_INGEST_NEO4J_CLOSE_FINISHED duration_seconds=%.3f",
                    perf_counter() - close_started,
                )
        flush_traces()


if __name__ == "__main__":
    main()

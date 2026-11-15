"""Generate textbook-grounded MCQs with Gemini, falling back to Ollama."""

import argparse
import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from enum import Enum
from threading import BoundedSemaphore, Event, Lock
from time import perf_counter

import httpx
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError

from learning_studio.prompts import entity_question_ingestion_prompt

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest_question")

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
    },
    "required": ["question", "options", "correct_option", "explanation"],
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
    correct_option = value.get("correct_option")
    if (
        not isinstance(value.get("question"), str)
        or not value["question"].strip()
        or not isinstance(options, list)
        or len(options) != 4
        or not all(isinstance(option, str) and option.strip() for option in options)
        or not isinstance(correct_option, int)
        or not 0 <= correct_option < 4
    ):
        raise ValueError("AI returned an invalid four-option question")
    return {
        "question": value["question"].strip(),
        "options": [option.strip() for option in options],
        "correct_option": correct_option,
        "explanation": str(value.get("explanation", "")).strip(),
    }


def textbook_name(properties: dict) -> str | None:
    for key in ("filename", "name", "source", "id"):
        value = properties.get(key)
        if isinstance(value, str) and value.lower().endswith(".pdf"):
            return value
    return None


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


def generate_question(
    settings: dict,
    entity: dict,
    client: httpx.Client,
    gemini_unavailable: Event | None = None,
    ollama_semaphore: BoundedSemaphore | None = None,
    gemini_circuit_lock: Lock | None = None,
    grok_unavailable: Event | None = None,
) -> dict:
    prompt = question_prompt(settings, entity)
    use_model = settings.get("use_model", IngestModel.GEMINI)
    if not isinstance(use_model, IngestModel):
        use_model = parse_ingest_model(str(use_model))
    use_gemini = use_model is IngestModel.GEMINI
    use_grok = use_model in (IngestModel.GEMINI, IngestModel.GROK)

    if use_gemini and settings["gemini_api_key"] and not (
        gemini_unavailable and gemini_unavailable.is_set()
    ):
        gemini_started = perf_counter()
        logger.info(
            "QUESTION_INGEST_PROVIDER_START name=%s provider=gemini model=%s timeout_seconds=%s",
            entity["entity_name"],
            settings["gemini_model"],
            settings["gemini_timeout"],
        )
        try:
            response = client.post(
                f"{settings['gemini_endpoint'].rstrip('/')}/"
                f"{settings['gemini_model'].removeprefix('models/')}:generateContent",
                headers={"x-goog-api-key": settings["gemini_api_key"]},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0,
                        "maxOutputTokens": settings["gemini_max_output_tokens"],
                        "responseMimeType": "application/json",
                    },
                },
                timeout=settings["gemini_timeout"],
            )
            response.raise_for_status()
            content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            question = {**parse_json(content), "provider": "gemini"}
            logger.info(
                "QUESTION_INGEST_PROVIDER_SUCCESS name=%s provider=gemini duration_seconds=%.3f",
                entity["entity_name"],
                perf_counter() - gemini_started,
            )
            return question
        except (
            httpx.HTTPError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            if gemini_unavailable is not None:
                if gemini_circuit_lock is None:
                    first_failure = not gemini_unavailable.is_set()
                    gemini_unavailable.set()
                else:
                    with gemini_circuit_lock:
                        first_failure = not gemini_unavailable.is_set()
                        gemini_unavailable.set()
                if first_failure:
                    logger.warning(
                        "QUESTION_INGEST_GEMINI_CIRCUIT_OPEN error_type=%s "
                        "remaining_requests=grok_then_ollama",
                        type(exc).__name__,
                    )
            logger.warning(
                "QUESTION_INGEST_GEMINI_FAILED name=%s duration_seconds=%.3f "
                "error_type=%s error=%s; falling back to Grok",
                entity["entity_name"],
                perf_counter() - gemini_started,
                type(exc).__name__,
                exc,
            )
    elif not use_gemini:
        logger.info(
            "QUESTION_INGEST_GEMINI_SKIPPED name=%s reason=USE_MODEL_%s",
            entity["entity_name"],
            use_model.value,
        )
    elif settings["gemini_api_key"]:
        logger.info(
            "QUESTION_INGEST_GEMINI_SKIPPED name=%s reason=provider_unavailable_for_this_run",
            entity["entity_name"],
        )
    else:
        logger.info(
            "QUESTION_INGEST_GEMINI_SKIPPED name=%s reason=GEMINI_API_KEY_not_configured",
            entity["entity_name"],
        )

    if use_grok and settings["grok_api_key"] and not (
        grok_unavailable and grok_unavailable.is_set()
    ):
        grok_started = perf_counter()
        logger.info(
            "QUESTION_INGEST_PROVIDER_START name=%s provider=grok model=%s timeout_seconds=%s",
            entity["entity_name"],
            settings["grok_model"],
            settings["grok_timeout"],
        )
        try:
            response = client.post(
                settings["grok_endpoint"],
                headers={"Authorization": f"Bearer {settings['grok_api_key']}"},
                json={
                    "model": settings["grok_model"],
                    "temperature": 0,
                    "max_tokens": settings["grok_max_output_tokens"],
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": "Return only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=settings["grok_timeout"],
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            question = {**parse_json(content), "provider": "grok"}
            logger.info(
                "QUESTION_INGEST_PROVIDER_SUCCESS name=%s provider=grok duration_seconds=%.3f",
                entity["entity_name"],
                perf_counter() - grok_started,
            )
            return question
        except (
            httpx.HTTPError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            if grok_unavailable is not None:
                if gemini_circuit_lock is None:
                    first_failure = not grok_unavailable.is_set()
                    grok_unavailable.set()
                else:
                    with gemini_circuit_lock:
                        first_failure = not grok_unavailable.is_set()
                        grok_unavailable.set()
                if first_failure:
                    logger.warning(
                        "QUESTION_INGEST_GROK_CIRCUIT_OPEN error_type=%s "
                        "remaining_requests=ollama",
                        type(exc).__name__,
                    )
            logger.warning(
                "QUESTION_INGEST_GROK_FAILED name=%s duration_seconds=%.3f "
                "%s; falling back to Ollama",
                entity["entity_name"],
                perf_counter() - grok_started,
                grok_error_detail(exc, settings["grok_api_key"]),
            )
    elif not use_grok:
        logger.info(
            "QUESTION_INGEST_GROK_SKIPPED name=%s reason=USE_MODEL_%s",
            entity["entity_name"],
            use_model.value,
        )
    elif settings["grok_api_key"]:
        logger.info(
            "QUESTION_INGEST_GROK_SKIPPED name=%s reason=provider_unavailable_for_this_run",
            entity["entity_name"],
        )
    else:
        logger.info(
            "QUESTION_INGEST_GROK_SKIPPED name=%s reason=GROK_API_KEY_not_configured",
            entity["entity_name"],
        )

    if not settings["ollama_model"]:
        raise RuntimeError(
            "Gemini/Grok question generation failed and OLLAMA_MODEL is not configured"
        )
    ollama_model = OLLAMA_MODEL_ALIASES.get(
        settings["ollama_model"], settings["ollama_model"]
    )
    if ollama_model != settings["ollama_model"]:
        logger.info(
            "QUESTION_INGEST_OLLAMA_MODEL_RESOLVED configured_model=%s local_model=%s",
            settings["ollama_model"],
            ollama_model,
        )
    ollama_started = perf_counter()
    logger.info(
        "QUESTION_INGEST_OLLAMA_QUEUE name=%s model=%s",
        entity["entity_name"],
        ollama_model,
    )
    semaphore_context = (
        ollama_semaphore if ollama_semaphore is not None else nullcontext()
    )
    with semaphore_context:
        ollama_queue_wait = perf_counter() - ollama_started
        logger.info(
            "QUESTION_INGEST_PROVIDER_START name=%s provider=ollama model=%s "
            "timeout_seconds=%s queue_wait_seconds=%.3f",
            entity["entity_name"],
            ollama_model,
            settings["timeout"],
            ollama_queue_wait,
        )
        ollama_request_started = perf_counter()
        try:
            response = client.post(
                f"{settings['ollama_endpoint'].rstrip('/')}/api/chat",
                json={
                    "model": ollama_model,
                    "stream": False,
                    "keep_alive": settings["keep_alive"],
                    "format": OLLAMA_QUESTION_SCHEMA,
                    "options": {
                        "temperature": 0,
                        "num_ctx": settings["context_tokens"],
                        "num_predict": settings["max_output_tokens"],
                    },
                    "messages": [
                        {"role": "system", "content": "Return only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=settings["timeout"],
            )
            response.raise_for_status()
            question = {
                **parse_json(response.json()["message"]["content"]),
                "provider": "ollama",
            }
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            logger.exception(
                "QUESTION_INGEST_PROVIDER_FAILED name=%s provider=ollama "
                "queue_wait_seconds=%.3f request_duration_seconds=%.3f",
                entity["entity_name"],
                ollama_queue_wait,
                perf_counter() - ollama_request_started,
            )
            raise
        logger.info(
            "QUESTION_INGEST_PROVIDER_SUCCESS name=%s provider=ollama "
            "queue_wait_seconds=%.3f request_duration_seconds=%.3f",
            entity["entity_name"],
            ollama_queue_wait,
            perf_counter() - ollama_request_started,
        )
    return question


def question_id_for_entity(entity: dict) -> str:
    source_id = hashlib.sha256(
        f"{entity['textbook']}|{entity['entity_type']}|{entity['entity_name']}".encode()
    ).hexdigest()
    return f"textbook-{source_id[:24]}"


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


def store_questions(
    driver, database: str, items: list[tuple[dict, dict]]
) -> list[str]:
    if not items:
        return []
    started = perf_counter()
    records = []
    for entity, question in items:
        question_id = question_id_for_entity(entity)
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
                "options": question["options"],
                "correct_option": question["correct_option"],
                "answer": answer,
                "explanation": question["explanation"],
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
            SET q.question = item.question, q.options = item.options,
                q.correct_option = item.correct_option, q.answer = item.answer,
                q.explanation = item.explanation, q.level = item.level,
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
        "gemini_timeout": float(os.getenv("GEMINI_QUESTION_TIMEOUT_SECONDS", "8")),
        "gemini_max_output_tokens": int(
            os.getenv("GEMINI_QUESTION_MAX_OUTPUT_TOKENS", "256")
        ),
        "grok_api_key": os.getenv("GROK_API_KEY", "").strip(),
        "grok_endpoint": os.getenv(
            "GROK_ENDPOINT", "https://api.x.ai/v1/chat/completions"
        ),
        "grok_model": os.getenv("GROK_MODEL", "grok-4.7").strip(),
        "grok_timeout": float(os.getenv("GROK_QUESTION_TIMEOUT_SECONDS", "15")),
        "grok_max_output_tokens": int(
            os.getenv("GROK_QUESTION_MAX_OUTPUT_TOKENS", "256")
        ),
        "ollama_endpoint": os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"),
        "ollama_model": os.getenv(
            "OLLAMA_MODEL", "qwen2.5-1.5b-instruct:local"
        ).strip(),
        "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "10m"),
        "timeout": float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300")),
        "context_tokens": int(os.getenv("OLLAMA_CONTEXT_TOKENS", "8192")),
        "max_output_tokens": int(os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "512")),
        "workers": max(1, int(os.getenv("QUESTION_INGEST_WORKERS", "4"))),
        "ollama_workers": max(1, int(os.getenv("QUESTION_INGEST_OLLAMA_WORKERS", "1"))),
        "batch_size": max(1, int(os.getenv("QUESTION_INGEST_BATCH_SIZE", "25"))),
    }
    available_models = {
        IngestModel.GEMINI: bool(settings["gemini_api_key"]),
        IngestModel.GROK: bool(settings["grok_api_key"]),
        IngestModel.OLLAMA: bool(settings["ollama_model"]),
    }
    if not available_models[settings["use_model"]]:
        logger.error(
            "QUESTION_INGEST_CONFIGURATION_FAILED use_model=%s reason=provider_not_configured",
            settings["use_model"].value,
        )
        raise RuntimeError(
            f"USE_MODEL={settings['use_model'].value} is selected but its credentials "
            "or model are not configured"
        )
    if settings["grok_api_key"] and not settings["grok_model"]:
        logger.error(
            "QUESTION_INGEST_CONFIGURATION_FAILED reason=GROK_MODEL_not_configured"
        )
        raise RuntimeError("GROK_MODEL must be configured when GROK_API_KEY is set")
    logger.info(
        "QUESTION_INGEST_CONFIGURATION database=%s textbook=%s limit=%s "
        "use_model=%s "
        "gemini_configured=%s gemini_model=%s gemini_timeout_seconds=%s "
        "gemini_max_output_tokens=%s grok_configured=%s grok_model=%s "
        "grok_timeout_seconds=%s grok_max_output_tokens=%s ollama_configured=%s "
        "ollama_model=%s ollama_timeout_seconds=%s workers=%s ollama_workers=%s batch_size=%s",
        os.getenv("NEO4J_DATABASE", "neo4j"),
        args.textbook or "all",
        args.limit if args.limit is not None else "unlimited",
        settings["use_model"].value,
        bool(settings["gemini_api_key"]),
        settings["gemini_model"],
        settings["gemini_timeout"],
        settings["gemini_max_output_tokens"],
        bool(settings["grok_api_key"]),
        settings["grok_model"] or "not-configured",
        settings["grok_timeout"],
        settings["grok_max_output_tokens"],
        bool(settings["ollama_model"]),
        settings["ollama_model"] or "disabled",
        settings["timeout"],
        settings["workers"],
        settings["ollama_workers"],
        settings["batch_size"],
    )
    if not settings["gemini_api_key"] and not settings["grok_api_key"]:
        logger.warning(
            "QUESTION_INGEST_CLOUD_MODELS_DISABLED reason=GEMINI_API_KEY_and_GROK_API_KEY_not_configured "
            "provider_order=ollama"
        )
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    driver = None
    try:
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
        question_ids = [question_id_for_entity(entity) for entity in entities]
        existing_ids = existing_question_ids(driver, database, question_ids)
        pending = []
        skipped_count = 0
        for index, entity in enumerate(entities, 1):
            entity_question_id = question_id_for_entity(entity)
            if any(
                candidate in existing_ids
                for candidate in question_id_candidates(
                    entity_question_id
                )
            ):
                skipped_count += 1
                logger.info(
                    "QUESTION_INGEST_SKIPPED %d/%d textbook=%s type=%s name=%s "
                    "question_id=%s reason=already_exists",
                    index,
                    len(entities),
                    entity["textbook"],
                    entity["entity_type"],
                    entity["entity_name"],
                    entity_question_id,
                )
            else:
                pending.append((index, entity))
        logger.info(
            "QUESTION_INGEST_START entities=%d pending=%d provider_order=%s "
            "workers=%d batch_size=%d skipped=%d",
            len(entities),
            len(pending),
            " -> ".join(
                [
                    f"{settings['use_model'].value.lower()}:"
                    + (
                        settings["gemini_model"]
                        if settings["use_model"] is IngestModel.GEMINI
                        else settings["grok_model"]
                        if settings["use_model"] is IngestModel.GROK
                        else settings["ollama_model"]
                    ),
                    *(
                        [f"grok:{settings['grok_model']}"]
                        if settings["use_model"] is IngestModel.GEMINI
                        and settings["grok_api_key"]
                        else []
                    ),
                    *(
                        [f"ollama:{settings['ollama_model']}"]
                        if settings["use_model"] is not IngestModel.OLLAMA
                        and settings["ollama_model"]
                        else []
                    ),
                ]
            ),
            settings["workers"],
            settings["batch_size"],
            skipped_count,
        )
        generated_count = 0
        failed_count = 0
        gemini_unavailable = Event()
        grok_unavailable = Event()
        gemini_circuit_lock = Lock()
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
                for index, entity in batch:
                    logger.info(
                        "QUESTION_INGEST_ENTITY %d/%d textbook=%s type=%s name=%s status=queued",
                        index,
                        len(entities),
                        entity["textbook"],
                        entity["entity_type"],
                        entity["entity_name"],
                    )
                    futures[
                        executor.submit(
                            generate_question,
                            settings,
                            entity,
                            client,
                            gemini_unavailable,
                            ollama_semaphore,
                            gemini_circuit_lock,
                            grok_unavailable,
                        )
                    ] = (index, entity)
                completed = []
                batch_failed = 0
                provider_counts = {"gemini": 0, "ollama": 0}
                for future in as_completed(futures):
                    index, entity = futures[future]
                    try:
                        question = future.result()
                        completed.append((entity, question))
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
                stored_this_batch = 0
                try:
                    stored_ids = store_questions(driver, database, completed)
                except (DriverError, Neo4jError):
                    logger.exception(
                        "QUESTION_INGEST_BATCH_STORE_FAILED batch=%d/%d count=%d",
                        batch_index,
                        batch_count,
                        len(completed),
                    )
                    for entity, question in completed:
                        try:
                            store_question(driver, database, entity, question)
                        except (DriverError, Neo4jError) as item_exc:
                            failed_count += 1
                            batch_failed += 1
                            logger.exception(
                                "QUESTION_INGEST_STORE_FAILED name=%s error_type=%s",
                                entity["entity_name"],
                                type(item_exc).__name__,
                            )
                        else:
                            generated_count += 1
                            stored_this_batch += 1
                            logger.info(
                                "QUESTION_INGEST_STORED id=%s provider=%s",
                                entity["entity_name"],
                                question["provider"],
                            )
                else:
                    generated_count += len(stored_ids)
                    stored_this_batch = len(stored_ids)
                    for entity, question in completed:
                        logger.info(
                            "QUESTION_INGEST_STORED id=%s provider=%s",
                            entity["entity_name"],
                            question["provider"],
                        )
                logger.info(
                    "QUESTION_INGEST_BATCH_FINISHED batch=%d/%d generated=%d "
                    "stored=%d failed=%d gemini=%d ollama=%d duration_seconds=%.3f",
                    batch_index,
                    batch_count,
                    len(completed),
                    stored_this_batch,
                    batch_failed,
                    provider_counts["gemini"],
                    provider_counts["ollama"],
                    perf_counter() - batch_started,
                )
        logger.info(
            "QUESTION_INGEST_FINISHED stored=%d skipped=%d failed=%d duration_seconds=%.3f",
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


if __name__ == "__main__":
    main()

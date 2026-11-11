"""Generate textbook-grounded MCQs with local Ollama and store them in Neo4j."""

import argparse
import hashlib
import json
import logging
import os
import re
import time

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


def parse_json(content: str) -> dict:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Ollama returned a non-object JSON value")
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
        raise ValueError("Ollama returned an invalid four-option question")
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
        return entities


def generate_question(settings: dict, entity: dict) -> dict:
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
    response = httpx.post(
        f"{settings['ollama_endpoint'].rstrip('/')}/api/chat",
        json={
            "model": settings["ollama_model"],
            "stream": False,
            "keep_alive": settings["keep_alive"],
            "format": "json",
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
    return parse_json(response.json()["message"]["content"])


def question_id_for_entity(entity: dict) -> str:
    source_id = hashlib.sha256(
        f"{entity['textbook']}|{entity['entity_type']}|{entity['entity_name']}".encode()
    ).hexdigest()
    return f"textbook-{source_id[:24]}"


def question_exists(driver, database: str, question_id: str) -> bool:
    clean_id = question_id.removeprefix("textbook-")
    candidate_ids = list(
        {
            question_id,
            f"textbook-{clean_id}",
            clean_id,
            f"textbook-{clean_id[:24]}",
            clean_id[:24],
        }
    )

    def _check(tx):
        result = tx.run(
            """
            MATCH (q:StudyQuestion)
            WHERE q.id IN $candidate_ids
            RETURN count(q) > 0 AS exists
            """,
            candidate_ids=candidate_ids,
        ).single()
        return bool(result and result["exists"])

    with driver.session(database=database) as session:
        return session.execute_read(_check)


def ensure_question_schema(driver, database: str) -> None:
    def _create_index(tx):
        tx.run(
            "CREATE INDEX study_question_id_index IF NOT EXISTS "
            "FOR (q:StudyQuestion) ON (q.id)"
        ).consume()

    try:
        with driver.session(database=database) as session:
            session.execute_write(_create_index)
    except Exception as exc:
        logger.warning("Could not ensure question index: %s", exc)


def store_question(driver, database: str, entity: dict, question: dict) -> str:
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
    answer = f"{question['options'][question['correct_option']]}. {question['explanation']}".strip()

    def _store(tx):
        tx.run(
            """
            MERGE (q:StudyQuestion {id: $question_id})
            SET q.question = $question, q.options = $options,
                q.correct_option = $correct_option, q.answer = $answer,
                q.explanation = $explanation, q.level = $level,
                q.textbook = $textbook, q.mode = 'textbook',
                q.document_id = $textbook, q.citations = $citations,
                q.ingestion_source = 'ollama_textbook',
                q.entity_type = $entity_type, q.entity_name = $entity_name,
                q.created_at = coalesce(q.created_at, datetime()),
                q.asked_at = coalesce(q.asked_at, q.created_at),
                q.liked = coalesce(q.liked, false)
            """,
            question_id=question_id,
            question=question["question"],
            options=question["options"],
            correct_option=question["correct_option"],
            answer=answer,
            explanation=question["explanation"],
            level=os.environ.get("QUESTION_LEVEL", "AP Telangana Group 1 screening exam"),
            textbook=entity["textbook"],
            citations=citations,
            entity_type=entity["entity_type"],
            entity_name=entity["entity_name"],
        ).consume()

    with driver.session(database=database) as session:
        session.execute_write(_store)
    return question_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textbook", help="Generate questions for one exact PDF name")
    parser.add_argument("--limit", type=int, help="Process at most this many entities")
    args = parser.parse_args()
    settings = {
        "group": os.getenv("STUDY_GROUP", "Group 1"),
        "level": os.getenv("QUESTION_LEVEL", "AP Telangana Group 1 screening exam"),
        "ollama_endpoint": os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "").strip(),
        "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "10m"),
        "timeout": float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300")),
        "context_tokens": int(os.getenv("OLLAMA_CONTEXT_TOKENS", "8192")),
        "max_output_tokens": int(os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "512")),
    }
    if not settings["ollama_model"]:
        raise RuntimeError("OLLAMA_MODEL must be configured for question ingestion")
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
        max_connection_lifetime=300,
        liveness_check_timeout=30,
        keep_alive=True,
    )
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    try:
        ensure_question_schema(driver, database)
        entities = load_entities(
            driver,
            database,
            args.textbook,
        )
        if args.limit is not None:
            entities = entities[: max(0, args.limit)]
        logger.info("QUESTION_INGEST_START entities=%d provider=ollama model=%s", len(entities), settings["ollama_model"])
        for index, entity in enumerate(entities, 1):
            logger.info(
                "QUESTION_INGEST_ENTITY %d/%d textbook=%s type=%s name=%s",
                index,
                len(entities),
                entity["textbook"],
                entity["entity_type"],
                entity["entity_name"],
            )
            question_id = question_id_for_entity(entity)
            if question_exists(driver, database, question_id):
                logger.info("QUESTION_INGEST_SKIPPED id=%s", entity["entity_name"])
                continue

            for attempt in range(1, 4):
                try:
                    question = generate_question(settings, entity)
                    store_question(
                        driver,
                        database,
                        entity,
                        question,
                    )
                    logger.info("QUESTION_INGEST_STORED id=%s", entity["entity_name"])
                    break
                except (
                    httpx.HTTPError,
                    KeyError,
                    TypeError,
                    ValueError,
                    DriverError,
                    Neo4jError,
                ) as exc:
                    if attempt == 3:
                        logger.error("QUESTION_INGEST_FAILED name=%s error=%s", entity["entity_name"], exc)
                    else:
                        time.sleep(attempt * 5)
        logger.info("QUESTION_INGEST_FINISHED")
    finally:
        driver.close()


if __name__ == "__main__":
    main()

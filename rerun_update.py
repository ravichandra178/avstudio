import argparse
import json
import os
import re
import time

import httpx
from dotenv import load_dotenv
from neo4j import GraphDatabase

TELUGU = re.compile(r"[\u0c00-\u0c7f]")
SOURCE_ARTIFACT = re.compile(
    r"https?://|www\.|references?\s*:|source_document_id|\bclick here\b",
    re.IGNORECASE,
)


def needs_refresh(summary_te: str | None) -> bool:
    text = (summary_te or "").strip()
    return (
        not text
        or not TELUGU.search(text)
        or bool(SOURCE_ARTIFACT.search(text))
    )


def make_telugu_summary(
    client: httpx.Client, api_key: str, model: str, topic: str,
    endpoint: str, summary_en: str, source_en: str,
) -> str:
    prompt = f"""
Create a concise, accurate Telugu study summary in exactly two paragraphs.
Use Telugu script. Base it only on the supplied textbook material.
Ignore URLs, references, page numbers, repeated headers, and navigation text.
Do not include questions, citations, or source links.

Topic: {topic}
Existing English summary:
{summary_en}

Textbook passage:
{source_en}

Return JSON only: {{"summary_te": "first paragraph\\n\\nsecond paragraph"}}
"""
    response = client.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return one valid JSON object only. Do not include "
                        "markdown or text outside the object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1200,
            "response_format": {"type": "json_object"},
        },
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    text = payload["choices"][0]["message"]["content"]
    summary = json.loads(text)["summary_te"].strip()
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", summary) if part.strip()]

    if (
        len(paragraphs) != 2
        or not TELUGU.search(summary)
        or SOURCE_ARTIFACT.search(summary)
    ):
        raise ValueError(f"Grok returned an invalid Telugu summary for {topic!r}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="Save regenerated summaries to Neo4j (default is preview only)",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GROK_API_KEY", "").strip()
    endpoint = os.getenv(
        "GROK_ENDPOINT", "https://api.groq.com/openai/v1/chat/completions"
    ).strip()
    model = os.getenv("GROK_MODEL", "").strip()
    if not api_key:
        raise RuntimeError("GROK_API_KEY is not configured")
    if not model:
        raise RuntimeError("GROK_MODEL is not configured")

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            records = list(session.run(
                """
                MATCH (sq:ScreeningQuestion)
                WHERE toLower(coalesce(sq.textbook, '')) CONTAINS 'pib_prism'
                RETURN sq.id AS id,
                       coalesce(sq.topic, sq.entity_name, '') AS topic,
                       coalesce(sq.summary_en, '') AS summary_en,
                       coalesce(sq.etext_en, '') AS source_en,
                       sq.summary_te AS summary_te
                ORDER BY sq.created_at DESC
                """
            ))

            targets = [record.data() for record in records
                       if needs_refresh(record["summary_te"])]
            print(f"Found {len(targets)} PIB Prism summaries to refresh.")
            if not args.write:
                for item in targets:
                    print(f"- {item['topic'] or item['id']}")
                print("Preview only. Rerun with --write to update Neo4j.")
                return

            with httpx.Client() as client:
                for index, item in enumerate(targets):
                    evidence = item["source_en"] or item["summary_en"]
                    if not evidence.strip():
                        print(f"SKIP {item['topic']}: no English source text")
                        continue

                    summary = make_telugu_summary(
                        client, api_key, model, item["topic"], endpoint,
                        item["summary_en"], evidence,
                    )
                    session.run(
                        """
                        MATCH (sq:ScreeningQuestion {id: $id})
                        WHERE toLower(coalesce(sq.textbook, '')) CONTAINS 'pib_prism'
                        SET sq.summary_te = $summary_te
                        """,
                        id=item["id"],
                        summary_te=summary,
                    ).consume()
                    print(f"UPDATED {item['topic'] or item['id']}")
                    if index + 1 < len(targets):
                        time.sleep(float(os.getenv("GROK_REQUEST_INTERVAL_SECONDS", "15")))
    finally:
        driver.close()


if __name__ == "__main__":
    main()
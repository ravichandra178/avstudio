"""Ingest one bilingual previous-exam PDF into separate PreviousQuestion nodes."""

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import uuid
from pathlib import Path

import httpx
import pymupdf
from dotenv import load_dotenv
from json_repair import repair_json
from neo4j import GraphDatabase
from pypdf import PdfReader

from learning_studio.language import is_language_text

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("ingest_previous")

SUPPORTED_MODELS = {"GROK", "GEMINI", "OLLAMA"}
EXTRACTION_PROMPT = """Extract every multiple-choice question from this exam-paper page.
The source may print English and Telugu versions of each question and option. Copy
each language exactly as printed; do not translate, rewrite, or invent missing
language text. Put empty strings in unavailable language fields. Keep options in
the same order and use a zero-based correct_option. Determine the correct answer
from the paper or reliable subject knowledge. Skip non-question content and
questions that do not have at least two answer options.
Keep language fields strictly separate: question_en and options_en must contain
English/Latin text only; question_te and options_te must contain Telugu script
only (numbers and punctuation are allowed). Never copy English into Telugu
fields or combine the parallel translations into one field. If one language is
unreadable, leave that language's fields empty and preserve the readable one.

The PDF text can contain OCR mistakes or garbled font characters. Recover a
question when its meaning and answer choices are still clear, using the paired
English/Telugu version to help interpret it. Do not return an empty list merely
because some characters are noisy.

Return only JSON in this shape:
{"questions":[{"question_en":"...","question_te":"...","options_en":["...","..."],
"options_te":["...","..."],"correct_option":0,"explanation_en":"...",
"explanation_te":"...","year":2024}]}

Use an empty string for a missing explanation or year. Each question must have
at least one question text and two options in at least one language.

Page text:
"""

QUESTION_CUE_RE = re.compile(
    r"\b(?:which|what|who|when|where|why|how|consider|select|identify|match|choose|"
    r"following|correct|incorrect|statement|statements)\b"
    r"|(?:ఏది|ఏమి|ఎవరు|ఎప్పుడు|ఎక్కడ|ఎలా|క్రింది|కింది|సరైన|తప్పు|జతపరచండి|పరిగణించండి)",
    re.IGNORECASE,
)
OPTION_LINE_RE = re.compile(r"^\s*(?:\(?[A-Da-d]\)?[.)]|[([]?[1-4][)\].])\s+\S")
TELUGU_SCRIPT_RE = re.compile(r"[\u0c00-\u0c7f]")


class ModelResponseError(ValueError):
    pass


def post_with_rate_limit_retry(
    client: httpx.Client, url: str, settings: dict, **request_kwargs
) -> httpx.Response:
    max_attempts = settings.get("rate_limit_max_attempts", 5)
    max_wait = settings.get("rate_limit_max_wait_seconds", 120)
    for attempt in range(1, max_attempts + 1):
        response = client.post(url, **request_kwargs)
        if response.status_code != 429 or attempt == max_attempts:
            response.raise_for_status()
            return response

        retry_after = response.headers.get("Retry-After")
        try:
            delay = float(retry_after) if retry_after else min(2**attempt, 30)
        except ValueError:
            try:
                retry_time = parsedate_to_datetime(retry_after)
                if retry_time.tzinfo is None:
                    retry_time = retry_time.replace(tzinfo=timezone.utc)
                delay = max(
                    0,
                    (retry_time - datetime.now(timezone.utc)).total_seconds(),
                )
            except (TypeError, ValueError, OverflowError):
                delay = min(2**attempt, 30)
        if delay < 0:
            delay = min(2**attempt, 30)
        if delay > max_wait:
            raise httpx.HTTPStatusError(
                f"Rate limit requested a {delay:g}s wait, above the configured "
                f"{max_wait:g}s maximum",
                request=response.request,
                response=response,
            )
        logger.warning(
            "Provider rate limit (429); retry %d/%d in %.1f seconds",
            attempt,
            max_attempts - 1,
            delay,
        )
        time.sleep(delay)

    raise RuntimeError("Rate-limit retry loop ended unexpectedly")


def is_question_page(text: str) -> bool:
    option_count = sum(
        bool(OPTION_LINE_RE.match(line)) for line in text.splitlines()
    )
    return option_count >= 2 and bool(QUESTION_CUE_RE.search(text))


def tesseract_command() -> str:
    command = shutil.which("tesseract")
    if not command:
        raise RuntimeError(
            "This PDF's embedded Telugu text is not extractable. Install Tesseract OCR "
            "with English and Telugu language data, then rerun "
            "`brew install tesseract tesseract-lang`."
        )
    result = subprocess.run(
        [command, "--list-langs"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    languages = set(result.stdout.splitlines()[1:])
    missing = {"eng", "tel"} - languages
    if missing:
        raise RuntimeError(
            "Tesseract is missing OCR language data: "
            + ", ".join(sorted(missing))
            + ". Install it with `brew install tesseract-lang`."
        )
    return command


def ocr_page(page, command: str) -> str:
    image = page.get_pixmap(matrix=pymupdf.Matrix(3, 3), alpha=False)
    result = subprocess.run(
        [command, "stdin", "stdout", "-l", "eng+tel", "--psm", "3"],
        input=image.tobytes("png"),
        capture_output=True,
        check=True,
        timeout=120,
    )
    return result.stdout.decode("utf-8", errors="replace").strip()


def parse_model(value: str) -> str:
    model = value.strip().upper()
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"USE_MODEL must be one of: {', '.join(sorted(SUPPORTED_MODELS))}")
    return model


def normalize_question(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"^\s*(?:q(?:uestion)?\s*)?\d+[\s.):\-]+", "", value)
    return " ".join(re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).split())


def question_hash(question: dict) -> str:
    canonical = normalize_question(question["question_en"] or question["question_te"])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_questions(content: str) -> list[dict]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Model response was not strict JSON; attempting JSON repair")
        payload = repair_json(cleaned, return_objects=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("questions"), list):
        preview = cleaned[:1000].replace("\n", "\\n")
        root_type = type(payload).__name__
        root_keys = list(payload)[:20] if isinstance(payload, dict) else None
        raise ModelResponseError(
            "AI response must contain a 'questions' list "
            f"(parsed root={root_type}, keys={root_keys}, response={preview!r})"
        )

    questions = []
    for item in payload["questions"]:
        if not isinstance(item, dict):
            raise ValueError("AI returned a non-object question")
        question_en = str(item.get("question_en") or "").strip()
        question_te = str(item.get("question_te") or "").strip()
        options_en = item.get("options_en") or []
        options_te = item.get("options_te") or []
        correct_option = item.get("correct_option")
        if (
            not isinstance(options_en, list)
            or not isinstance(options_te, list)
            or not all(isinstance(value, str) for value in options_en + options_te)
            or not 2 <= max(len(options_en), len(options_te)) <= 6
            or isinstance(correct_option, bool)
            or not isinstance(correct_option, int)
            or not 0 <= correct_option < max(len(options_en), len(options_te))
        ):
            logger.warning("Skipped invalid question structure from model output")
            continue
        options_en = [value.strip() for value in options_en]
        options_te = [value.strip() for value in options_te]
        if options_en and not all(options_en) or options_te and not all(options_te):
            logger.warning("Skipped question with an empty answer option")
            continue

        if question_en and not is_language_text(question_en, "en"):
            logger.warning("Dropped mixed-script English question field")
            question_en = ""
            options_en = []
        if question_te and not is_language_text(question_te, "te"):
            logger.warning("Dropped mixed-script Telugu question field")
            question_te = ""
            options_te = []
        if options_en and any(not is_language_text(option, "en") for option in options_en):
            logger.warning("Dropped mixed-script English options")
            question_en = ""
            options_en = []
        if options_te and any(not is_language_text(option, "te") for option in options_te):
            logger.warning("Dropped mixed-script Telugu options")
            question_te = ""
            options_te = []
        if not question_en or len(options_en) < 2:
            question_en = ""
            options_en = []
        if not question_te or len(options_te) < 2:
            question_te = ""
            options_te = []
        if (
            not question_en and not question_te
            or options_en and correct_option >= len(options_en)
            or options_te and correct_option >= len(options_te)
        ):
            logger.warning("Skipped question without a valid language and matching options")
            continue

        explanation_en = str(item.get("explanation_en") or "").strip()
        explanation_te = str(item.get("explanation_te") or "").strip()
        if explanation_en and not is_language_text(explanation_en, "en"):
            explanation_en = ""
        if explanation_te and not is_language_text(explanation_te, "te"):
            explanation_te = ""
        answer_en = options_en[correct_option] if options_en else ""
        answer_te = options_te[correct_option] if options_te else ""
        if explanation_en and answer_en:
            answer_en = f"{answer_en}. {explanation_en}".strip(". ")
        if explanation_te and answer_te:
            answer_te = f"{answer_te}. {explanation_te}".strip(". ")
        question = {
            "question_en": question_en,
            "question_te": question_te,
            "options_en": options_en,
            "options_te": options_te,
            "correct_option": correct_option,
            "answer_en": answer_en,
            "answer_te": answer_te,
            "year": item.get("year") if isinstance(item.get("year"), int) else None,
        }
        question["question_hash"] = question_hash(question)
        question["id"] = f"previous-{question['question_hash']}"
        questions.append(question)
    return questions


def generate_questions(page_text: str, settings: dict, client: httpx.Client) -> list[dict]:
    prompt = EXTRACTION_PROMPT + page_text
    model = settings["use_model"]
    if model == "GROK":
        endpoint = settings["grok_endpoint"].rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        response = post_with_rate_limit_retry(
            client,
            endpoint,
            settings,
            headers={"Authorization": f"Bearer {settings['grok_api_key']}"},
            json={
                "model": settings["grok_model"],
                "temperature": 0,
                "max_tokens": settings["max_output_tokens"],
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return exactly one complete JSON object matching the requested "
                            "schema. Do not include markdown, analysis, or text outside JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=settings["timeout"],
        )
        content = response.json()["choices"][0]["message"]["content"]
    elif model == "GEMINI":
        response = post_with_rate_limit_retry(
            client,
            f"{settings['gemini_endpoint'].rstrip('/')}/"
            f"{settings['gemini_model'].removeprefix('models/')}:generateContent",
            settings,
            headers={"x-goog-api-key": settings["gemini_api_key"]},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": settings["max_output_tokens"],
                    "responseMimeType": "application/json",
                },
            },
            timeout=settings["timeout"],
        )
        content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        response = post_with_rate_limit_retry(
            client,
            f"{settings['ollama_endpoint'].rstrip('/')}/api/chat",
            settings,
            json={
                "model": settings["ollama_model"],
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": "Return one valid JSON object only."},
                    {"role": "user", "content": prompt},
                ],
                "options": {
                    "temperature": 0,
                    "num_ctx": settings["context_tokens"],
                    "num_predict": settings["max_output_tokens"],
                },
            },
            timeout=settings["timeout"],
        )
        content = response.json()["message"]["content"]
    questions = parse_questions(content)
    if not questions:
        logger.info("Model found no multiple-choice questions on this page")
    return questions


def _store_question(tx, question: dict, source_name: str, page_number: int) -> bool:
    properties = {
        **question,
        "question": question["question_en"] or question["question_te"],
        "options": question["options_en"] or question["options_te"],
        "answer": question["answer_en"] or question["answer_te"],
        "level": os.getenv("QUESTION_LEVEL", "AP Telangana Group 1 screening exam"),
        "source_name": source_name,
        "page": page_number,
        "citations": json.dumps(
            [{"source": source_name, "page": page_number}], ensure_ascii=False
        ),
    }
    existing = tx.run(
        """
        MATCH (existing:PreviousQuestion)
        WHERE existing.question_hash = $question_hash
           OR ($question_en <> '' AND toLower(trim(existing.question_en)) =
                toLower(trim($question_en)))
           OR ($question_te <> '' AND toLower(trim(existing.question_te)) =
                toLower(trim($question_te)))
           OR toLower(trim(coalesce(existing.question, ''))) =
                toLower(trim($legacy_question))
        WITH existing LIMIT 1
        SET existing += $properties
        RETURN existing.id AS question_id
        """,
        question_hash=question["question_hash"],
        question_en=question["question_en"],
        question_te=question["question_te"],
        legacy_question=question["question_en"] or question["question_te"],
        properties=properties,
    ).single()
    if existing:
        return False

    tx.run(
        """
        CREATE (q:PreviousQuestion)
        SET q = $properties, q.created_at = datetime()
        """,
        properties=properties,
    ).consume()
    return True


def store_question(driver, question: dict, source_name: str, page_number: int) -> bool:
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        return session.execute_write(
            _store_question, question, source_name, page_number
        )


def show_progress(
    current: int,
    total: int,
    question_pages: int,
    extracted: int,
    added: int,
    skipped: int,
) -> None:
    width = 24
    filled = int(width * current / total) if total else width
    bar = "=" * filled + " " * (width - filled)
    line = (
        f"Pages [{bar}] {current}/{total} "
        f"| question pages {question_pages}, found {extracted}, "
        f"added {added}, duplicates skipped {skipped}"
    )
    print(line, file=sys.stderr, flush=True)


def run_ingestion(
    pdf_path: Path,
    driver,
    settings: dict,
    client: httpx.Client,
    progress=show_progress,
) -> tuple[int, int]:
    reader = PdfReader(str(pdf_path))
    pdf_document = pymupdf.open(str(pdf_path))
    added = skipped = pages_with_text = question_pages = extracted = 0
    seen_hashes = set()
    source_name = pdf_path.name
    ocr_engine = None
    extraction_failures = 0

    try:
        for page_number, page in enumerate(reader.pages, start=1):
            native_text = (page.extract_text() or "").strip()
            text = native_text
            extraction_method = "native"
            if not TELUGU_SCRIPT_RE.search(native_text):
                if ocr_engine is None:
                    ocr_engine = tesseract_command()
                text = ocr_page(pdf_document[page_number - 1], ocr_engine)
                extraction_method = "tesseract"
            logger.info(
                "PDF_PAGE_TEXT page=%d method=%s chars=%d\n"
                "----- BEGIN PAGE %d TEXT -----\n%s\n"
                "----- END PAGE %d TEXT -----",
                page_number,
                extraction_method,
                len(text),
                page_number,
                text or "[no text extracted]",
                page_number,
            )
            if text:
                pages_with_text += 1

            if is_question_page(text):
                question_pages += 1
                try:
                    questions = generate_questions(text, settings, client)
                except ModelResponseError as exc:
                    extraction_failures += 1
                    logger.error(
                        "QUESTION_PARSE_FAILED page=%d extracted_text_chars=%d error=%s",
                        page_number,
                        len(text),
                        exc,
                    )
                    questions = []
                extracted += len(questions)
                for question in questions:
                    digest = question["question_hash"]
                    if digest in seen_hashes:
                        skipped += 1
                        continue
                    seen_hashes.add(digest)
                    if store_question(driver, question, source_name, page_number):
                        added += 1
                    else:
                        skipped += 1
            progress(
                page_number,
                len(reader.pages),
                question_pages,
                extracted,
                added,
                skipped,
            )
    finally:
        pdf_document.close()

    if not pages_with_text:
        raise ValueError(
            "No text could be extracted from the PDF or OCR. Check the paper and OCR setup."
        )
    if not extracted:
        raise ValueError(
            f"No questions were extracted (question pages detected: {question_pages}, "
            f"questions found: 0, duplicates: 0, malformed model responses: "
            f"{extraction_failures}). Review PDF_PAGE_TEXT and QUESTION_PARSE_FAILED logs."
        )
    return added, skipped


def load_settings() -> dict:
    use_model = parse_model(os.getenv("USE_MODEL", "OLLAMA"))
    settings = {
        "use_model": use_model,
        "grok_api_key": os.getenv("GROK_API_KEY", "").strip(),
        "grok_endpoint": os.getenv(
            "GROK_ENDPOINT", "https://api.groq.com/openai/v1/chat/completions"
        ),
        "grok_model": os.getenv("GROK_MODEL", "").strip(),
        "gemini_api_key": os.getenv("GEMINI_API_KEY", "").strip(),
        "gemini_endpoint": os.getenv(
            "GEMINI_ENDPOINT",
            "https://generativelanguage.googleapis.com/v1beta/models",
        ),
        "gemini_model": os.getenv("GEMINI_MODEL", "models/gemini-3.8-flash").strip(),
        "ollama_endpoint": os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "").strip(),
        "timeout": float(os.getenv("PREVIOUS_INGEST_TIMEOUT_SECONDS", "120")),
        "max_output_tokens": int(os.getenv("PREVIOUS_INGEST_MAX_OUTPUT_TOKENS", "4096")),
        "context_tokens": int(os.getenv("OLLAMA_CONTEXT_TOKENS", "8192")),
        "rate_limit_max_attempts": max(
            1, int(os.getenv("PREVIOUS_INGEST_RATE_LIMIT_ATTEMPTS", "5"))
        ),
        "rate_limit_max_wait_seconds": max(
            1, float(os.getenv("PREVIOUS_INGEST_RATE_LIMIT_MAX_WAIT_SECONDS", "120"))
        ),
    }
    required = {
        "GROK": ("grok_api_key", "GROK_API_KEY", "grok_model", "GROK_MODEL"),
        "GEMINI": ("gemini_api_key", "GEMINI_API_KEY", "gemini_model", "GEMINI_MODEL"),
        "OLLAMA": ("ollama_model", "OLLAMA_MODEL"),
    }[use_model]
    missing = [
        required[index + 1]
        for index in range(0, len(required), 2)
        if not settings[required[index]]
    ]
    if missing:
        raise ValueError(f"USE_MODEL={use_model} requires " + " and ".join(missing))
    return settings


def resolve_pdf_path(value: str) -> Path:
    pdf_path = Path(value).expanduser()
    if not pdf_path.is_absolute() and not pdf_path.is_file():
        project_pdf = Path(__file__).resolve().parent / "my_pdfs" / pdf_path
        if project_pdf.is_file():
            pdf_path = project_pdf
    pdf_path = pdf_path.resolve()
    if pdf_path.suffix.lower() != ".pdf" or not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {value!r}. Pass an existing PDF path or place the file "
            "in ./my_pdfs."
        )
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, help="Path to one previous-paper PDF")
    args = parser.parse_args()
    try:
        pdf_path = resolve_pdf_path(args.pdf)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    settings = load_settings()
    logger.info(
        "Ingesting %s with %s (%s)",
        pdf_path.name,
        settings["use_model"],
        settings[f"{settings['use_model'].lower()}_model"],
    )
    driver = None
    try:
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "change-me"),
            ),
        )
        driver.verify_connectivity()
        try:
            with httpx.Client() as client:
                added, skipped = run_ingestion(pdf_path, driver, settings, client)
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Provider request failed with HTTP %d after the configured retry policy: %s",
                exc.response.status_code,
                " ".join(exc.response.text.split())[:1000],
            )
            raise SystemExit(1) from None
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
            logger.error("%s", exc)
            raise SystemExit(1) from None
        logger.info(
            "Finished %s: %d question(s) added, %d duplicate(s) skipped",
            pdf_path.name,
            added,
            skipped,
        )
    finally:
        if driver is not None:
            driver.close()


if __name__ == "__main__":
    main()

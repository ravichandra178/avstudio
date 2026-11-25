import json
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from neo4j.exceptions import Neo4jError

from .ai import AIExtractionError, AIProvider
from .config import get_settings
from .graph import KnowledgeGraph
from .language import is_language_text
from .models import (
    ChatRequest,
    ChatResponse,
    GraphResponse,
    LearningPlanRequest,
    LearningPlanResponse,
    QuestionLikeRequest,
    QuestionRequest,
    QuestionResponse,
    ScreeningAttemptRequest,
    ScreeningAttemptResponse,
    ScreeningQuestion,
    SearchResult,
    StudyProgress,
    StudySessionRequest,
    StudySessionResponse,
    TranslationRequest,
    TranslationResponse,
    UploadResponse,
)
from .prompts import (
    FALLBACK_CONTEXT,
    FALLBACK_LABEL,
    chat_prompt,
    learning_plan_prompt,
    question_prompt,
)
from .storage import DocumentStorage
from .tasks import ingest_document
from .tracing import flush_traces

settings = get_settings()
logger = logging.getLogger(__name__)
_TELUGU_SCRIPT = re.compile(r"[\u0c00-\u0c7f]")


def lesson_paragraphs(value: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", value.strip()) if paragraph.strip()]


def passage_summary_fallback(
    topic: str, passages: list[dict], related_terms: list[str] | None = None
) -> tuple[str, str]:
    source_paragraphs = []
    for passage in passages:
        text = str(passage.get("text") or "").replace("●", "\n")
        for paragraph in re.split(r"\n\s*\n", text):
            cleaned = re.sub(r"\s+", " ", paragraph).strip(" \t-*•")
            if cleaned and len(cleaned.split()) >= 5:
                source_paragraphs.append(cleaned)

    sentences = []
    for paragraph in source_paragraphs:
        sentences.extend(
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph)
            if len(sentence.split()) >= 5
        )

    topic_terms = [
        topic,
        *re.findall(r"\(([^)]+)\)", topic),
        *(related_terms or []),
    ]
    topic_matches = [
        sentence
        for sentence in sentences
        if any(
            term.strip().rstrip(".").casefold() in sentence.casefold()
            for term in topic_terms
            if term.strip()
        )
    ]
    ordered_sentences = topic_matches if len(topic_matches) >= 2 else sentences
    selected = ordered_sentences[:9]
    if len(selected) >= 4:
        midpoint = (len(selected) + 1) // 2
        summary = " ".join(selected[:midpoint]) + "\n\n" + " ".join(selected[midpoint:])
    elif len(selected) >= 2:
        summary = " ".join(selected[:1]) + "\n\n" + " ".join(selected[1:])
    elif source_paragraphs:
        summary = "\n\n".join(source_paragraphs[:2])
    else:
        summary = (
            f"The indexed textbook passage for {topic} is not available as readable prose. "
            "Review the source page for the original material."
        )

    summary_parts = lesson_paragraphs(summary)
    if len(summary_parts) > 3:
        summary = "\n\n".join(summary_parts[:3])
    etext = selected[0] if selected else source_paragraphs[0] if source_paragraphs else summary_parts[0]
    return etext, summary


def related_fact_texts(topic: str, citations: list[dict]) -> list[str]:
    facts = []
    seen = set()
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        related_concepts = citation.get("related_concepts", [])
        if not isinstance(related_concepts, list):
            continue
        for related in related_concepts:
            if not isinstance(related, dict):
                continue
            relationship = str(related.get("relationship") or "").upper()
            fact = str(related.get("name") or "").strip()
            if (
                relationship == "MENTIONS"
                or not fact
                or fact.casefold() == topic.casefold()
                or fact.casefold() in seen
            ):
                continue
            seen.add(fact.casefold())
            facts.append(fact.rstrip(".") + ".")
    return facts


def related_fact_passages(topic: str, facts: list[str]) -> list[dict]:
    if not facts:
        return []
    return [
        {
            "text": f"Indexed textbook facts about {topic}:\n"
            + "\n".join(f"- {fact}" for fact in facts),
            "page": "not recorded",
        }
    ]


def generate_textbook_lesson(
    provider: AIProvider,
    topic: str,
    textbook: str,
    passages: list[dict],
    related_terms: list[str] | None = None,
) -> tuple[str, str, str, str]:
    source = "\n\n".join(
        f"Page {item.get('page') or 'not recorded'}: {item['text']}"
        for item in passages
    )
    try:
        generated = provider.structured_answer(
            "Prepare clear, connected study material for the requested topic using only "
            "the supplied textbook passages. Every factual statement must be directly "
            "supported by those passages. Do not add background knowledge, causes, "
            "outcomes, benefits, implementation details, examples, or exam advice unless "
            "the passages explicitly state them. You may reorganize and paraphrase, but "
            "preserve the source meaning and scope. Do not copy raw fragments or join "
            "unrelated sentences. "
            "Return JSON only with exactly four string keys: etext, summary, etext_te, "
            "and summary_te. "
            "etext should be a brief, complete introductory paragraph. "
            "summary must contain exactly 2 or 3 coherent paragraphs, with blank lines "
            "between paragraphs; aim for 120-180 words only when the source provides "
            "enough material. Never pad with repeated or unsupported claims. Each "
            "paragraph should develop a related idea, not be a sentence fragment. "
            "etext_te and summary_te must accurately translate their English fields "
            "into natural Telugu using Telugu script. Preserve the same paragraph count "
            "and blank-line separators in summary_te. "
            f"Topic: {topic}\nTextbook: {textbook}",
            source,
        )
    except (AIExtractionError, httpx.HTTPError) as exc:
        logger.warning(
            "TEXTBOOK_LESSON_GENERATION_FALLBACK topic=%s reason=provider_error error=%s",
            topic,
            exc,
        )
        etext, summary = passage_summary_fallback(topic, passages, related_terms)
        return etext, summary, "", ""

    if isinstance(generated, dict):
        etext = generated.get("etext")
        summary = generated.get("summary")
        if (
            isinstance(etext, str)
            and etext.strip()
            and isinstance(summary, str)
            and len(lesson_paragraphs(summary)) in {2, 3}
        ):
            return (
                etext.strip(),
                summary.strip(),
                str(generated.get("etext_te") or "").strip(),
                str(generated.get("summary_te") or "").strip(),
            )

    logger.warning(
        "TEXTBOOK_LESSON_GENERATION_FALLBACK topic=%s reason=invalid_model_output",
        topic,
    )
    etext, summary = passage_summary_fallback(topic, passages, related_terms)
    return etext, summary, "", ""


def bilingual_text_fields(
    provider: AIProvider,
    etext_en: str,
    summary_en: str,
    etext_te: str = "",
    summary_te: str = "",
) -> dict:
    def valid_telugu(value: str) -> bool:
        return bool(value) and is_language_text(value, "te") and bool(_TELUGU_SCRIPT.search(value))

    summary_paragraph_count = len(lesson_paragraphs(summary_en))
    valid_etext_te = valid_telugu(etext_te)
    valid_summary_te = valid_telugu(summary_te) and (
        summary_paragraph_count not in {2, 3}
        or len(lesson_paragraphs(summary_te)) == summary_paragraph_count
    )
    if valid_etext_te and valid_summary_te:
        return {
            "etext_en": etext_en,
            "etext_te": etext_te,
            "summary_en": summary_en,
            "summary_te": summary_te,
        }

    try:
        translated = provider.structured_answer(
            "Translate the supplied textbook lesson text faithfully into natural Telugu. "
            "Return JSON only with exactly etext_te and summary_te. "
            "Use Telugu script. Preserve the same paragraph count and blank-line paragraph "
            "separators in summary_te. Do not add facts or change the meaning.",
            json.dumps(
                {"etext": etext_en, "summary": summary_en},
                ensure_ascii=False,
            ),
        )
    except (AIExtractionError, httpx.HTTPError) as exc:
        logger.warning("TELUGU_LESSON_TRANSLATION_FALLBACK reason=provider_error error=%s", exc)
        translated = {}

    if not isinstance(translated, dict):
        translated = {}
    translated_etext = translated.get("etext_te")
    translated_summary = translated.get("summary_te")
    valid_translated_etext = valid_telugu(translated_etext)
    valid_translated_summary = valid_telugu(translated_summary) and (
        summary_paragraph_count not in {2, 3}
        or len(lesson_paragraphs(translated_summary)) == summary_paragraph_count
    )
    final_etext_te = translated_etext if valid_translated_etext else etext_te if valid_etext_te else ""
    final_summary_te = (
        translated_summary
        if valid_translated_summary
        else summary_te
        if valid_summary_te
        else ""
    )
    if not final_etext_te or not final_summary_te:
        logger.warning(
            "TELUGU_LESSON_TRANSLATION_PARTIAL etext_valid=%s summary_valid=%s",
            bool(final_etext_te),
            bool(final_summary_te),
        )
    return {
        "etext_en": etext_en,
        "etext_te": final_etext_te.strip(),
        "summary_en": summary_en,
        "summary_te": final_summary_te.strip(),
    }


def bilingual_question_fields(
    provider: AIProvider,
    question_en: str,
    options_en: list[str],
    answer_en: str,
    explanation_en: str,
    question_te: str | None = None,
    options_te: list[str] | None = None,
    answer_te: str | None = None,
    explanation_te: str | None = None,
) -> dict:
    def valid_telugu(value: str) -> bool:
        return is_language_text(value, "te") and bool(_TELUGU_SCRIPT.search(value))

    if (
        isinstance(question_te, str)
        and valid_telugu(question_te)
        and isinstance(options_te, list)
        and len(options_te) == len(options_en)
        and all(isinstance(item, str) and valid_telugu(item) for item in options_te)
        and (not explanation_en or isinstance(explanation_te, str) and valid_telugu(explanation_te))
        and (not answer_en or isinstance(answer_te, str) and valid_telugu(answer_te))
    ):
        return {
            "question_en": question_en,
            "options_en": options_en,
            "answer_en": answer_en,
            "explanation_en": explanation_en,
            "question_te": question_te,
            "options_te": options_te,
            "answer_te": answer_te or "",
            "explanation_te": explanation_te or "",
        }

    try:
        translated = provider.structured_answer(
            "Translate the supplied multiple-choice question faithfully into natural Telugu. "
            "Return JSON only with exactly question_te, options_te, answer_te, explanation_te. "
            "Keep all options in the same order and do not change their meaning. "
            "Use Telugu script for the translated question and choices. Preserve proper names "
            "and scientific terms accurately; do not add facts. If answer or explanation is "
            "empty, return an empty string for its translated field.",
            json.dumps(
                {
                    "question": question_en,
                    "options": options_en,
                    "answer": answer_en,
                    "explanation": explanation_en,
                },
                ensure_ascii=False,
            ),
        )
    except (AIExtractionError, httpx.HTTPError) as exc:
        logger.warning("TELUGU_TRANSLATION_FALLBACK reason=provider_error error=%s", exc)
        translated = {}

    if not isinstance(translated, dict):
        translated = {}
    translated_question = translated.get("question_te")
    translated_options = translated.get("options_te")
    translated_answer = translated.get("answer_te", "")
    translated_explanation = translated.get("explanation_te", "")
    if (
        not isinstance(translated_question, str)
        or not valid_telugu(translated_question)
        or not isinstance(translated_options, list)
        or len(translated_options) != len(options_en)
        or not all(isinstance(item, str) and valid_telugu(item) for item in translated_options)
        or not isinstance(translated_answer, str)
        or (answer_en and not valid_telugu(translated_answer))
        or not isinstance(translated_explanation, str)
        or (explanation_en and not valid_telugu(translated_explanation))
    ):
        logger.warning(
            "TELUGU_TRANSLATION_FALLBACK reason=invalid_response "
            "question=%r",
            question_en,
        )
        return {
            "question_en": question_en,
            "options_en": options_en,
            "answer_en": answer_en,
            "explanation_en": explanation_en,
            "question_te": "",
            "options_te": [],
            "answer_te": "",
            "explanation_te": "",
        }
    return {
        "question_en": question_en,
        "options_en": options_en,
        "answer_en": answer_en,
        "explanation_en": explanation_en,
        "question_te": translated_question.strip(),
        "options_te": [item.strip() for item in translated_options],
        "answer_te": translated_answer.strip(),
        "explanation_te": translated_explanation.strip(),
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        graph = KnowledgeGraph(settings)
        try:
            graph.ensure_schema()
        finally:
            graph.close()
    except Exception as exc:
        logger.warning("Neo4j schema initialization deferred on startup: %s", exc)
    try:
        yield
    finally:
        flush_traces()


app = FastAPI(title="Learning Studio", version="0.1.0", lifespan=lifespan)
FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

try:
    from mangum import Mangum

    handler = Mangum(app)
except ImportError:
    handler = app

@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(f"{settings.api_prefix}/translate", response_model=TranslationResponse)
def translate_ui_text(request: TranslationRequest) -> dict:
    if request.target_language == "en":
        return {"translations": request.texts}

    instruction = (
        "Translate every string in the supplied JSON array from English to natural "
        "Telugu (తెలుగు). Preserve names, numbers, percentages, URLs, page numbers, "
        "punctuation, symbols, icons, and HTML-free formatting. "
        "Do not add explanations or extra keys. Return a JSON object with a "
        '\"translations\" array containing exactly one Telugu string for every input '
        "string, in the same order."
    )
    context = json.dumps(request.texts, ensure_ascii=False)
    try:
        grok_key = settings.grok_api_key.strip()
        grok_model = settings.grok_model.strip()
        if grok_key and grok_model:
            grok_endpoint = settings.grok_endpoint.strip().rstrip("/")
            if grok_endpoint.endswith("/chat/completions"):
                grok_response_endpoint = grok_endpoint
            elif grok_endpoint.endswith("/v1"):
                grok_response_endpoint = f"{grok_endpoint}/chat/completions"
            else:
                grok_response_endpoint = grok_endpoint
            grok_response = httpx.post(
                grok_response_endpoint,
                headers={"Authorization": f"Bearer {grok_key}"},
                json={
                    "model": grok_model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": "Translate UI text accurately and return JSON only."},
                        {"role": "user", "content": f"{instruction}\n\n{context}"},
                    ],
                    "response_format": {"type": "json_object"},
                },
                timeout=60,
            )
            grok_response.raise_for_status()
            grok_answer = grok_response.json()["choices"][0]["message"]["content"]
            result = json.loads(grok_answer.strip().removeprefix("```json").removesuffix("```").strip())
        else:
            result = AIProvider(settings, feature="ui-translation").structured_answer(
                instruction,
                context,
            )
    except (AIExtractionError, httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
        try:
            result = AIProvider(settings, feature="ui-translation").structured_answer(
                instruction,
                context,
            )
        except (AIExtractionError, httpx.HTTPError, KeyError, IndexError) as exc:
            raise HTTPException(status_code=502, detail=f"Could not translate UI text: {exc}") from exc

    translations = result.get("translations")
    if not isinstance(translations, list) or len(translations) != len(request.texts):
        raise HTTPException(status_code=502, detail="Translation provider returned an invalid result")
    return {
        "translations": [
            translation if isinstance(translation, str) else original
            for translation, original in zip(translations, request.texts)
        ]
    }


@app.get(f"{settings.api_prefix}/library")
def library() -> dict:
    graph = KnowledgeGraph(settings)
    try:
        result = graph.library()
        if not result["textbooks"] and settings.fallback_for_textbook:
            result["textbooks"] = [{
                "id": "fallback",
                "filename": settings.fallback_for_textbook,
                "topics": [],
                "fallback": True,
            }]
        return {"group": settings.study_group, **result}
    finally:
        graph.close()


@app.post(f"{settings.api_prefix}/documents", response_model=UploadResponse, status_code=202)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:  # noqa: B008
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are supported")
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="PDF exceeds the upload size limit")
    document_id = str(uuid.uuid4())
    key = DocumentStorage(settings).put(document_id, file.filename or "document.pdf", content)
    try:
        ingest_document.delay(document_id, file.filename or "document.pdf", key)
    except Exception as exc:
        logger.warning("Background task queueing failed: %s", exc)
    return UploadResponse(document_id=document_id, status="queued")


@app.get(f"{settings.api_prefix}/search", response_model=list[SearchResult])
def search(q: str, limit: int = 20) -> list[dict]:
    if not q.strip() or not 1 <= limit <= 100:
        raise HTTPException(status_code=400, detail="q is required and limit must be 1..100")
    graph = KnowledgeGraph(settings)
    try:
        try:
            embedding = AIProvider(settings, feature="semantic-search").embed([q])[0]
            vector_results = graph.vector_search(embedding, limit)
        except (AIExtractionError, httpx.HTTPError, KeyError, IndexError, Neo4jError):
            vector_results = []
        if vector_results:
            return [
                {
                    "concept": item["text"][:120],
                    "description": item["text"],
                    "source_document_id": item["source_document_id"],
                    "page": item["page"],
                    "related_concepts": [],
                }
                for item in vector_results
            ]
        return graph.search(q, limit)
    finally:
        graph.close()


@app.get(f"{settings.api_prefix}/graph", response_model=GraphResponse)
def get_graph(document_id: str | None = None) -> dict:
    graph = KnowledgeGraph(settings)
    try:
        return graph.graph(document_id)
    finally:
        graph.close()


@app.post(f"{settings.api_prefix}/sessions", response_model=StudySessionResponse)
def create_session(request: StudySessionRequest) -> StudySessionResponse:
    session_id = str(uuid.uuid4())
    graph = KnowledgeGraph(settings)
    try:
        graph.create_session(session_id, request.group, request.textbook)
    finally:
        graph.close()
    return StudySessionResponse(session_id=session_id)


@app.get(f"{settings.api_prefix}/sessions/{{session_id}}/progress", response_model=StudyProgress)
def study_progress(session_id: str) -> dict:
    graph = KnowledgeGraph(settings)
    try:
        return graph.progress(session_id)
    finally:
        graph.close()


@app.post(
    f"{settings.api_prefix}/sessions/{{session_id}}/learning-plan",
    response_model=LearningPlanResponse,
)
def learning_plan(session_id: str, request: LearningPlanRequest) -> LearningPlanResponse:
    graph = KnowledgeGraph(settings)
    try:
        session_context = graph.session_context(session_id)
        if not session_context:
            raise HTTPException(status_code=404, detail="Study session not found")
        textbook = session_context["textbook"]
        raw_topic = (request.topic or "").strip()
        related_from = (request.related_from or "").strip()

        # Step 1: Zero-lag fetch for selected entity question from Neo4j
        db_question = None
        source_mode = "selected_entity"
        if raw_topic and not related_from:
            db_question = graph.entity_question(session_id, textbook, raw_topic)

        # Asking from an entity means select a different, directly related entity first.
        if not db_question:
            anchor = related_from or raw_topic or graph.last_screening_entity(session_id)
            db_question = graph.related_learning_question(session_id, textbook, anchor)
            if db_question:
                source_mode = "related_entity" if anchor else "textbook_bank"
                logger.info(
                    "LEARNING_RELATED_QUESTION session=%s anchor=%s selected_entity=%s",
                    session_id,
                    anchor,
                    db_question.get("entity_name"),
                )

        # If found in database: return immediately with zero lag
        if db_question:
            q_text = str(db_question.get("question") or "").strip()
            raw_options = db_question.get("options") or []
            if isinstance(raw_options, str):
                raw_options = raw_options.strip()
                if raw_options.startswith("[") and raw_options.endswith("]"):
                    try:
                        parsed = json.loads(raw_options)
                        raw_options = parsed if isinstance(parsed, list) else [raw_options]
                    except Exception:
                        raw_options = [opt.strip() for opt in raw_options.split(",") if opt.strip()]
                else:
                    raw_options = [opt.strip() for opt in raw_options.split(",") if opt.strip()]
            elif not isinstance(raw_options, (list, tuple, set)):
                raw_options = []

            options = [str(opt).strip() for opt in raw_options if opt is not None and str(opt).strip()]

            # Only use db_question if it has valid question text and at least 2 options
            if q_text and len(options) >= 2:
                entity_name = db_question.get("entity_name") or raw_topic or textbook
                correct_option = db_question.get("correct_option")
                if correct_option is None or not isinstance(correct_option, int) or not (0 <= correct_option < len(options)):
                    answer = str(db_question.get("answer") or "").strip()
                    matched_idx = next(
                        (i for i, opt in enumerate(options) if opt and opt.lower() in answer.lower()),
                        0,
                    )
                    correct_option = matched_idx

                explanation = str(db_question.get("explanation") or db_question.get("answer") or "").strip()
                try:
                    bilingual = bilingual_question_fields(
                        AIProvider(
                            settings,
                            session_id=session_id,
                            feature="question-translation",
                        ),
                        q_text,
                        options,
                        str(db_question.get("answer_en") or db_question.get("answer") or ""),
                        explanation,
                        db_question.get("question_te"),
                        db_question.get("options_te"),
                        db_question.get("answer_te"),
                        db_question.get("explanation_te"),
                    )
                except (AIExtractionError, httpx.HTTPError) as exc:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Could not create the Telugu question translation: {exc}",
                    ) from exc
                citations = db_question.get("citations")
                if not isinstance(citations, list):
                    citations = []

                # Retrieve textbook passages for the entity
                passages = graph.textbook_passages(textbook, topic=entity_name, limit=2)
                if not passages and raw_topic and raw_topic != entity_name:
                    passages = graph.textbook_passages(textbook, topic=raw_topic, limit=2)
                related_facts = related_fact_texts(entity_name, citations)
                if not passages:
                    for fact in related_facts:
                        passages = graph.textbook_passages(textbook, topic=fact, limit=2)
                        if passages:
                            break
                if not passages:
                    passages = related_fact_passages(entity_name, related_facts)

                etext_te = ""
                summary_te = ""
                if passages:
                    etext, summary, etext_te, summary_te = generate_textbook_lesson(
                        AIProvider(
                            settings,
                            session_id=session_id,
                            feature="lesson-generation",
                        ),
                        entity_name,
                        textbook,
                        passages,
                        related_facts,
                    )
                    etext = redact_quiz_answer(
                        etext,
                        options,
                        correct_option,
                    )
                    if not etext:
                        etext = (
                            f"Review the textbook material about {entity_name} before "
                            "answering the knowledge check."
                        )
                else:
                    etext = (
                        f"No descriptive passage about {entity_name} was found in "
                        f"{textbook}. The index only contains an entity mention, so it "
                        "cannot provide a textbook-grounded summary."
                    )
                    summary = "No descriptive textbook passage is indexed for this topic."

                lesson_texts = bilingual_text_fields(
                    AIProvider(
                        settings,
                        session_id=session_id,
                        feature="lesson-translation",
                    ),
                    etext,
                    summary,
                    etext_te,
                    summary_te,
                )

                # Persist screening question to session and mark as asked
                screening_question_id = str(uuid.uuid4())
                graph.save_learning_question(
                    screening_question_id,
                    session_id,
                    entity_name,
                    request.mode,
                    q_text,
                    options,
                    correct_option,
                    explanation,
                    settings.question_level,
                    textbook,
                    entity_name=entity_name,
                    study_question_id=db_question.get("question_id"),
                    question_te=bilingual["question_te"],
                    options_te=bilingual["options_te"],
                    explanation_te=bilingual["explanation_te"],
                    etext_en=lesson_texts["etext_en"],
                    etext_te=lesson_texts["etext_te"],
                    summary_en=lesson_texts["summary_en"],
                    summary_te=lesson_texts["summary_te"],
                )
                if (
                    db_question.get("question_id")
                    and bilingual["question_te"]
                    and len(bilingual["options_te"]) == len(options)
                ):
                    graph.save_study_question_translation(
                        db_question["question_id"],
                        bilingual["question_te"],
                        bilingual["options_te"],
                        bilingual["answer_te"],
                        bilingual["explanation_te"],
                    )
                    graph.mark_question_asked(session_id, db_question["question_id"])
                try:
                    next_question = graph.related_learning_question(
                        session_id, textbook, entity_name
                    )
                except Neo4jError:
                    logger.exception(
                        "LEARNING_RELATED_SUGGESTION_FAILED session=%s entity=%s",
                        session_id,
                        entity_name,
                    )
                    next_topic = None
                else:
                    candidate_topic = (
                        str(next_question["entity_name"]).strip()
                        if next_question and next_question.get("entity_name")
                        else ""
                    )
                    next_topic = (
                        candidate_topic
                        if candidate_topic.casefold() != entity_name.casefold()
                        else None
                    )

                logger.info(
                    "LEARNING_QUESTION_ZERO_LAG session=%s source=%s entity=%s question_id=%s",
                    session_id,
                    source_mode,
                    entity_name,
                    db_question.get("question_id"),
                )

                return LearningPlanResponse(
                    session_id=session_id,
                    mode=request.mode,
                    topic=entity_name,
                    next_topic=next_topic,
                    etext=etext,
                    summary=summary,
                    etext_en=lesson_texts["etext_en"],
                    etext_te=lesson_texts["etext_te"],
                    summary_en=lesson_texts["summary_en"],
                    summary_te=lesson_texts["summary_te"],
                    question=ScreeningQuestion(
                        question_id=screening_question_id,
                        question=q_text,
                        options=options,
                        question_en=bilingual["question_en"],
                        question_te=bilingual["question_te"],
                        options_en=bilingual["options_en"],
                        options_te=bilingual["options_te"],
                    ),
                    citations=citations,
                    source="textbook",
                )
            else:
                logger.warning(
                    "LEARNING_QUESTION_INVALID session=%s question_id=%s raw_options=%r; falling back to LLM",
                    session_id,
                    db_question.get("question_id"),
                    db_question.get("options"),
                )

        if related_from:
            raise HTTPException(
                status_code=404,
                detail=f"No unanswered question for an entity other than {related_from} was found in this textbook",
            )

        # Step 3: Fallback 2 - Use LLM when no questions found in Neo4j
        logger.info("LEARNING_QUESTION_FALLBACK_LLM session=%s topic=%s", session_id, raw_topic)
        topic = raw_topic
        citations = (
            graph.search(topic, limit=8, textbook=session_context["textbook"])
            if topic
            else graph.textbook_topics(session_context["textbook"], limit=8)
        )
        passages = graph.textbook_passages(
            session_context["textbook"],
            topic=topic or None,
            limit=6,
        )
        entity_evidence = graph.textbook_entity_evidence(
            session_context["textbook"],
            topic=topic or None,
            limit=24,
        )
        if not passages and not entity_evidence and not settings.fallback_for_textbook:
            raise HTTPException(
                status_code=404,
                detail="No supporting textbook passage was found for this lesson",
            )
        context = "\n".join(
            [
                f"Group: {session_context.get('group')}",
                f"Textbook: {session_context.get('textbook')}",
                f"Exam level: {settings.question_level}",
                *(f"Concept: {item['concept']} (page {item.get('page')}): {item.get('description')}" for item in citations),
                *(
                    f"TEXTBOOK PASSAGE (page {item.get('page')}): {item['text']}"
                    for item in passages
                ),
            ]
        )
        passage_text = "\n".join(
            f"(page {item.get('page')}): {item['text']}" for item in passages
        )
        evidence_text = "\n".join(
            f"{item['entity_type']}: {item['entity_name']} "
            f"[{item['relationship']}]"
            for item in entity_evidence
        )
        ground_truth = passage_text or evidence_text
        source = "textbook" if ground_truth else "fallback"
        logger.info(
            "LEARNING_PLAN_SOURCE session=%s textbook=%s source=%s passages=%d entities=%d",
            session_id,
            session_context["textbook"],
            source,
            len(passages),
            len(entity_evidence),
        )
        if not ground_truth:
            context = (
                f"Fallback textbook scope: {settings.fallback_for_textbook}\n"
                f"{FALLBACK_CONTEXT}\n"
            )
        instruction = learning_plan_prompt(
            request.mode,
            topic,
            settings.study_group,
            settings.question_level,
            "Telugu" if request.language == "te" else "English",
            passages=ground_truth,
            fallback=not ground_truth,
        )
        provider = AIProvider(
            settings,
            session_id=session_id,
            feature="learning-plan",
        )
        try:
            generated = provider.structured_answer(
                instruction + (
                    " This is an explicitly configured AI fallback; label the summary as fallback."
                    if not ground_truth else ""
                ),
                context,
            )
        except (AIExtractionError, httpx.HTTPError) as exc:
            raise HTTPException(status_code=502, detail=f"Could not generate textbook lesson: {exc}") from exc
        options = generated.get("options")
        correct_option = generated.get("correct_option")
        question_text = generated.get("question")
        if (
            not isinstance(question_text, str)
            or not question_text.strip()
            or not isinstance(options, list)
            or len(options) < 2
            or len(options) > 6
            or not isinstance(correct_option, int)
        ):
            raise HTTPException(status_code=502, detail="AI returned an invalid screening question")
        if not 0 <= correct_option < len(options):
            raise HTTPException(status_code=502, detail="AI returned an invalid answer index")
        options = [str(option).strip() for option in options]
        question_text = question_text.strip()
        explanation = str(generated.get("explanation", "")).strip()
        generated_etext = str(generated.get("etext", "")).strip()
        generated_summary = str(generated.get("summary", "")).strip()
        if len(lesson_paragraphs(generated_summary)) not in {2, 3}:
            logger.warning(
                "LEARNING_PLAN_SUMMARY_FALLBACK session=%s topic=%s reason=invalid_model_output",
                session_id,
                topic,
            )
            if passages:
                fallback_etext, generated_summary = passage_summary_fallback(topic, passages)
                generated_etext = generated_etext or fallback_etext
            else:
                generated_etext = (
                    f"No descriptive textbook passage is indexed for {topic or 'this topic'}."
                )
                generated_summary = (
                    generated_etext
                    + "\n\nThe available index does not contain enough source material "
                    "to produce a textbook-grounded summary."
                )
        generated_options_te = generated.get("options_te")
        generated_answer_te = None
        if (
            isinstance(generated_options_te, list)
            and len(generated_options_te) == len(options)
            and all(isinstance(option, str) for option in generated_options_te)
        ):
            generated_answer_te = (
                f"{generated_options_te[correct_option]}. "
                f"{generated.get('explanation_te', '')}"
            ).strip()
        try:
            bilingual = bilingual_question_fields(
                AIProvider(
                    settings,
                    session_id=session_id,
                    feature="question-translation",
                ),
                question_text,
                options,
                f"{options[correct_option]}. {explanation}".strip(),
                explanation,
                generated.get("question_te"),
                generated_options_te,
                generated_answer_te,
                generated.get("explanation_te"),
            )
        except (AIExtractionError, httpx.HTTPError) as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Could not create the Telugu question translation: {exc}",
            ) from exc
        lesson_texts = bilingual_text_fields(
            AIProvider(
                settings,
                session_id=session_id,
                feature="lesson-translation",
            ),
            generated_etext,
            generated_summary,
            generated.get("etext_te", ""),
            generated.get("summary_te", ""),
        )
        question_id = str(uuid.uuid4())
        graph.save_learning_question(
            question_id,
            session_id,
            request.topic,
            request.mode,
            question_text,
            options,
            correct_option,
            explanation,
            settings.question_level,
            session_context["textbook"],
            entity_name=request.topic,
            question_te=bilingual["question_te"],
            options_te=bilingual["options_te"],
            explanation_te=bilingual["explanation_te"],
            etext_en=lesson_texts["etext_en"],
            etext_te=lesson_texts["etext_te"],
            summary_en=lesson_texts["summary_en"],
            summary_te=lesson_texts["summary_te"],
        )
        return LearningPlanResponse(
            session_id=session_id,
            mode=request.mode,
            topic=topic or session_context["textbook"],
            etext=(
                lesson_texts["etext_te"]
                if request.language == "te" and lesson_texts["etext_te"]
                else lesson_texts["etext_en"]
            ),
            summary=(
                lesson_texts["summary_te"]
                if request.language == "te" and lesson_texts["summary_te"]
                else lesson_texts["summary_en"]
            ),
            etext_en=lesson_texts["etext_en"],
            etext_te=lesson_texts["etext_te"],
            summary_en=lesson_texts["summary_en"],
            summary_te=lesson_texts["summary_te"],
            question=ScreeningQuestion(
                question_id=question_id,
                question=question_text,
                options=options,
                question_en=bilingual["question_en"],
                question_te=bilingual["question_te"],
                options_en=bilingual["options_en"],
                options_te=bilingual["options_te"],
            ),
            citations=citations,
            source=source,
        )
    finally:
        graph.close()


@app.post(
    f"{settings.api_prefix}/sessions/{{session_id}}/screening/{{question_id}}/attempt",
    response_model=ScreeningAttemptResponse,
)
def screening_attempt(
    session_id: str, question_id: str, request: ScreeningAttemptRequest
) -> ScreeningAttemptResponse:
    graph = KnowledgeGraph(settings)
    try:
        question = graph.screening_question(session_id, question_id)
        if not question:
            raise HTTPException(status_code=404, detail="Screening question not found")
        options = question.get("options", [])
        if request.option_index >= len(options):
            raise HTTPException(status_code=400, detail="Invalid option index")
        attempts_used = int(question["attempts_used"])
        explanation_en = str(
            question.get("explanation_en") or question.get("explanation") or ""
        ).strip()
        explanation_te = str(
            question.get("explanation_te") or ""
        ).strip()

        def feedback_text(correct: bool, revealed: bool) -> tuple[str, str]:
            if correct:
                return "Correct! Great job.", "సరైన సమాధానం! బాగా చేశారు."
            if not revealed:
                return (
                    "Incorrect. Try again (1 attempt remaining).",
                    "సమాధానం తప్పు. మళ్లీ ప్రయత్నించండి (ఇంకా 1 ప్రయత్నం ఉంది).",
                )
            letter = chr(65 + int(question["correct_option"]))
            return (
                f"Two incorrect attempts. The correct answer was Option {letter}. "
                f"{explanation_en}",
                f"రెండు ప్రయత్నాలు తప్పాయి. సరైన సమాధానం ఎంపిక {letter}. "
                f"{explanation_te}",
            )

        if attempts_used >= 2 or question.get("resolved", False):
            is_correct = request.option_index == int(question["correct_option"])
            feedback_en, feedback_te = feedback_text(is_correct, True)
            return ScreeningAttemptResponse(
                correct=is_correct,
                attempts_used=attempts_used,
                attempts_remaining=0,
                answer_revealed=True,
                correct_option_index=int(question["correct_option"]),
                feedback=feedback_te if request.language == "te" else feedback_en,
                feedback_en=feedback_en,
                feedback_te=feedback_te,
                progress=min(100, attempts_used * 10),
            )
        question = graph.record_screening_attempt(session_id, question_id, request.option_index)
        if not question:
            raise HTTPException(status_code=409, detail="Screening question is already resolved")
        is_correct = request.option_index == int(question["correct_option"])
        attempts_used = int(question["attempts_used"])
        revealed = is_correct or attempts_used >= 2
        feedback_en, feedback_te = feedback_text(is_correct, revealed)

        return ScreeningAttemptResponse(
            correct=is_correct,
            attempts_used=attempts_used,
            attempts_remaining=max(0, 2 - attempts_used),
            answer_revealed=revealed,
            correct_option_index=int(question["correct_option"]) if revealed else None,
            feedback=feedback_te if request.language == "te" else feedback_en,
            feedback_en=feedback_en,
            feedback_te=feedback_te,
            progress=min(100, attempts_used * 10 + (50 if is_correct else 0)),
        )
    finally:
        graph.close()


def extract_entity_from_chat(message: str, fallback_topic: str | None = None) -> str:
    cleaned = message.strip()
    patterns = [
        r"^ask\s+from\s+(?:the\s+)?textbook\s+(?:about|on)\s*[:\-]?\s*\[?([^\]\?\.\n]+)\]?",
        r"^ask\s+from\s*[:\-]?\s*\[?([^\]\?\.\n]+)\]?",
        r"^ask\s+about\s*[:\-]?\s*\[?([^\]\?\.\n]+)\]?",
        r"^tell\s+me\s+about\s*[:\-]?\s*\[?([^\]\?\.\n]+)\]?",
        r"^summarize\s*[:\-]?\s*\[?([^\]\?\.\n]+)\]?",
        r"^what\s+(?:does\s+the\s+textbook\s+say\s+about|is)\s*[:\-]?\s*\[?([^\]\?\.\n]+)\]?",
        r"^explain\s*[:\-]?\s*\[?([^\]\?\.\n]+)\]?",
    ]
    for pat in patterns:
        m = re.match(pat, cleaned, re.IGNORECASE)
        if m:
            candidate = re.sub(
                r"\s+in\s+(?:two|2)\s+paragraphs?$",
                "",
                m.group(1).strip().strip("'\"[]"),
                flags=re.IGNORECASE,
            ).strip()
            if candidate:
                return candidate
    return (fallback_topic or "").strip()


def redact_quiz_answer(text: str, options: list[str], correct_option: int) -> str:
    answer = options[correct_option].strip()
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
        if sentence.strip()
    ]
    safe_sentences = [
        sentence
        for sentence in sentences
        if not re.search(r"\b(?:correct\s+)?answer\s+is\b", sentence, re.IGNORECASE)
    ]
    if answer:
        answer_pattern = re.compile(rf"(?<!\w){re.escape(answer)}(?!\w)", re.IGNORECASE)
        safe_sentences = [answer_pattern.sub("[hidden]", sentence) for sentence in safe_sentences]
    return " ".join(safe_sentences).strip()


def textbook_topic_excerpt(text: str, topic: str, max_words: int) -> str:
    excerpt = re.sub(r"\s+", " ", text).strip()
    topic_match = re.search(re.escape(topic), excerpt, re.IGNORECASE) if topic else None
    if topic_match:
        excerpt = excerpt[topic_match.start():]

    bullets = re.split(r"\s*[●•]\s*", excerpt)
    if len(bullets) > 1:
        excerpt = " ".join(part.strip() for part in bullets[:4] if part.strip())
    else:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", excerpt)
            if sentence.strip()
        ]
        excerpt = " ".join(sentences)

    words = excerpt.split()
    if len(words) <= max_words:
        return excerpt

    bounded = " ".join(words[:max_words])
    sentence_ends = list(re.finditer(r"[.!?](?=\s|$)", bounded))
    if sentence_ends:
        return bounded[: sentence_ends[-1].end()]
    return f"{bounded.rstrip(' ,;:')}…"


def synthesize_fast_two_paragraphs(
    topic: str,
    textbook: str,
    passages: list[dict],
    evidence: list[dict],
    citations: list[dict],
) -> str:
    display_topic = topic or "This subject"

    # Keep chat responses bounded to facts actually returned by the textbook lookup.
    p1 = ""
    if passages:
        raw = passages[0]["text"].replace("\n", " ").strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if s.strip()]
        p1 = " ".join(sentences[:2])
        if not p1.endswith("."):
            p1 += "."
    elif citations and citations[0].get("description"):
        p1 = f"{display_topic}: {citations[0]['description']}".strip()
    elif evidence:
        entity_types = ", ".join(sorted({item["entity_type"] for item in evidence if item.get("entity_type")}))
        p1 = f"The textbook knowledge graph identifies {display_topic} as: {entity_types}."
    else:
        p1 = f"No matching passage or entity details for {display_topic} were found in {textbook}."

    p2 = ""
    if len(passages) > 1:
        raw2 = passages[1]["text"].replace("\n", " ").strip()
        sentences2 = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw2) if s.strip()]
        p2 = " ".join(sentences2[:2])
        if p2 and not p2.endswith("."):
            p2 += "."
    elif evidence:
        related = sorted(
            {
                f"{item['relationship']} {item['entity_name']}"
                for item in evidence
                if item.get("relationship") and item.get("entity_name")
                and item["relationship"] != "MENTIONS"
            }
        )
        if related:
            p2 = f"Direct textbook graph relationships: {', '.join(related[:4])}."
        else:
            p2 = (
                "The index contains an entity mention but no descriptive textbook passage "
                "or additional factual relationship for this topic."
            )
    elif citations:
        p2 = "The indexed textbook has no further details for this topic."
    else:
        p2 = "No supporting textbook evidence was found, so no additional facts are being inferred."

    return f"{p1}\n\n{p2}"


@app.post(f"{settings.api_prefix}/sessions/{{session_id}}/chat", response_model=ChatResponse)
def chat(session_id: str, request: ChatRequest) -> ChatResponse:
    clean_msg = request.message.strip().lower()
    change_triggers = {
        "change question",
        "next question",
        "another question",
        "new question",
        "skip question",
        "give me another question",
        "different question",
    }
    graph = KnowledgeGraph(settings)
    try:
        session_context = graph.session_context(session_id)
        if not session_context:
            raise HTTPException(status_code=404, detail="Study session not found")
        textbook = session_context.get("textbook", "")

        # Fast path 1: Change question intent
        if clean_msg in change_triggers or any(clean_msg.startswith(t) for t in change_triggers):
            answer = (
                "I've updated the question player with a new question from your textbook.\n\n"
                "You can attempt the new knowledge check above or ask more questions here anytime!"
            )
            graph.save_message(session_id, "user", request.message)
            graph.save_message(session_id, "assistant", answer)
            return ChatResponse(session_id=session_id, answer=answer, citations=[], action="change_question")

        # Fast path 2: App-aware entity resolution
        active_topic = (request.topic or "").strip() or graph.last_screening_entity(session_id) or ""
        entity_name = extract_entity_from_chat(request.message, active_topic)

        citations = []
        entity_evidence = []
        passages = []

        if entity_name:
            citations = graph.search(entity_name, limit=8, textbook=textbook)
            entity_evidence = graph.textbook_entity_evidence(textbook, topic=entity_name, limit=16)
            passages = graph.textbook_passages(textbook, topic=entity_name, limit=4)

        if not citations:
            citations = graph.search(request.message, limit=8, textbook=textbook)
        if not passages:
            passages = graph.textbook_passages(textbook, topic=request.message, limit=4)

        if not citations and not passages and not entity_evidence and active_topic and active_topic != entity_name:
            citations = graph.search(active_topic, limit=8, textbook=textbook)
            entity_evidence = graph.textbook_entity_evidence(textbook, topic=active_topic, limit=16)
            passages = graph.textbook_passages(textbook, topic=active_topic, limit=4)

        history = graph.recent_messages(session_id)
        passage_summary = "\n".join(f"(page {item.get('page', '')}): {item['text']}" for item in passages[:3])
        evidence_summary = "\n".join(
            f"{item['entity_type']}: {item['entity_name']} [{item['relationship']}]"
            for item in entity_evidence[:8]
        )

        ground_context = passage_summary or evidence_summary or "\n".join(
            f"{c['concept']}: {c.get('description', '')}" for c in citations[:4]
        )

        instruction = chat_prompt(
            request.message,
            "Telugu" if request.language == "te" else "English",
            passages=ground_context,
            entity=entity_name or active_topic,
            textbook=textbook,
        )

        context_blocks = [
            f"Study group: {session_context.get('group')}",
            f"Textbook: {textbook}",
            *(f"{item['role']}: {item['content']}" for item in history[-4:]),
        ]
        if entity_name:
            context_blocks.append(f"Focus Entity: {entity_name}")
        if ground_context:
            context_blocks.append(f"Ground Truth:\n{ground_context}")
        context = "\n".join(context_blocks)

        has_descriptive_evidence = any(
            passage.get("text", "").strip() for passage in passages
        ) or any(
            citation.get("description") for citation in citations
        ) or any(
            item.get("relationship")
            and item.get("relationship", "").upper() != "MENTIONS"
            and item.get("entity_name")
            for item in entity_evidence
        )
        has_graph_evidence = bool(passages or entity_evidence or citations)

        if has_descriptive_evidence:
            logger.info(
                "CHAT_SOURCE_TEXTBOOK_FAST session=%s textbook=%s topic=%s passages=%d entities=%d citations=%d",
                session_id,
                textbook,
                entity_name or active_topic,
                len(passages),
                len(entity_evidence),
                len(citations),
            )
            answer = synthesize_fast_two_paragraphs(
                entity_name or active_topic or request.message,
                textbook,
                passages,
                entity_evidence,
                citations,
            )
        elif has_graph_evidence:
            logger.info(
                "CHAT_SOURCE_GEMINI_SEARCH_FALLBACK session=%s textbook=%s topic=%s timeout_seconds=%s",
                session_id,
                textbook,
                entity_name or active_topic,
                settings.chat_fallback_timeout_seconds,
            )
            search_instruction = (
                f"{instruction}\n\n"
                "The indexed textbook contains only a mention of this entity, not explanatory "
                "facts. Use Google Search grounding to research it. Clearly identify the response "
                "as an AI-generated summary based on external sources, and do not present external "
                "details as facts found in the textbook."
            )
            try:
                answer = AIProvider(
                    settings,
                    session_id=session_id,
                    feature="chat",
                ).gemini_search_answer(
                    search_instruction,
                    context,
                    timeout_seconds=settings.chat_fallback_timeout_seconds,
                )
            except (AIExtractionError, httpx.HTTPError) as exc:
                logger.error(
                    "CHAT_GEMINI_SEARCH_FALLBACK_FAILED session=%s error_type=%s error=%s",
                    session_id,
                    type(exc).__name__,
                    exc,
                )
                topic_label = entity_name or active_topic or request.message
                answer = (
                    f"I couldn't get an internet-grounded summary of {topic_label} "
                    "from Gemini just now.\n\n"
                    "The indexed textbook only contains a mention of this topic, not enough "
                    "detail to summarize it. Please try again shortly."
                )
        else:
            logger.info(
                "CHAT_SOURCE_AI_FALLBACK session=%s provider_order=%s timeout_seconds=4",
                session_id,
                settings.providers,
            )
            try:
                answer = AIProvider(
                    settings,
                    session_id=session_id,
                    feature="chat",
                ).answer(
                    instruction,
                    context,
                    timeout_seconds=min(float(settings.question_generation_timeout_seconds), 4.0),
                )
            except (AIExtractionError, httpx.HTTPError) as exc:
                logger.warning("CHAT_AI_FALLBACK_FAILED session=%s error=%s", session_id, exc)
                answer = synthesize_fast_two_paragraphs(
                    entity_name or active_topic or request.message,
                    textbook,
                    passages,
                    entity_evidence,
                    citations,
                )

        graph.save_message(session_id, "user", request.message)
        graph.save_message(session_id, "assistant", answer)
        return ChatResponse(session_id=session_id, answer=answer, citations=citations)
    finally:
        graph.close()


@app.post(
    f"{settings.api_prefix}/sessions/{{session_id}}/questions",
    response_model=QuestionResponse,
)
def play_question(session_id: str, request: QuestionRequest) -> QuestionResponse:
    graph = KnowledgeGraph(settings)
    try:
        history_mode = request.mode if request.mode == "textbook" else None
        history = graph.question_history(session_id, history_mode, request.document_id)
        session_context = graph.session_context(session_id)
        if not session_context:
            raise HTTPException(status_code=404, detail="Study session not found")
        if request.mode in {"previous", "textbook"}:
            bank = graph.question_bank(
                session_id,
                request.mode,
                settings.question_level,
                session_context["textbook"],
                language=request.language,
            )
            if bank:
                item = bank[0]
                question_en = str(
                    item.get("question_en") or item.get("question") or ""
                ).strip()
                raw_options_en = item.get("options_en") or item.get("options") or []
                options_en = (
                    [str(option).strip() for option in raw_options_en]
                    if isinstance(raw_options_en, (list, tuple))
                    else []
                )
                answer_en = str(
                    item.get("answer_en") or item.get("answer") or ""
                ).strip()
                bilingual = {
                    "question_en": question_en,
                    "options_en": options_en,
                    "answer_en": answer_en,
                    "question_te": item.get("question_te"),
                    "options_te": item.get("options_te"),
                    "answer_te": item.get("answer_te"),
                    "explanation_en": str(
                        item.get("explanation_en") or item.get("explanation") or ""
                    ),
                    "explanation_te": item.get("explanation_te"),
                }
                if request.mode == "textbook":
                    try:
                        bilingual = bilingual_question_fields(
                            AIProvider(
                                settings,
                                session_id=session_id,
                                feature="question-translation",
                            ),
                            question_en,
                            options_en,
                            answer_en,
                            bilingual["explanation_en"],
                            item.get("question_te"),
                            item.get("options_te"),
                            item.get("answer_te"),
                            item.get("explanation_te"),
                        )
                    except (AIExtractionError, httpx.HTTPError) as exc:
                        raise HTTPException(
                            status_code=502,
                            detail=f"Could not create the Telugu question translation: {exc}",
                        ) from exc
                    if (
                        bilingual["question_te"]
                        and len(bilingual["options_te"]) == len(options_en)
                    ):
                        graph.save_study_question_translation(
                            item["question_id"],
                            bilingual["question_te"],
                            bilingual["options_te"],
                            bilingual["answer_te"],
                            bilingual["explanation_te"],
                        )
                question = (
                    bilingual["question_te"]
                    if request.language == "te"
                    else bilingual["question_en"]
                )
                options = (
                    bilingual["options_te"]
                    if request.language == "te"
                    else bilingual["options_en"]
                )
                answer = (
                    bilingual["answer_te"]
                    if request.language == "te"
                    else bilingual["answer_en"]
                )
                return QuestionResponse(
                    question_id=str(uuid.uuid4()),
                    question=question,
                    options=options,
                    answer=answer,
                    question_en=bilingual["question_en"],
                    question_te=bilingual["question_te"],
                    options_en=bilingual["options_en"],
                    options_te=bilingual["options_te"],
                    answer_en=bilingual["answer_en"],
                    answer_te=bilingual["answer_te"],
                    mode=request.mode,
                    citations=item["citations"],
                    liked=bool(item.get("liked", False)),
                )
        if request.mode == "previous":
            raise HTTPException(
                status_code=404,
                detail=f"No {settings.question_level} previous-year questions found",
            )
        if request.mode == "repeated" and not history:
            raise HTTPException(status_code=404, detail="No repeated questions found")
        context_items = (
            graph.search(request.concept or "", limit=12, textbook=session_context["textbook"])
            if request.concept
            else graph.textbook_topics(session_context["textbook"], limit=12)
        )
        passages = graph.textbook_passages(
            session_context["textbook"],
            topic=request.concept or None,
            limit=6,
        )
        entity_evidence = graph.textbook_entity_evidence(
            session_context["textbook"],
            topic=request.concept or None,
            limit=24,
        )
        if (
            request.mode == "textbook"
            and not passages
            and not entity_evidence
            and not settings.fallback_for_textbook
        ):
            raise HTTPException(
                status_code=404,
                detail="No supporting textbook passage found for the selected textbook",
            )
        context = "\n".join(
            f"{item['concept']} (page {item.get('page')}): {item.get('description')}"
            for item in context_items
        )
        context += "\n".join(
            f"\nTEXTBOOK PASSAGE (page {item.get('page')}): {item['text']}"
            for item in passages
        )
        context += "\n".join(
            f"\nTEXTBOOK ENTITY ({item['entity_type']}): {item['entity_name']} "
            f"[{item['relationship']}]"
            for item in entity_evidence
        )
        prior = "\n".join(
            f"Q: {item['question']}\nOptions: {item.get('options', [])}\nA: {item['answer']}"
            for item in history
        )
        instruction = question_prompt(
            request.mode,
            settings.study_group,
            settings.question_level,
            "Telugu" if request.language == "te" else "English",
            passages=context,
        )
        logger.info("QUESTION_GENERATION_START session=%s textbook=%s provider_order=%s",
                    session_id, session_context["textbook"], settings.providers)
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            AIProvider(
                settings,
                session_id=session_id,
                feature="question-generation",
            ).structured_answer,
            instruction,
            f"{context}\n{prior}",
        )
        try:
            generated = future.result(timeout=settings.question_generation_timeout_seconds)
            logger.info("QUESTION_GENERATION_SUCCESS session=%s", session_id)
        except FutureTimeoutError:
            future.cancel()
            logger.warning(
                "QUESTION_GENERATION_TIMEOUT session=%s timeout_seconds=%s; trying liked fallback",
                session_id, settings.question_generation_timeout_seconds,
            )
            fallback = graph.liked_question(session_id, session_context["textbook"])
            if not fallback:
                raise HTTPException(
                    status_code=504,
                    detail="Question generation exceeded the 5-second window and no liked question is available",
                )
            logger.info("QUESTION_FALLBACK_LIKED session=%s question_id=%s",
                        session_id, fallback["question_id"])
            return QuestionResponse(
                question_id=fallback["question_id"],
                question=fallback["question"],
                options=fallback["options"],
                answer=fallback.get("answer"),
                mode=request.mode,
                citations=fallback["citations"],
                liked=True,
            )
        except (AIExtractionError, httpx.HTTPError) as exc:
            logger.warning("QUESTION_GENERATION_ERROR session=%s error=%s", session_id, exc)
            fallback = graph.liked_question(session_id, session_context["textbook"])
            if not fallback:
                raise HTTPException(status_code=502, detail=f"Could not generate MCQ: {exc}") from exc
            logger.info("QUESTION_FALLBACK_LIKED session=%s question_id=%s",
                        session_id, fallback["question_id"])
            return QuestionResponse(
                question_id=fallback["question_id"],
                question=fallback["question"],
                options=fallback["options"],
                answer=fallback.get("answer"),
                mode=request.mode,
                citations=fallback["citations"],
                liked=True,
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        question = generated.get("question")
        options = generated.get("options")
        correct_option = generated.get("correct_option")
        explanation = generated.get("explanation", "")
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(options, list)
            or len(options) != 4
            or not all(isinstance(option, str) and option.strip() for option in options)
            or not isinstance(correct_option, int)
            or not 0 <= correct_option < len(options)
        ):
            raise HTTPException(status_code=502, detail="AI returned an invalid MCQ")
        options = [option.strip() for option in options]
        answer = f"{options[correct_option]}. {explanation}".strip()
        generated_options_te = generated.get("options_te")
        generated_answer_te = None
        if (
            isinstance(generated_options_te, list)
            and len(generated_options_te) == len(options)
            and all(isinstance(option, str) for option in generated_options_te)
        ):
            generated_answer_te = (
                f"{generated_options_te[correct_option]}. "
                f"{generated.get('explanation_te', '')}"
            ).strip()
        try:
            bilingual = bilingual_question_fields(
                AIProvider(
                    settings,
                    session_id=session_id,
                    feature="question-translation",
                ),
                question.strip(),
                options,
                answer,
                str(explanation),
                generated.get("question_te"),
                generated_options_te,
                generated_answer_te,
                generated.get("explanation_te"),
            )
        except (AIExtractionError, httpx.HTTPError) as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Could not create the Telugu question translation: {exc}",
            ) from exc
        question_id = str(uuid.uuid4())
        graph.save_question(
            question_id,
            session_id,
            question.strip(),
            answer,
            options,
            correct_option,
            settings.question_level,
            session_context["textbook"],
            request.mode,
            request.document_id,
            context_items,
            question_te=bilingual["question_te"],
            options_te=bilingual["options_te"],
            answer_te=bilingual["answer_te"],
            explanation_en=str(explanation),
            explanation_te=bilingual["explanation_te"],
        )
        return QuestionResponse(
            question_id=question_id,
            question=(
                bilingual["question_te"]
                if request.language == "te"
                else bilingual["question_en"]
            ),
            options=(
                bilingual["options_te"]
                if request.language == "te"
                else bilingual["options_en"]
            ),
            answer=(
                bilingual["answer_te"]
                if request.language == "te"
                else bilingual["answer_en"]
            ),
            question_en=bilingual["question_en"],
            question_te=bilingual["question_te"],
            options_en=bilingual["options_en"],
            options_te=bilingual["options_te"],
            answer_en=bilingual["answer_en"],
            answer_te=bilingual["answer_te"],
            mode=request.mode,
            citations=context_items,
            liked=False,
        )
    finally:
        graph.close()


@app.post(
    f"{settings.api_prefix}/sessions/{{session_id}}/questions/{{question_id}}/like"
)
def like_question(
    session_id: str, question_id: str, request: QuestionLikeRequest
) -> dict[str, bool]:
    graph = KnowledgeGraph(settings)
    try:
        liked = graph.set_question_liked(session_id, question_id, request.liked)
        if not liked and request.liked:
            raise HTTPException(status_code=404, detail="Question not found")
        logger.info(
            "QUESTION_LIKE session=%s question_id=%s liked=%s",
            session_id,
            question_id,
            request.liked,
        )
        return {"liked": liked}
    finally:
        graph.close()


def _split_question(value: str) -> tuple[str, str]:
    """Parse the legacy question format for compatibility with existing callers."""
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    question = next(
        (line.split(":", 1)[1].strip() for line in lines if line.upper().startswith("QUESTION:")),
        "",
    )
    answer = next(
        (line.split(":", 1)[1].strip() for line in lines if line.upper().startswith("ANSWER:")),
        "",
    )
    if not question or not answer:
        raise HTTPException(status_code=502, detail="AI returned an invalid question format")
    return question, answer

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import re
import uuid

import httpx
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from neo4j.exceptions import Neo4jError

from .config import get_settings
from .graph import KnowledgeGraph
from .ai import AIExtractionError, AIProvider
from .models import (
    ChatRequest,
    ChatResponse,
    GraphResponse,
    QuestionRequest,
    QuestionResponse,
    SearchResult,
    StudySessionRequest,
    StudySessionResponse,
    StudyProgress,
    LearningPlanRequest,
    LearningPlanResponse,
    QuestionLikeRequest,
    ScreeningAttemptRequest,
    ScreeningAttemptResponse,
    ScreeningQuestion,
    UploadResponse,
)
from .storage import DocumentStorage
from .tasks import ingest_document
from .prompts import (
    FALLBACK_CONTEXT,
    FALLBACK_LABEL,
    chat_prompt,
    learning_plan_prompt,
    question_prompt,
)

settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    graph = KnowledgeGraph(settings)
    try:
        graph.ensure_schema()
    finally:
        graph.close()
    yield


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
            embedding = AIProvider(settings).embed([q])[0]
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

        # Step 1: Zero-lag fetch for selected entity question from Neo4j
        db_question = None
        source_mode = "selected_entity"
        if raw_topic:
            db_question = graph.entity_question(session_id, textbook, raw_topic)

        # Step 2: Fallback 1 / Next Question - Fetch closest related question from database
        if not db_question:
            anchor = raw_topic or graph.last_screening_entity(session_id)
            db_question = graph.related_learning_question(session_id, textbook, anchor)
            if db_question:
                source_mode = "related_entity" if anchor else "textbook_bank"

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
                citations = db_question.get("citations")
                if not isinstance(citations, list):
                    citations = []

                # Retrieve textbook passages for the entity
                passages = graph.textbook_passages(textbook, topic=entity_name, limit=2)
                if not passages and raw_topic and raw_topic != entity_name:
                    passages = graph.textbook_passages(textbook, topic=raw_topic, limit=2)

                if passages:
                    etext = passages[0]["text"]
                    summary = f"Key concept: {entity_name}. Review the passage above and answer the question below."
                else:
                    etext = f"{entity_name}: {explanation}" if explanation else f"Review key exam topic: {entity_name}."
                    summary = f"Key exam topic: {entity_name}. Review the concept facts and test your understanding below."

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
                )
                if db_question.get("question_id"):
                    graph.mark_question_asked(session_id, db_question["question_id"])

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
                    etext=etext,
                    summary=summary,
                    question=ScreeningQuestion(
                        question_id=screening_question_id,
                        question=q_text,
                        options=options,
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
        provider = AIProvider(settings)
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
        question_id = str(uuid.uuid4())
        graph.save_learning_question(
            question_id,
            session_id,
            request.topic,
            request.mode,
            question_text.strip(),
            [str(option) for option in options],
            correct_option,
            str(generated.get("explanation", "")),
            settings.question_level,
            session_context["textbook"],
            entity_name=request.topic,
        )
        return LearningPlanResponse(
            session_id=session_id,
            mode=request.mode,
            topic=topic or session_context["textbook"],
            etext=str(generated.get("etext", "")),
            summary=str(generated.get("summary", "")),
            question=ScreeningQuestion(
                question_id=question_id,
                question=question_text.strip(),
                options=[str(option) for option in options],
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
        if attempts_used >= 3 or question.get("resolved", False):
            is_correct = request.option_index == int(question["correct_option"])
            return ScreeningAttemptResponse(
                correct=is_correct,
                attempts_used=attempts_used,
                attempts_remaining=0,
                answer_revealed=True,
                correct_option_index=int(question["correct_option"]),
                feedback=str(question["explanation"]),
                progress=min(100, attempts_used * 10),
            )
        question = graph.record_screening_attempt(session_id, question_id, request.option_index)
        if not question:
            raise HTTPException(status_code=409, detail="Screening question is already resolved")
        is_correct = request.option_index == int(question["correct_option"])
        attempts_used = int(question["attempts_used"])
        revealed = is_correct or attempts_used >= 3
        feedback = "Correct. Nice work." if is_correct else "Not quite. Review the explanation and try again."
        if revealed and not is_correct:
            feedback = str(question["explanation"])
        return ScreeningAttemptResponse(
            correct=is_correct,
            attempts_used=attempts_used,
            attempts_remaining=max(0, 3 - attempts_used),
            answer_revealed=revealed,
            correct_option_index=int(question["correct_option"]) if revealed else None,
            feedback=feedback,
            progress=min(100, attempts_used * 10 + (50 if is_correct else 0)),
        )
    finally:
        graph.close()


def extract_entity_from_chat(message: str, fallback_topic: str | None = None) -> str:
    cleaned = message.strip()
    patterns = [
        r"^ask\s+from\s*[:\-]?\s*\[?([^\]\?\.\n]+)\]?",
        r"^ask\s+about\s*[:\-]?\s*\[?([^\]\?\.\n]+)\]?",
        r"^tell\s+me\s+about\s*[:\-]?\s*\[?([^\]\?\.\n]+)\]?",
        r"^summarize\s*[:\-]?\s*\[?([^\]\?\.\n]+)\]?",
        r"^what\s+is\s*[:\-]?\s*\[?([^\]\?\.\n]+)\]?",
        r"^explain\s*[:\-]?\s*\[?([^\]\?\.\n]+)\]?",
    ]
    for pat in patterns:
        m = re.match(pat, cleaned, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().strip("'\"[]")
            if candidate:
                return candidate
    return (fallback_topic or "").strip()


def synthesize_fast_two_paragraphs(
    topic: str,
    textbook: str,
    passages: list[dict],
    evidence: list[dict],
    citations: list[dict],
) -> str:
    display_topic = topic or "This subject"

    # Paragraph 1: Definition and overview from passages, citations, or evidence
    p1 = ""
    if passages:
        raw = passages[0]["text"].replace("\n", " ").strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if s.strip()]
        p1 = " ".join(sentences[:2])
        if not p1.endswith("."):
            p1 += "."
    elif citations and citations[0].get("description"):
        p1 = f"{display_topic} is an essential concept in {textbook}. {citations[0]['description']}".strip()
    elif evidence:
        p1 = f"{display_topic} is categorized as a {evidence[0]['entity_type']} in {textbook}, playing an integral role in the study curriculum."
    else:
        p1 = f"{display_topic} represents a key conceptual topic covered in {textbook} for competitive examinations."

    # Paragraph 2: Relationships, connected entities, and exam relevance
    p2 = ""
    if evidence:
        rels = [f"{e['relationship']} {e['entity_name']}" for e in evidence[:4] if e.get("entity_name")]
        if rels:
            p2 = f"In the knowledge graph, {display_topic} is linked with {', '.join(rels)}. Understanding these core linkages and structural interactions is vital for analytical exam questions."
    if not p2:
        if len(passages) > 1:
            raw2 = passages[1]["text"].replace("\n", " ").strip()
            sentences2 = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw2) if s.strip()]
            p2 = " ".join(sentences2[:2])
            if not p2.endswith("."):
                p2 += "."
        else:
            p2 = f"Detailed analysis of {display_topic} reinforces conceptual clarity and contextual recall across related chapters in {textbook}."

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

        answer = None
        provider = AIProvider(settings)
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(provider.answer, instruction, context)
                answer = future.result(timeout=min(float(settings.question_generation_timeout_seconds), 4.0))
        except (FutureTimeoutError, TimeoutError):
            logger.warning("AI provider answer timed out; using instant 2-paragraph synthesis")
        except Exception as exc:
            logger.warning("AI provider answer failed (%s); using instant 2-paragraph synthesis", exc)

        if not answer or not answer.strip():
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
                session_id, request.mode, settings.question_level, session_context["textbook"]
            )
            if bank:
                item = bank[0]
                return QuestionResponse(
                    question_id=str(uuid.uuid4()),
                    question=item["question"],
                    options=item["options"],
                    answer=item.get("answer"),
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
            AIProvider(settings).structured_answer, instruction, f"{context}\n{prior}"
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
        )
        return QuestionResponse(
            question_id=question_id,
            question=question.strip(),
            options=options,
            answer=answer,
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

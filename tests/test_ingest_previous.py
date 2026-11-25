import json
from copy import deepcopy
import logging
from unittest.mock import MagicMock, patch

import httpx
import pytest

from ingest_previous import (
    ModelResponseError,
    is_question_page,
    generate_questions,
    ocr_page,
    parse_model,
    parse_questions,
    post_with_rate_limit_retry,
    question_hash,
    resolve_pdf_path,
    run_ingestion,
)
from learning_studio.language import is_language_text


MODEL_RESPONSE = {
    "questions": [
        {
            "question_en": "1. Which is a planet?",
            "question_te": "గ్రహం ఏది?",
            "options_en": ["Mars", "Sun"],
            "options_te": ["అంగారకుడు", "సూర్యుడు"],
            "correct_option": 0,
            "explanation_en": "Mars orbits the Sun.",
            "explanation_te": "అంగారకుడు సూర్యుని చుట్టూ తిరుగుతుంది.",
            "year": 2024,
        }
    ]
}


def test_parse_model_accepts_only_supported_selection():
    assert parse_model(" grok ") == "GROK"
    with pytest.raises(ValueError, match="USE_MODEL"):
        parse_model("AUTO")


def test_question_page_detector_skips_instruction_pages():
    assert not is_question_page(
        "Read all instructions carefully.\nUse a blue or black pen.\n"
        "Candidates should answer all questions."
    )
    assert is_question_page(
        "28. Which of the following is correct?\n"
        "(1) First answer\n(2) Second answer\n"
    )


def test_ocr_page_uses_bilingual_tesseract_languages():
    page = MagicMock()
    page.get_pixmap.return_value.tobytes.return_value = b"png-data"
    result = MagicMock(stdout="Which option?\nతెలుగు".encode())

    with patch("ingest_previous.subprocess.run", return_value=result) as run:
        text = ocr_page(page, "/usr/local/bin/tesseract")

    assert "తెలుగు" in text
    assert run.call_args.args[0] == [
        "/usr/local/bin/tesseract",
        "stdin",
        "stdout",
        "-l",
        "eng+tel",
        "--psm",
        "3",
    ]


def test_rate_limit_retry_honors_retry_after():
    request = httpx.Request("POST", "https://example.test")
    limited = httpx.Response(429, headers={"Retry-After": "0"}, request=request)
    success = httpx.Response(200, json={"ok": True}, request=request)
    client = MagicMock()
    client.post.side_effect = [limited, success]

    with patch("ingest_previous.time.sleep") as sleep:
        response = post_with_rate_limit_retry(
            client,
            "https://example.test",
            {"rate_limit_max_attempts": 2, "rate_limit_max_wait_seconds": 5},
        )

    assert response.status_code == 200
    sleep.assert_called_once_with(0)
    assert client.post.call_count == 2


def test_rate_limit_retry_fails_after_configured_attempts():
    request = httpx.Request("POST", "https://example.test")
    limited = httpx.Response(429, request=request)
    client = MagicMock()
    client.post.return_value = limited

    with (
        patch("ingest_previous.time.sleep"),
        pytest.raises(httpx.HTTPStatusError),
    ):
        post_with_rate_limit_retry(
            client,
            "https://example.test",
            {"rate_limit_max_attempts": 2, "rate_limit_max_wait_seconds": 5},
        )

    assert client.post.call_count == 2


def test_resolve_pdf_path_finds_paper_in_project_pdf_directory():
    pdf_path = resolve_pdf_path("Group-I-Paper-I-2024.pdf")

    assert pdf_path.name == "Group-I-Paper-I-2024.pdf"
    assert pdf_path.parent.name == "my_pdfs"


def test_resolve_pdf_path_rejects_missing_or_non_pdf(tmp_path):
    with pytest.raises(FileNotFoundError, match="place the file in ./my_pdfs"):
        resolve_pdf_path(str(tmp_path / "missing.pdf"))
    text_file = tmp_path / "paper.txt"
    text_file.touch()
    with pytest.raises(FileNotFoundError, match="existing PDF path"):
        resolve_pdf_path(str(text_file))


def test_parse_questions_keeps_both_languages_and_builds_stable_hash():
    parsed = parse_questions(json.dumps(MODEL_RESPONSE))[0]

    assert parsed["question_en"] == "1. Which is a planet?"
    assert parsed["question_te"] == "గ్రహం ఏది?"
    assert parsed["options_te"] == ["అంగారకుడు", "సూర్యుడు"]
    assert parsed["answer_en"].startswith("Mars.")
    assert parsed["answer_te"].startswith("అంగారకుడు.")
    assert question_hash(parsed) == parsed["question_hash"]
    assert question_hash({**parsed, "question_en": "Which is a planet"}) == parsed["question_hash"]
    assert parsed["id"] == f"previous-{parsed['question_hash']}"


def test_parse_questions_uses_available_language_for_missing_translation():
    source_question = MODEL_RESPONSE["questions"][0]
    question = parse_questions(json.dumps({
        "questions": [{**source_question, "question_te": "", "options_te": []}]
    }))[0]

    assert question["question_te"] == ""
    assert question["options_te"] == []
    assert question["answer_te"] == ""


def test_parse_questions_drops_mixed_language_field_without_copying_english():
    question_data = deepcopy(MODEL_RESPONSE["questions"][0])
    question_data["question_en"] = "Which events బంక place?"
    question_data["question_te"] = "పైన పేర్కొన్న సంఘటనలు ఎప్పుడు జరిగాయి?"
    parsed = parse_questions(json.dumps({"questions": [question_data]}))

    assert len(parsed) == 1
    assert parsed[0]["question_en"] == ""
    assert parsed[0]["options_en"] == []
    assert parsed[0]["question_te"] == "పైన పేర్కొన్న సంఘటనలు ఎప్పుడు జరిగాయి?"


def test_language_field_validation_rejects_mixed_script():
    assert not is_language_text("Which of the above events బంక place", "en")
    assert not is_language_text("Which events బంక place", "te")
    assert is_language_text("పైన పేర్కొన్న సంఘటనలు ఎప్పుడు?", "te")
    assert is_language_text("Which events took place?", "en")


def test_parse_questions_rejects_bad_correct_option():
    parsed = parse_questions(json.dumps({
            "questions": [{
                "question_en": "Question?",
                "options_en": ["A", "B"],
                "correct_option": 3,
            }]
        }))
    assert parsed == []


def test_parse_questions_repairs_json_wrapped_in_model_explanation():
    response = (
        "Here is the result:\n"
        + json.dumps(MODEL_RESPONSE)
        + "\nExtraction complete."
    )

    parsed = parse_questions(response)

    assert len(parsed) == 1
    assert parsed[0]["question_te"] == "గ్రహం ఏది?"


def test_parse_questions_includes_response_shape_in_diagnostic():
    response = '{"result":"not the requested schema"}'

    with pytest.raises(ModelResponseError, match=r"keys=\['result'\].*not the requested schema"):
        parse_questions(response)


def test_grok_request_uses_prompt_json_not_incompatible_json_mode():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(
        200,
        json={
            "choices": [{
                "message": {"content": json.dumps(MODEL_RESPONSE)}
            }]
        },
        request=request,
    )
    client = MagicMock()
    client.post.return_value = response
    settings = {
        "use_model": "GROK",
        "grok_endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "grok_api_key": "test-key",
        "grok_model": "openai/gpt-oss-20b",
        "max_output_tokens": 4096,
        "timeout": 30,
    }

    questions = generate_questions(
        "Which planet?\n(1) Mars\n(2) Venus", settings, client
    )

    request_json = client.post.call_args.kwargs["json"]
    assert questions[0]["question_te"] == "గ్రహం ఏది?"
    assert "response_format" not in request_json
    assert "Return exactly one complete JSON object" in request_json["messages"][0]["content"]


def test_run_ingestion_shows_progress_and_logs_extracted_page_text(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="ingest_previous")
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.touch()
    pages = [MagicMock(), MagicMock(), MagicMock()]
    question_page = "Which planet?\n(1) Mars\n(2) Venus\nతెలుగు"
    pages[0].extract_text.return_value = question_page
    pages[1].extract_text.return_value = question_page
    pages[2].extract_text.return_value = "Candidate instructions తెలుగు"
    progress = []
    document = MagicMock()

    with (
        patch("ingest_previous.PdfReader") as reader,
        patch("ingest_previous.pymupdf.open", return_value=document),
        patch("ingest_previous.generate_questions", return_value=parse_questions(
            json.dumps(MODEL_RESPONSE)
        )),
        patch("ingest_previous.store_question", side_effect=[True, False]) as store,
    ):
        reader.return_value.pages = pages
        added, skipped = run_ingestion(
            pdf_path,
            MagicMock(),
            {"use_model": "GROK"},
            MagicMock(),
            progress=lambda *values: progress.append(values),
        )

    assert (added, skipped) == (1, 1)
    assert store.call_count == 1
    assert [entry[0] for entry in progress] == [1, 2, 3]
    assert progress[-1][2:5] == (2, 2, 1)
    assert "PDF_PAGE_TEXT page=1 method=native" in caplog.text
    assert question_page in caplog.text
    document.close.assert_called_once()


def test_run_ingestion_continues_after_malformed_model_response(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="ingest_previous")
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.touch()
    pages = [MagicMock(), MagicMock()]
    for page in pages:
        page.extract_text.return_value = "Which planet?\n(1) Mars\n(2) Venus\nతెలుగు"
    document = MagicMock()
    valid_question = parse_questions(json.dumps(MODEL_RESPONSE))
    progress = []

    with (
        patch("ingest_previous.PdfReader") as reader,
        patch("ingest_previous.pymupdf.open", return_value=document),
        patch(
            "ingest_previous.generate_questions",
            side_effect=[ModelResponseError("bad response shape"), valid_question],
        ),
        patch("ingest_previous.store_question", return_value=True),
    ):
        reader.return_value.pages = pages
        added, skipped = run_ingestion(
            pdf_path,
            MagicMock(),
            {"use_model": "GROK"},
            MagicMock(),
            progress=lambda *values: progress.append(values),
        )

    assert (added, skipped) == (1, 0)
    assert len(progress) == 2
    assert "QUESTION_PARSE_FAILED page=1" in caplog.text


def test_run_ingestion_reports_no_extraction_as_not_duplicates(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.touch()
    page = MagicMock()
    page.extract_text.return_value = "Garbled PDF text తెలుగు"
    document = MagicMock()

    with (
        patch("ingest_previous.PdfReader") as reader,
        patch("ingest_previous.pymupdf.open", return_value=document),
        patch("ingest_previous.generate_questions", return_value=[]),
        pytest.raises(ValueError, match=r"questions found: 0, duplicates: 0"),
    ):
        reader.return_value.pages = [page]
        run_ingestion(
            pdf_path,
            MagicMock(),
            {"use_model": "GROK"},
            MagicMock(),
            progress=lambda *values: None,
        )
    document.close.assert_called_once()


def test_run_ingestion_ocr_fallback_reads_question_bilingual_text(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.touch()
    page = MagicMock()
    page.extract_text.return_value = "unmapped glyph data"
    document = MagicMock()
    parsed_question = parse_questions(json.dumps(MODEL_RESPONSE))

    with (
        patch("ingest_previous.PdfReader") as reader,
        patch("ingest_previous.pymupdf.open", return_value=document),
        patch("ingest_previous.tesseract_command", return_value="tesseract") as find_ocr,
        patch(
            "ingest_previous.ocr_page",
            return_value="1. Which planet?\n(1) Mars\n(2) Venus\nతెలుగు",
        ) as ocr,
        patch("ingest_previous.generate_questions", return_value=parsed_question),
        patch("ingest_previous.store_question", return_value=True) as store,
    ):
        reader.return_value.pages = [page]
        added, skipped = run_ingestion(
            pdf_path,
            MagicMock(),
            {"use_model": "GROK"},
            MagicMock(),
            progress=lambda *values: None,
        )

    assert (added, skipped) == (1, 0)
    find_ocr.assert_called_once()
    ocr.assert_called_once_with(document.__getitem__.return_value, "tesseract")
    store.assert_called_once()
    document.close.assert_called_once()


def test_store_query_upgrades_matching_previous_question():
    from ingest_previous import _store_question

    tx = MagicMock()
    tx.run.return_value.single.return_value = {"question_id": "previous-1"}
    question = parse_questions(json.dumps(MODEL_RESPONSE))[0]

    created = _store_question(tx, question, "paper.pdf", 4)

    query = tx.run.call_args.args[0]
    assert created is False
    assert "existing.question_hash = $question_hash" in query
    assert "existing.question_te" in query
    assert "SET existing += $properties" in query
    assert tx.run.call_count == 1
    assert tx.run.call_args.kwargs["properties"]["question_te"] == "గ్రహం ఏది?"


def test_store_query_creates_previous_question_when_no_match():
    from ingest_previous import _store_question

    tx = MagicMock()
    tx.run.return_value.single.return_value = None
    question = parse_questions(json.dumps(MODEL_RESPONSE))[0]

    created = _store_question(tx, question, "paper.pdf", 4)

    assert created is True
    assert "CREATE (q:PreviousQuestion)" in tx.run.call_args.args[0]


def test_question_bank_selects_language_specific_previous_question_fields():
    from learning_studio.graph import KnowledgeGraph

    graph = KnowledgeGraph.__new__(KnowledgeGraph)
    session = MagicMock()
    session.__enter__.return_value = session
    graph.session = MagicMock(return_value=session)
    record = MagicMock()
    record.data.return_value = {
        "question_id": "previous-1",
        "question": "తెలుగు ప్రశ్న",
        "question_en": "Is it Telugu?",
        "question_te": "ఇది తెలుగు ప్రశ్ననా?",
        "options_en": ["Yes", "No"],
        "options_te": ["అవును", "కాదు"],
        "answer_en": "Yes",
        "answer_te": "అవును",
        "options": ["అవును", "కాదు"],
        "answer": "అవును",
        "citations": "[]",
    }
    session.run.return_value = [record]

    questions = graph.question_bank(
        "session-1",
        "previous",
        "screening",
        "courseware.pdf",
        language="te",
    )

    query = session.run.call_args.args[0]
    assert "q.question_te" in query
    assert "q.options_te" in query
    assert session.run.call_args.kwargs["language"] == "te"
    assert questions[0]["question"] == "ఇది తెలుగు ప్రశ్ననా?"
    assert questions[0]["options"] == ["అవును", "కాదు"]


def test_question_bank_hides_mixed_script_legacy_question():
    from learning_studio.graph import KnowledgeGraph

    graph = KnowledgeGraph.__new__(KnowledgeGraph)
    session = MagicMock()
    session.__enter__.return_value = session
    graph.session = MagicMock(return_value=session)
    records = []
    for question in (
        {
            "question_id": "mixed",
            "question_en": "Which of the above main events బంక place",
            "question_te": "ఏ సంఘటనలు జరిగాయి?",
            "options_en": ["First", "Second"],
            "options_te": ["మొదటిది", "రెండవది"],
            "answer_en": "First",
            "answer_te": "మొదటిది",
            "citations": "[]",
        },
        {
            "question_id": "clean",
            "question_en": "Which events took place?",
            "question_te": "ఏ సంఘటనలు జరిగాయి?",
            "options_en": ["First", "Second"],
            "options_te": ["మొదటిది", "రెండవది"],
            "answer_en": "First",
            "answer_te": "మొదటిది",
            "citations": "[]",
        },
    ):
        record = MagicMock()
        record.data.return_value = question
        records.append(record)
    session.run.return_value = records

    questions = graph.question_bank(
        "session-1",
        "previous",
        "screening",
        "paper.pdf",
        language="en",
    )

    assert [question["question_id"] for question in questions] == ["clean"]


def test_previous_question_can_be_liked():
    from learning_studio.graph import KnowledgeGraph

    graph = KnowledgeGraph.__new__(KnowledgeGraph)
    session = MagicMock()
    session.__enter__.return_value = session
    graph.session = MagicMock(return_value=session)
    session.run.return_value.single.return_value = {"liked": True}

    assert graph.set_question_liked("session-1", "previous-1", True)
    assert "OR q:PreviousQuestion" in session.run.call_args.args[0]

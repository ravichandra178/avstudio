from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore, Event, Lock
from time import sleep
from unittest.mock import MagicMock

import httpx
import pytest

import ingest_question
from ingest_question import (
    IngestModel,
    existing_question_ids,
    generate_question,
    generate_questions_for_entity,
    grok_error_detail,
    load_entities,
    normalized_question,
    parse_ingest_model,
    parse_json,
    parse_toc_chapters,
    question_id_for_entity,
    question_prompt,
    store_entity_questions,
    store_questions,
)

ENTITY = {
    "textbook": "Science.pdf",
    "entity_type": "Person",
    "entity_name": "Har Gobind Khorana",
    "relationships": [{"relationship": "MENTIONS", "name": "Har Gobind Khorana"}],
}
QUESTION_JSON = (
    '{"question":"Which field did Khorana study?",'
    '"options":["Biochemistry","Physics","Astronomy","Geology"],'
    '"correct_option":0,"explanation":"He was a biochemist.",'
    '"question_te":"ఖోరానా ఏ రంగాన్ని అధ్యయనం చేశారు?",'
    '"options_te":["జీవరసాయన శాస్త్రం","భౌతిక శాస్త్రం","ఖగోళ శాస్త్రం","భూగర్భ శాస్త్రం"],'
    '"explanation_te":"ఆయన జీవరసాయన శాస్త్రవేత్త."}'
)


def settings(**overrides):
    values = {
        "group": "Group 1",
        "level": "screening exam",
        "gemini_api_key": "test-key",
        "gemini_endpoint": "https://generativelanguage.googleapis.com/v1beta/models",
        "gemini_model": "models/gemini-3.8-flash",
        "gemini_timeout": 8,
        "gemini_max_output_tokens": 256,
        "grok_api_key": "",
        "grok_endpoint": "https://api.x.ai/v1/chat/completions",
        "grok_model": "grok-4.7",
        "grok_timeout": 15,
        "grok_max_output_tokens": 256,
        "ollama_endpoint": "http://localhost:11434",
        "ollama_model": "qwen2.5:14b",
        "keep_alive": "10m",
        "timeout": 300,
        "context_tokens": 8192,
        "max_output_tokens": 512,
    }
    return values | overrides


def response(payload):
    result = MagicMock()
    result.json.return_value = payload
    result.raise_for_status.return_value = None
    return result


def test_generate_question_uses_gemini_first(caplog):
    caplog.set_level("INFO")
    client = MagicMock()
    client.post.return_value = response(
        {"candidates": [{"content": {"parts": [{"text": QUESTION_JSON}]}}]}
    )

    question = generate_question(settings(), ENTITY, client)

    assert question["provider"] == "gemini"
    assert question["options"][0] == "Biochemistry"
    call = client.post.call_args
    assert call.args[0].endswith("/gemini-3.8-flash:generateContent")
    assert call.kwargs["headers"] == {"x-goog-api-key": "test-key"}
    assert call.kwargs["json"]["generationConfig"]["responseMimeType"] == "application/json"
    assert call.kwargs["json"]["generationConfig"]["maxOutputTokens"] == 256
    assert call.kwargs["timeout"] == 8
    assert "QUESTION_INGEST_PROVIDER_START" in caplog.text
    assert "provider=gemini" in caplog.text
    assert "QUESTION_INGEST_PROVIDER_SUCCESS" in caplog.text
    assert "test-key" not in caplog.text


def test_gemini_retries_transient_rate_limit_after_provider_delay(monkeypatch):
    sleeps = []
    rate_limited = httpx.Response(
        429,
        json={"error": {"message": "Rate limit exceeded"}},
        headers={"Retry-After": "0.5"},
        request=httpx.Request(
            "POST",
            "https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent",
        ),
    )
    client = MagicMock()
    client.post.side_effect = [
        rate_limited,
        response({"candidates": [{"content": {"parts": [{"text": QUESTION_JSON}]}}]}),
    ]
    monkeypatch.setattr(ingest_question, "sleep", sleeps.append)

    question = generate_question(settings(), ENTITY, client)

    assert question["provider"] == "gemini"
    assert client.post.call_count == 2
    assert sleeps == [0.5]


def test_generate_question_does_not_fall_back_to_ollama_when_gemini_fails(caplog):
    caplog.set_level("INFO")
    client = MagicMock()
    client.post.side_effect = httpx.ConnectError("Gemini unavailable")

    with pytest.raises(RuntimeError, match="Ollama fallback is disabled"):
        generate_question(settings(), ENTITY, client)

    assert client.post.call_count == 1
    assert client.post.call_args_list[0].args[0].endswith(
        "/gemini-3.8-flash:generateContent"
    )
    assert "QUESTION_INGEST_GEMINI_FAILED" in caplog.text
    assert "QUESTION_INGEST_GROK_SKIPPED" in caplog.text
    assert "http://localhost:11434/api/chat" not in str(client.post.call_args_list)


def test_generate_question_uses_grok_after_gemini_failure():
    client = MagicMock()
    client.post.side_effect = [
        httpx.ConnectError("Gemini unavailable"),
        response({"choices": [{"message": {"content": QUESTION_JSON}}]}),
    ]

    question = generate_question(
        settings(grok_api_key="test-grok-key", grok_model="grok-3-mini"),
        ENTITY,
        client,
    )

    assert question["provider"] == "groq"
    assert client.post.call_count == 2
    grok_call = client.post.call_args_list[1]
    assert grok_call.args[0] == "https://api.x.ai/v1/chat/completions"
    assert grok_call.kwargs["headers"] == {
        "Authorization": "Bearer test-grok-key"
    }
    assert grok_call.kwargs["json"]["model"] == "grok-3-mini"
    assert grok_call.kwargs["json"]["response_format"] == {"type": "json_object"}
    assert grok_call.kwargs["timeout"] == 15


def test_use_model_grok_selects_grok_as_primary_provider():
    client = MagicMock()
    client.post.return_value = response(
        {"choices": [{"message": {"content": QUESTION_JSON}}]}
    )
    ingest_settings = settings(
        use_model=IngestModel.GROK,
        grok_api_key="test-grok-key",
    )

    question = generate_question(ingest_settings, ENTITY, client)

    assert question["provider"] == "groq"
    client.post.assert_called_once()
    assert client.post.call_args.args[0] == "https://api.x.ai/v1/chat/completions"


def test_use_model_ollama_skips_configured_cloud_providers(caplog):
    caplog.set_level("INFO")
    client = MagicMock()
    client.post.return_value = response({"message": {"content": QUESTION_JSON}})
    ingest_settings = settings(
        use_model=IngestModel.OLLAMA,
        grok_api_key="test-grok-key",
        ollama_model="qwen2.5-1.5b-instruct:local",
    )

    question = generate_question(ingest_settings, ENTITY, client)

    assert question["provider"] == "ollama"
    assert client.post.call_count == 1
    assert client.post.call_args.args[0] == "http://localhost:11434/api/chat"
    assert (
        client.post.call_args.kwargs["json"]["model"]
        == "hf.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF:latest"
    )
    assert client.post.call_args.kwargs["json"]["format"]["properties"][
        "correct_option"
    ] == {"type": "integer", "minimum": 0, "maximum": 3}
    assert client.post.call_args.kwargs["json"]["format"]["properties"]["options"][
        "maxItems"
    ] == 4
    assert client.post.call_args.kwargs["json"]["format"]["properties"][
        "options_te"
    ]["maxItems"] == 4
    assert "reason=USE_MODEL_OLLAMA" in caplog.text


def test_parse_ingest_model_is_case_insensitive_and_rejects_unknown_values():
    assert parse_ingest_model("grok") is IngestModel.GROK
    assert parse_ingest_model("OLLAMA") is IngestModel.OLLAMA
    try:
        parse_ingest_model("unknown")
    except ValueError as exc:
        assert "GEMINI, GROK, OLLAMA" in str(exc)
    else:
        raise AssertionError("Invalid USE_MODEL value should be rejected")


def test_parse_json_keeps_telugu_question_fields():
    question = parse_json(QUESTION_JSON)

    assert question["question_te"] == "ఖోరానా ఏ రంగాన్ని అధ్యయనం చేశారు?"
    assert question["options_te"][0] == "జీవరసాయన శాస్త్రం"
    assert question["explanation_te"] == "ఆయన జీవరసాయన శాస్త్రవేత్త."


def test_parse_toc_chapters_handles_split_titles_and_page_numbers():
    contents = """
    Contents
    01
    Communicating Swachhata, Changing Behaviour
    07
    02 From Beacons to Landmarks: India’s Changing Lighthouse Story
    23
    03 Five Years of PM GatiShakti: Transforming
    India’s Infrastructure Planning
    405
    """

    assert parse_toc_chapters(contents) == [
        ("Communicating Swachhata, Changing Behaviour", 7),
        ("From Beacons to Landmarks: India’s Changing Lighthouse Story", 23),
        (
            "Five Years of PM GatiShakti: Transforming India’s Infrastructure Planning",
            405,
        ),
    ]


def test_load_entities_adds_toc_chapters_with_section_text():
    def record(data):
        result = MagicMock()
        result.data.return_value = data
        return result

    toc = (
        "Contents\n01 Chapter One\n493\n02 Chapter Two\n533\n"
    )
    existing_entities = [
        record(
            {
                "entity_name": "Existing Chapter",
                "entity_type": "Topic",
                "document_properties": {"source": "Science.pdf"},
                "relationships": [],
            }
        )
    ]
    documents = [
        record(
            {
                "document_properties": {
                    "source": "Science.pdf",
                    "page": "2-4",
                    "text": toc,
                }
            }
        ),
        record(
            {
                "document_properties": {
                    "source": "Science.pdf",
                    "page": "248-250",
                    "text": "Chapter One source facts.",
                }
            }
        ),
        record(
            {
                "document_properties": {
                    "source": "Science.pdf",
                    "page": "268-270",
                    "text": "Chapter Two source facts.",
                }
            }
        ),
    ]
    session = MagicMock()
    session.__enter__.return_value = session
    session.run.side_effect = [existing_entities, documents]
    driver = MagicMock()
    driver.session.return_value = session

    entities = load_entities(driver, "neo4j", "Science.pdf")

    by_name = {entity["entity_name"]: entity for entity in entities}
    assert set(by_name) == {"Existing Chapter", "Chapter One", "Chapter Two"}
    assert by_name["Chapter One"]["chapter_page"] == 493
    assert "Chapter One source facts." in by_name["Chapter One"]["source_text"]
    assert by_name["Chapter Two"]["chapter_page"] == 533
    assert "Chapter Two source facts." in by_name["Chapter Two"]["source_text"]


def test_question_prompt_includes_chapter_source_passage():
    prompt = question_prompt(
        settings(),
        ENTITY
        | {
            "chapter_page": 167,
            "source_text": "[Pages 167-168]\nThe chapter passage.",
        },
    )

    assert "Textbook source passage (pages 167 onward)" in prompt
    assert "The chapter passage." in prompt


def test_grok_retries_tpm_rate_limit_after_provider_delay(monkeypatch):
    sleeps = []
    monkeypatch.setattr(ingest_question, "_grok_quota_exhausted", ingest_question.Event())
    monkeypatch.setattr(ingest_question, "_grok_last_request", 0.0)
    monkeypatch.setattr(ingest_question, "sleep", sleeps.append)
    rate_limited = httpx.Response(
        429,
        json={
            "error": {
                "message": "Tokens per minute exceeded. Please try again in 9.5s."
            }
        },
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    client = MagicMock()
    client.post.side_effect = [
        rate_limited,
        response({"choices": [{"message": {"content": QUESTION_JSON}}]}),
    ]

    question = ingest_question._generate_grok_question(
        settings(
            grok_api_key="test-grok-key",
            grok_request_interval=0,
        ),
        ENTITY,
        client,
    )

    assert question["question_te"] == "ఖోరానా ఏ రంగాన్ని అధ్యయనం చేశారు?"
    assert client.post.call_count == 2
    assert sleeps == [9.5]


def test_grok_does_not_retry_daily_token_quota(monkeypatch):
    sleeps = []
    monkeypatch.setattr(ingest_question, "_grok_quota_exhausted", ingest_question.Event())
    monkeypatch.setattr(ingest_question, "_grok_last_request", 0.0)
    monkeypatch.setattr(ingest_question, "sleep", sleeps.append)
    rate_limited = httpx.Response(
        429,
        json={"error": {"message": "Tokens per day limit reached"}},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    client = MagicMock()
    client.post.return_value = rate_limited

    with pytest.raises(httpx.HTTPStatusError):
        ingest_question._generate_grok_question(
            settings(
                grok_api_key="test-grok-key",
                grok_request_interval=0,
            ),
            ENTITY,
            client,
        )

    assert client.post.call_count == 1
    assert sleeps == []


def test_grok_retries_invalid_telugu_output(monkeypatch):
    monkeypatch.setattr(ingest_question, "_grok_quota_exhausted", ingest_question.Event())
    monkeypatch.setattr(ingest_question, "_grok_last_request", 0.0)
    malformed = QUESTION_JSON.replace(
        "ఖోరానా ఏ రంగాన్ని అధ్యయనం చేశారు?",
        "What field did Khorana study?",
    )
    client = MagicMock()
    client.post.side_effect = [
        response({"choices": [{"message": {"content": malformed}}]}),
        response({"choices": [{"message": {"content": QUESTION_JSON}}]}),
    ]

    question = ingest_question._generate_grok_question(
        settings(
            grok_api_key="test-grok-key",
            grok_request_interval=0,
        ),
        ENTITY,
        client,
    )

    assert question["question_te"] == "ఖోరానా ఏ రంగాన్ని అధ్యయనం చేశారు?"
    assert client.post.call_count == 2
    retry_prompt = client.post.call_args_list[1].kwargs["json"]["messages"][1]["content"]
    assert "Correction for this retry" in retry_prompt


def test_daily_quota_stops_grok_calls_for_remaining_entities(monkeypatch):
    monkeypatch.setattr(ingest_question, "_grok_quota_exhausted", ingest_question.Event())
    monkeypatch.setattr(ingest_question, "_grok_last_request", 0.0)
    quota_response = httpx.Response(
        429,
        json={"error": {"message": "Tokens per day limit reached"}},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    client = MagicMock()
    client.post.return_value = quota_response
    grok_settings = settings(
        use_model=IngestModel.GROK,
        grok_api_key="test-grok-key",
        grok_request_interval=0,
    )

    _, first_failures = generate_questions_for_entity(
        grok_settings, ENTITY, [], 3, client
    )
    _, later_failures = generate_questions_for_entity(
        grok_settings, ENTITY | {"entity_name": "A second entity"}, [], 3, client
    )

    assert first_failures == 3
    assert later_failures == 3
    assert client.post.call_count == 1


def test_generate_questions_for_entity_generates_distinct_questions(monkeypatch):
    generated = iter(
        [
            {"question": "Which field did Khorana study?", "provider": "groq"},
            {"question": "Which field did Khorana study?", "provider": "groq"},
            {"question": "What scientific contribution is attributed to Khorana?", "provider": "groq"},
        ]
    )
    prompts = []

    def fake_generate_question(_settings, entity, _client):
        prompts.append(entity | {"previous_questions": list(entity["previous_questions"])})
        return next(generated)

    monkeypatch.setattr(ingest_question, "generate_question", fake_generate_question)

    questions, failures = generate_questions_for_entity(
        settings(),
        ENTITY,
        ["Which field did Khorana study?"],
        1,
        MagicMock(),
    )

    assert failures == 0
    assert len(questions) == 1
    assert questions[0]["question"].startswith("What scientific contribution")
    assert questions[0]["question_id"].startswith("textbook-")
    assert prompts[0]["question_number"] == 2
    assert prompts[1]["previous_questions"] == ["Which field did Khorana study?"]
    assert prompts[2]["variation_attempt"] == 3
    assert normalized_question("Same question?") == normalized_question("Same   QUESTION!")


def test_parse_json_rejects_telugu_in_english_question_fields():
    malformed = (
        QUESTION_JSON
        .replace("Which field did Khorana study?", "ఖోరానా ఏ రంగాన్ని అధ్యయనం చేశారు?")
        .replace('"Biochemistry","Physics","Astronomy","Geology"',
                 '"జీవరసాయన శాస్త్రం","భౌతిక శాస్త్రం","ఖగోళ శాస్త్రం","భూగర్భ శాస్త్రం"')
    )

    try:
        parse_json(malformed)
    except ValueError as exc:
        assert "invalid four-option question" in str(exc)
    else:
        raise AssertionError("Invalid English-language fields should be rejected")


def test_grok_failure_does_not_fall_back_to_ollama():
    client = MagicMock()
    client.post.side_effect = httpx.ConnectError("Grok unavailable")

    with pytest.raises(httpx.ConnectError, match="Grok unavailable"):
        generate_question(
            settings(
                use_model=IngestModel.GROK,
                grok_api_key="test-grok-key",
                grok_model="grok-3-mini",
            ),
            ENTITY,
            client,
        )

    assert client.post.call_count == 1
    assert client.post.call_args.args[0] == "https://api.x.ai/v1/chat/completions"


def test_grok_http_error_logs_status_body_and_redacts_api_key_without_ollama(
    caplog,
):
    caplog.set_level("WARNING")
    grok_key = "private-grok-key"
    grok_response = httpx.Response(
        503,
        json={"error": {"message": f"temporarily unavailable {grok_key}"}},
        request=httpx.Request("POST", "https://api.x.ai/v1/chat/completions"),
    )
    client = MagicMock()
    client.post.return_value = grok_response

    with pytest.raises(httpx.HTTPStatusError):
        generate_question(
            settings(
                use_model=IngestModel.GROK,
                grok_api_key=grok_key,
            ),
            ENTITY,
            client,
        )

    assert "http_status=503" in caplog.text
    assert "temporarily unavailable" in caplog.text
    assert "[REDACTED]" in caplog.text
    assert grok_key not in caplog.text
    assert client.post.call_count == 1


def test_grok_error_detail_truncates_long_response():
    grok_key = "private-grok-key"
    grok_response = httpx.Response(
        400,
        text=f"detail {grok_key} " + ("x" * 1500),
        request=httpx.Request("POST", "https://api.x.ai/v1/chat/completions"),
    )
    error = httpx.HTTPStatusError(
        "Bad request",
        request=grok_response.request,
        response=grok_response,
    )

    detail = grok_error_detail(error, grok_key)

    assert "http_status=400" in detail
    assert "[REDACTED]" in detail
    assert grok_key not in detail
    assert len(detail) < 1100


def test_gemini_failure_opens_run_circuit_for_later_entities():
    client = MagicMock()
    client.post.side_effect = [
        httpx.ReadTimeout("Gemini read timed out"),
        response({"choices": [{"message": {"content": QUESTION_JSON}}]}),
        response({"choices": [{"message": {"content": QUESTION_JSON}}]}),
    ]
    gemini_unavailable = Event()
    gemini_settings = settings(grok_api_key="test-grok-key")

    first = generate_question(
        gemini_settings, ENTITY, client, gemini_unavailable
    )
    second = generate_question(
        gemini_settings,
        ENTITY | {"entity_name": "A second entity"},
        client,
        gemini_unavailable,
    )

    assert first["provider"] == "groq"
    assert second["provider"] == "groq"
    assert gemini_unavailable.is_set()
    assert client.post.call_count == 3
    assert client.post.call_args_list[0].args[0].endswith(
        "/gemini-3.8-flash:generateContent"
    )
    assert all(
        call.args[0] == "https://api.x.ai/v1/chat/completions"
        for call in client.post.call_args_list[1:]
    )


def test_generate_question_uses_ollama_when_gemini_key_is_missing():
    client = MagicMock()
    client.post.return_value = response({"message": {"content": QUESTION_JSON}})

    question = generate_question(
        settings(use_model=IngestModel.OLLAMA, gemini_api_key=""),
        ENTITY,
        client,
    )

    assert question["provider"] == "ollama"
    client.post.assert_called_once()
    assert client.post.call_args.args[0] == "http://localhost:11434/api/chat"


def test_ollama_fallback_concurrency_is_limited_and_queue_is_logged(caplog):
    caplog.set_level("INFO")
    client = MagicMock()
    state_lock = Lock()
    active_requests = 0
    max_active_requests = 0

    def post(*args, **kwargs):
        nonlocal active_requests, max_active_requests
        with state_lock:
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
        sleep(0.03)
        with state_lock:
            active_requests -= 1
        return response({"message": {"content": QUESTION_JSON}})

    client.post.side_effect = post
    semaphore = BoundedSemaphore(1)
    ollama_settings = settings(
        use_model=IngestModel.OLLAMA,
        gemini_api_key="",
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda entity: generate_question(
                    ollama_settings,
                    entity,
                    client,
                    ollama_semaphore=semaphore,
                ),
                [ENTITY, ENTITY | {"entity_name": "A second entity"}],
            )
        )

    assert [item["provider"] for item in results] == ["ollama", "ollama"]
    assert max_active_requests == 1
    assert "QUESTION_INGEST_OLLAMA_QUEUE" in caplog.text
    assert "queue_wait_seconds=" in caplog.text
    assert "request_duration_seconds=" in caplog.text


def test_existing_question_ids_checks_all_ids_in_one_query():
    existing_id = "textbook-0123456789abcdef01234567"
    driver = MagicMock()
    session = MagicMock()
    session.__enter__.return_value = session
    driver.session.return_value = session
    session.execute_read.side_effect = lambda callback: callback(MagicMock(
        run=MagicMock(return_value=[{"id": existing_id}])
    ))

    result = existing_question_ids(
        driver,
        "neo4j",
        [existing_id, "textbook-abcdefabcdefabcdefabcdef"],
    )

    assert result == {existing_id}
    driver.session.assert_called_once_with(database="neo4j")
    assert session.execute_read.call_count == 1


def test_store_questions_writes_batch_with_provider_source():
    driver = MagicMock()
    session = MagicMock()
    session.__enter__.return_value = session
    driver.session.return_value = session
    entity = ENTITY | {
        "entity_name": "Baudhayan",
    }
    question = {
        "question": "Which mathematician is mentioned?",
        "options": ["Baudhayan", "Aryabhata", "Brahmagupta", "Bhaskara"],
        "correct_option": 0,
        "explanation": "Textbook mention.",
        "question_te": "ఏ గణిత శాస్త్రవేత్త ప్రస్తావించబడ్డారు?",
        "options_te": ["బౌధాయనుడు", "ఆర్యభటుడు", "బ్రహ్మగుప్తుడు", "భాస్కరుడు"],
        "explanation_te": "పాఠ్యపుస్తకంలో ప్రస్తావించబడింది.",
        "provider": "gemini",
    }

    ids = store_questions(driver, "neo4j", [(entity, question)])

    assert ids == [question_id_for_entity(entity)]
    stored_query = session.execute_write.call_args.args[0]
    tx = MagicMock()
    stored_query(tx)
    assert "UNWIND $records AS item" in tx.run.call_args.args[0]
    params = tx.run.call_args.kwargs
    assert len(params["records"]) == 1
    assert params["records"][0]["ingestion_source"] == "gemini_textbook"
    assert params["records"][0]["entity_name"] == "Baudhayan"
    assert params["records"][0]["question_te"] == "ఏ గణిత శాస్త్రవేత్త ప్రస్తావించబడ్డారు?"
    assert params["records"][0]["options_te"][0] == "బౌధాయనుడు"


def test_store_entity_questions_persists_results_for_one_completed_entity(monkeypatch):
    question = {"question": "Which mathematician is mentioned?", "provider": "groq"}
    store_batch = MagicMock(return_value=["question-id"])
    monkeypatch.setattr(ingest_question, "store_questions", store_batch)
    driver = MagicMock()

    stored, failed = store_entity_questions(
        driver, "neo4j", ENTITY, [question]
    )

    assert (stored, failed) == (1, 0)
    store_batch.assert_called_once_with(
        driver,
        "neo4j",
        [(ENTITY, question)],
    )

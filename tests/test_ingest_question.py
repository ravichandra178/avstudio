from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore, Event, Lock
from time import sleep
from unittest.mock import MagicMock

import httpx

from ingest_question import (
    IngestModel,
    existing_question_ids,
    generate_question,
    grok_error_detail,
    parse_ingest_model,
    question_id_for_entity,
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
    '"correct_option":0,"explanation":"He was a biochemist."}'
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


def test_generate_question_falls_back_to_ollama_when_gemini_fails(caplog):
    caplog.set_level("INFO")
    client = MagicMock()
    client.post.side_effect = [
        httpx.ConnectError("Gemini unavailable"),
        response({"message": {"content": QUESTION_JSON}}),
    ]

    question = generate_question(settings(), ENTITY, client)

    assert question["provider"] == "ollama"
    assert client.post.call_count == 2
    assert client.post.call_args_list[0].args[0].endswith(
        "/gemini-3.8-flash:generateContent"
    )
    assert client.post.call_args_list[1].args[0] == "http://localhost:11434/api/chat"
    assert "QUESTION_INGEST_GEMINI_FAILED" in caplog.text
    assert "falling back to Grok" in caplog.text
    assert "QUESTION_INGEST_GROK_SKIPPED" in caplog.text
    assert "QUESTION_INGEST_PROVIDER_SUCCESS" in caplog.text


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

    assert question["provider"] == "grok"
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

    assert question["provider"] == "grok"
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
        == "qwen2.5-1.5b-instruct:local"
    )
    assert client.post.call_args.kwargs["json"]["format"]["properties"][
        "correct_option"
    ] == {"type": "integer", "minimum": 0, "maximum": 3}
    assert client.post.call_args.kwargs["json"]["format"]["properties"]["options"][
        "maxItems"
    ] == 4
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


def test_grok_failure_falls_back_to_ollama():
    client = MagicMock()
    client.post.side_effect = [
        httpx.ConnectError("Gemini unavailable"),
        httpx.ConnectError("Grok unavailable"),
        response({"message": {"content": QUESTION_JSON}}),
    ]

    question = generate_question(
        settings(grok_api_key="test-grok-key", grok_model="grok-3-mini"),
        ENTITY,
        client,
    )

    assert question["provider"] == "ollama"
    assert client.post.call_count == 3
    assert client.post.call_args_list[2].args[0] == "http://localhost:11434/api/chat"


def test_grok_http_error_logs_status_body_and_redacts_api_key(caplog):
    caplog.set_level("WARNING")
    grok_key = "private-grok-key"
    grok_response = httpx.Response(
        503,
        json={"error": {"message": f"temporarily unavailable {grok_key}"}},
        request=httpx.Request("POST", "https://api.x.ai/v1/chat/completions"),
    )
    client = MagicMock()
    client.post.side_effect = [
        httpx.ConnectError("Gemini unavailable"),
        grok_response,
        response({"message": {"content": QUESTION_JSON}}),
    ]

    question = generate_question(
        settings(grok_api_key=grok_key),
        ENTITY,
        client,
    )

    assert question["provider"] == "ollama"
    assert "QUESTION_INGEST_GROK_FAILED" in caplog.text
    assert "http_status=503" in caplog.text
    assert "temporarily unavailable" in caplog.text
    assert "[REDACTED]" in caplog.text
    assert grok_key not in caplog.text


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
        response({"message": {"content": QUESTION_JSON}}),
        response({"message": {"content": QUESTION_JSON}}),
    ]
    gemini_unavailable = Event()

    first = generate_question(settings(), ENTITY, client, gemini_unavailable)
    second = generate_question(
        settings(),
        ENTITY | {"entity_name": "A second entity"},
        client,
        gemini_unavailable,
    )

    assert first["provider"] == "ollama"
    assert second["provider"] == "ollama"
    assert gemini_unavailable.is_set()
    assert client.post.call_count == 3
    assert client.post.call_args_list[0].args[0].endswith(
        "/gemini-3.8-flash:generateContent"
    )
    assert all(
        call.args[0] == "http://localhost:11434/api/chat"
        for call in client.post.call_args_list[1:]
    )


def test_generate_question_uses_ollama_when_gemini_key_is_missing():
    client = MagicMock()
    client.post.return_value = response({"message": {"content": QUESTION_JSON}})

    question = generate_question(
        settings(gemini_api_key=""),
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
    ollama_settings = settings(gemini_api_key="")

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

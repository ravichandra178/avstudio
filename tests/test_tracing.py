from contextlib import contextmanager
from types import SimpleNamespace

from learning_studio import tracing


def test_langfuse_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("USE_LANGFUSE", raising=False)
    monkeypatch.delenv("LANGFUSE_TRACING_ENABLED", raising=False)
    assert not tracing._tracing_enabled()

    monkeypatch.setenv("USE_LANGFUSE", "true")
    assert tracing._tracing_enabled()

    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    assert not tracing._tracing_enabled()


class FakeObservation:
    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


class FakeLangfuse:
    def __init__(self):
        self.observations = []

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        observation = FakeObservation()
        self.observations.append((kwargs, observation))
        yield observation


def test_tracing_excludes_content_by_default_and_records_usage(monkeypatch):
    langfuse = FakeLangfuse()
    propagated = []

    @contextmanager
    def propagate_attributes(**kwargs):
        propagated.append(kwargs)
        yield

    monkeypatch.setattr(tracing, "_langfuse", langfuse)
    monkeypatch.setattr(tracing, "_propagate_attributes", propagate_attributes)
    monkeypatch.setattr(tracing, "_capture_content", False)

    with tracing.trace_operation(
        "chat-response",
        feature="chat",
        input={"message": "private student text"},
        session_id="session-1",
    ), tracing.trace_generation(
        "generate-answer",
        provider="groq",
        model="test-model",
        input=[{"role": "user", "content": "private student text"}],
    ) as generation:
        tracing.update_generation(
            generation,
            output="private model response",
            usage_details={"input": 12, "output": 8, "total": 20},
        )

    root_args, _ = langfuse.observations[0]
    generation_args, generation = langfuse.observations[1]
    assert root_args["input"] == {
        "content_capture": False,
        "input_characters": len("private student text"),
    }
    assert generation_args["input"]["content_capture"] is False
    assert "private student text" not in repr(generation_args["input"])
    assert generation.updates == [
        {
            "output": {
                "content_capture": False,
                "output_characters": len("private model response"),
            },
            "usage_details": {"input": 12, "output": 8, "total": 20},
        }
    ]
    assert propagated == [{"session_id": "session-1"}]


def test_tracing_can_capture_content_when_explicitly_enabled(monkeypatch):
    langfuse = FakeLangfuse()
    monkeypatch.setattr(tracing, "_langfuse", langfuse)
    monkeypatch.setattr(tracing, "_capture_content", True)

    with tracing.trace_generation(
        "generate-answer",
        provider="gemini",
        model="test-model",
        input={"prompt": "approved test prompt"},
    ) as generation:
        tracing.update_generation(generation, output="approved test output")

    args, observation = langfuse.observations[0]
    assert args["input"] == {"prompt": "approved test prompt"}
    assert observation.updates == [{"output": "approved test output"}]


def test_token_usage_maps_provider_response_fields():
    assert tracing.token_usage(
        {"usage": {"prompt_tokens": 12, "completion_tokens": 8}},
        "groq",
    ) == {"input": 12, "output": 8, "total": 20}
    assert tracing.token_usage(
        {"usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 5}},
        "gemini",
    ) == {"input": 7, "output": 5, "total": 12}
    assert tracing.token_usage(
        {"prompt_eval_count": 4, "eval_count": 3},
        "ollama",
    ) == {"input": 4, "output": 3, "total": 7}
    assert tracing.token_usage({}, "openai") is None


def test_export_mask_removes_observation_content_by_default(monkeypatch):
    identifier = object()
    span = SimpleNamespace(
        attributes={
            "langfuse.observation.input": '"private prompt"',
            "langfuse.observation.output": '"private response"',
            "langfuse.observation.model.name": '"test-model"',
        }
    )
    params = SimpleNamespace(spans={identifier: span})
    monkeypatch.setattr(tracing, "_capture_content", False)

    result = tracing.mask_otel_spans(params=params)

    patch = result.span_patches[identifier]
    assert set(patch.delete_attributes) == {
        "langfuse.observation.input",
        "langfuse.observation.output",
    }
    assert patch.set_attributes == {"app.content_capture": False}


def test_export_mask_can_be_disabled_for_explicit_content_capture(monkeypatch):
    monkeypatch.setattr(tracing, "_capture_content", True)

    assert tracing.mask_otel_spans(params=SimpleNamespace(spans={})) is None

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

_capture_content = os.getenv("LANGFUSE_CAPTURE_CONTENT", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _tracing_enabled() -> bool:
    return _env_flag("USE_LANGFUSE") and _env_flag(
        "LANGFUSE_TRACING_ENABLED", default=True
    )


_enabled = _tracing_enabled()
_langfuse = None
_propagate_attributes = None

def mask_otel_spans(*, params: Any) -> Any:
    if _capture_content:
        return None

    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    patches = {}
    sensitive_keys = {
        "langfuse.observation.input",
        "langfuse.observation.output",
        "gen_ai.prompt",
        "gen_ai.completion",
    }
    for identifier, span in params.spans.items():
        sensitive_attributes = tuple(
            key
            for key, value in span.attributes.items()
            if key in sensitive_keys and not _is_safe_content_summary(value)
        )
        if sensitive_attributes:
            patches[identifier] = OtelSpanPatch(
                delete_attributes=sensitive_attributes,
                set_attributes={"app.content_capture": False},
            )
    return MaskOtelSpansResult(span_patches=patches) if patches else None


def _is_safe_content_summary(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and parsed.get("content_capture") is False


if _enabled and os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    from langfuse import Langfuse, propagate_attributes

    _langfuse = Langfuse(
        mask_otel_spans=mask_otel_spans,
        environment=os.getenv("ENVIRONMENT", "development"),
    )
    _propagate_attributes = propagate_attributes


def _input_length(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(_input_length(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_input_length(item) for item in value)
    return len(str(value)) if value is not None else 0


def _safe_input(value: Any) -> Any:
    if _capture_content:
        return value
    return {"content_capture": False, "input_characters": _input_length(value)}


@contextmanager
def trace_operation(
    name: str,
    *,
    feature: str,
    input: Any = None,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> Iterator[Any]:
    if _langfuse is None:
        yield None
        return

    attributes = {"feature": feature, **(metadata or {})}
    session_context = (
        _propagate_attributes(session_id=session_id[:200])
        if session_id and _propagate_attributes
        else nullcontext()
    )
    with session_context, _langfuse.start_as_current_observation(
        as_type="span",
        name=name,
        input=_safe_input(input),
        metadata=attributes,
    ) as span:
        yield span


@contextmanager
def trace_generation(
    name: str,
    *,
    provider: str,
    model: str,
    as_type: str = "generation",
    input: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any]:
    if _langfuse is None:
        yield None
        return

    with _langfuse.start_as_current_observation(
        as_type=as_type,
        name=name,
        model=model,
        input=_safe_input(input),
        metadata={"provider": provider, **(metadata or {})},
    ) as generation:
        yield generation


def update_generation(
    generation: Any,
    *,
    output: Any,
    summary: Any = None,
    usage_details: dict[str, int] | None = None,
) -> None:
    if generation is None:
        return

    recorded_output = output if _capture_content else (
        {"content_capture": False, **summary}
        if summary is not None
        else {"content_capture": False, "output_characters": _input_length(output)}
    )
    updates = {"output": recorded_output}
    if usage_details:
        updates["usage_details"] = usage_details
    generation.update(**updates)


def update_operation(
    observation: Any,
    *,
    output: Any,
    summary: Any = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if observation is None:
        return

    recorded_output = output if _capture_content else (
        {"content_capture": False, **summary}
        if summary is not None
        else {"content_capture": False, "output_characters": _input_length(output)}
    )
    updates = {"output": recorded_output}
    if metadata:
        updates["metadata"] = metadata
    observation.update(**updates)


def token_usage(
    payload: dict[str, Any], provider: str
) -> dict[str, int] | None:
    if provider == "ollama":
        input_tokens = payload.get("prompt_eval_count")
        output_tokens = payload.get("eval_count")
    elif provider == "gemini":
        usage = payload.get("usageMetadata") or {}
        if not isinstance(usage, dict):
            return None
        input_tokens = usage.get("promptTokenCount")
        output_tokens = usage.get("candidatesTokenCount")
    else:
        usage = payload.get("usage") or {}
        if not isinstance(usage, dict):
            return None
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")

    details = {}
    if isinstance(input_tokens, int) and not isinstance(input_tokens, bool):
        details["input"] = input_tokens
    if isinstance(output_tokens, int) and not isinstance(output_tokens, bool):
        details["output"] = output_tokens
    if details:
        details["total"] = sum(details.values())
        return details
    return None


def flush_traces() -> None:
    if _langfuse is not None:
        _langfuse.flush()


def langchain_callback_handler() -> Any:
    if _langfuse is None:
        return None
    trace_id = _langfuse.get_current_trace_id()
    parent_span_id = _langfuse.get_current_observation_id()
    if not trace_id or not parent_span_id:
        return None
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        logger.warning(
            "LANGFUSE_LANGCHAIN_INTEGRATION_UNAVAILABLE; using manual model observation"
        )
        return None
    return CallbackHandler(
        trace_context={"trace_id": trace_id, "parent_span_id": parent_span_id}
    )

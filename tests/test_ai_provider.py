import httpx
import pytest
from unittest.mock import patch

from learning_studio.ai import AIExtractionError, AIProvider
from learning_studio.config import Settings


def test_provider_errors_redact_configured_credentials():
    secret = "test-gemini-secret"
    settings = Settings(
        _env_file=None,
        ai_provider_order="gemini",
        gemini_api_key=secret,
    )
    provider = AIProvider(settings)
    request = httpx.Request(
        "POST",
        f"https://generativelanguage.googleapis.com/generate?key={secret}",
    )
    response = httpx.Response(429, request=request)
    with pytest.raises(httpx.HTTPStatusError) as http_error:
        response.raise_for_status()
    error = http_error.value

    with (
        patch.object(provider, "_chat", side_effect=error),
        pytest.raises(AIExtractionError) as caught,
    ):
        provider.answer("instruction", "context")

    assert secret not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)
    assert "429 Too Many Requests" in str(caught.value)

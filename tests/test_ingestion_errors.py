from types import SimpleNamespace

from learning_studio.ingestion_errors import is_daily_token_quota_error


def test_identifies_provider_daily_token_quota_error():
    error = SimpleNamespace(
        status_code=429,
        body={
            "error": {
                "message": "Limit reached on tokens per day (TPD)",
                "code": "rate_limit_exceeded",
            }
        },
    )

    assert is_daily_token_quota_error(error)


def test_does_not_classify_request_rate_limit_as_daily_token_quota():
    error = SimpleNamespace(
        status_code=429,
        body={"error": {"message": "Requests per minute limit reached"}},
    )

    assert not is_daily_token_quota_error(error)


def test_does_not_classify_other_http_errors_as_daily_token_quota():
    error = SimpleNamespace(
        status_code=400,
        body={"error": {"message": "Tokens per day limit reached"}},
    )

    assert not is_daily_token_quota_error(error)

class DailyTokenQuotaError(RuntimeError):
    """Raised when an AI provider's daily token allowance is exhausted."""


def is_daily_token_quota_error(error: Exception) -> bool:
    if getattr(error, "status_code", None) != 429:
        return False

    details = [str(error)]
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        api_error = body.get("error")
        if isinstance(api_error, dict):
            details.extend(
                str(api_error.get(key, ""))
                for key in ("message", "code", "type")
            )
        else:
            details.append(str(body))

    message = " ".join(details).lower()
    return "tokens per day" in message or "daily token" in message or "tpd" in message

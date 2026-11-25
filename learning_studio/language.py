import re

_TELUGU_LETTERS = re.compile(r"[\u0c05-\u0c39\u0c58-\u0c5a\u0c60-\u0c61]")
_LATIN_LETTERS = re.compile(r"[A-Za-z]")


def is_language_text(value: str, language: str) -> bool:
    telugu_count = len(_TELUGU_LETTERS.findall(value))
    latin_count = len(_LATIN_LETTERS.findall(value))
    if language == "en":
        return telugu_count == 0
    if language == "te":
        letter_count = telugu_count + latin_count
        return letter_count == 0 or (
            telugu_count > 0 and latin_count / letter_count <= 0.2
        )
    raise ValueError(f"Unsupported language: {language}")

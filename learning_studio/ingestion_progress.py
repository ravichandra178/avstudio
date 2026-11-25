import json
from pathlib import Path
from typing import Any


def load_chunk_progress(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        progress = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"Warning: could not read {path}: {error}. Reprocessing incomplete chunks.")
        return {}
    if not isinstance(progress, dict):
        print(f"Warning: invalid chunk progress in {path}. Reprocessing incomplete chunks.")
        return {}
    return {
        key: value
        for key, value in progress.items()
        if isinstance(key, str) and isinstance(value, dict)
    }


def save_chunk_progress(path: Path, progress: dict[str, dict[str, Any]]) -> None:
    temporary_file = path.with_suffix(".tmp")
    temporary_file.write_text(
        json.dumps(progress, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_file.replace(path)


def completed_chunk_ids(
    progress: dict[str, dict[str, Any]],
    state_key: str,
    state_value: str,
) -> set[str]:
    entry = progress.get(state_key)
    if entry is None or entry.get("source") != state_value:
        return set()
    completed = entry.get("completed")
    if not isinstance(completed, list):
        return set()
    return {chunk_id for chunk_id in completed if isinstance(chunk_id, str)}


def record_completed_chunks(
    progress: dict[str, dict[str, Any]],
    state_key: str,
    state_value: str,
    chunk_ids: set[str],
) -> None:
    completed = completed_chunk_ids(progress, state_key, state_value)
    completed.update(chunk_ids)
    progress[state_key] = {
        "source": state_value,
        "completed": sorted(completed),
    }

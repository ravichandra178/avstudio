from learning_studio.ingestion_progress import (
    completed_chunk_ids,
    load_chunk_progress,
    record_completed_chunks,
    save_chunk_progress,
)


def test_completed_chunks_persist_and_are_scoped_to_source_version(tmp_path):
    progress_file = tmp_path / "ingested_chunks.json"
    progress = {}
    record_completed_chunks(
        progress,
        "course.pdf",
        "sha256-version-1",
        {"chunk-a", "chunk-b"},
    )
    save_chunk_progress(progress_file, progress)

    loaded_progress = load_chunk_progress(progress_file)

    assert completed_chunk_ids(
        loaded_progress,
        "course.pdf",
        "sha256-version-1",
    ) == {"chunk-a", "chunk-b"}
    assert completed_chunk_ids(
        loaded_progress,
        "course.pdf",
        "sha256-version-2",
    ) == set()


def test_recording_chunks_preserves_prior_completed_chunks():
    progress = {}
    record_completed_chunks(progress, "course.pdf", "version", {"chunk-a"})
    record_completed_chunks(progress, "course.pdf", "version", {"chunk-b"})

    assert completed_chunk_ids(progress, "course.pdf", "version") == {
        "chunk-a",
        "chunk-b",
    }


def test_recording_new_source_version_drops_old_chunk_ids():
    progress = {}
    record_completed_chunks(progress, "course.pdf", "old-version", {"old-chunk"})
    record_completed_chunks(progress, "course.pdf", "new-version", {"new-chunk"})

    assert completed_chunk_ids(progress, "course.pdf", "new-version") == {"new-chunk"}

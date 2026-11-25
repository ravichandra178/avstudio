import io
import logging

from pypdf import PdfReader

from .ai import AIExtractionError, AIProvider
from .config import get_settings
from .graph import KnowledgeGraph
from .storage import DocumentStorage
from .tracing import flush_traces, trace_operation, update_operation

logger = logging.getLogger(__name__)

try:
    from celery import Celery

    celery = Celery(
        "learning_studio",
        broker=get_settings().redis_url,
        backend=get_settings().redis_url,
    )
    task_decorator = celery.task
except ImportError:
    celery = None

    def task_decorator(func):
        class _TaskWrapper:
            def __init__(self, fn):
                self.fn = fn

            def delay(self, *args, **kwargs):
                logger.warning(
                    "Celery is not installed. Task '%s' was invoked via delay in fallback mode.",
                    self.fn.__name__,
                )
                return None

            def __call__(self, *args, **kwargs):
                return self.fn(*args, **kwargs)

        return _TaskWrapper(func)


@task_decorator
def ingest_document(document_id: str, filename: str, storage_key: str) -> None:
    with trace_operation(
        "ingest-document",
        feature="document-ingestion",
        input={"document_id": document_id, "filename": filename},
        metadata={"document_id": document_id, "filename": filename},
    ) as observation:
        try:
            _ingest_document(document_id, filename, storage_key)
            update_operation(
                observation,
                output={"status": "completed"},
                summary={"status": "completed"},
            )
        finally:
            flush_traces()


def _ingest_document(document_id: str, filename: str, storage_key: str) -> None:
    settings = get_settings()
    storage = DocumentStorage(settings)
    graph = KnowledgeGraph(settings)
    try:
        graph.upsert_document(document_id, filename)
        reader = PdfReader(io.BytesIO(storage.get(storage_key)))
        chunks = []
        for index, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text:
                chunks.append({"id": f"{document_id}-{index}", "page": index + 1, "text": page_text})
        text = "\n".join(chunk["text"] for chunk in chunks)
        concepts = AIProvider(settings, feature="document-ingestion").extract(
            text, document_id
        )
        graph.save_concepts(document_id, concepts)
        if chunks:
            try:
                graph.ensure_vector_index(settings.embedding_dimensions)
                embeddings = AIProvider(
                    settings,
                    feature="document-ingestion",
                ).embed([chunk["text"] for chunk in chunks])
                graph.save_chunks(document_id, chunks, embeddings)
            except AIExtractionError:
                logger.warning("Vector indexing skipped for %s; embedding provider is not configured", document_id)
    finally:
        graph.close()

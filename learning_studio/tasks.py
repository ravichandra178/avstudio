import io
import logging

from celery import Celery
from pypdf import PdfReader

from .ai import AIProvider
from .ai import AIExtractionError
from .config import get_settings
from .graph import KnowledgeGraph
from .storage import DocumentStorage

celery = Celery("learning_studio", broker=get_settings().redis_url, backend=get_settings().redis_url)
logger = logging.getLogger(__name__)


@celery.task
def ingest_document(document_id: str, filename: str, storage_key: str) -> None:
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
        concepts = AIProvider(settings).extract(text, document_id)
        graph.save_concepts(document_id, concepts)
        if chunks:
            try:
                graph.ensure_vector_index(settings.embedding_dimensions)
                embeddings = AIProvider(settings).embed([chunk["text"] for chunk in chunks])
                graph.save_chunks(document_id, chunks, embeddings)
            except AIExtractionError:
                logger.warning("Vector indexing skipped for %s; embedding provider is not configured", document_id)
    finally:
        graph.close()

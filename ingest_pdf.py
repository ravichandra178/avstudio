import argparse
import atexit
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer

from learning_studio.ingestion_errors import (
    DailyTokenQuotaError,
    is_daily_token_quota_error,
)
from learning_studio.ingestion_progress import (
    completed_chunk_ids,
    load_chunk_progress,
    record_completed_chunks,
    save_chunk_progress,
)
from learning_studio.tracing import (
    flush_traces,
    langchain_callback_handler,
    trace_generation,
    trace_operation,
    update_generation,
    update_operation,
)

load_dotenv()
atexit.register(flush_traces)
use_model = os.getenv("USE_MODEL", "OLLAMA").strip().upper()
supported_models = {"GROK", "GEMINI", "OLLAMA"}
if use_model not in supported_models:
    raise ValueError(f"USE_MODEL must be one of: {', '.join(sorted(supported_models))}")
pdf_directory = Path("./my_pdfs")
parser = argparse.ArgumentParser(description="Ingest PDFs into the knowledge graph.")
parser.add_argument(
    "--pdf",
    help="Ingest one PDF from ./my_pdfs (accepts a filename or path). Default: all PDFs.",
)
parser.add_argument(
    "--pages",
    help="Ingest only a page range, for example: --pages 5-6",
)
args = parser.parse_args()
selected_pdf = None
if args.pdf:
    requested_pdf = Path(args.pdf)
    if not requested_pdf.is_absolute():
        requested_pdf = (
            pdf_directory / requested_pdf
            if requested_pdf.parent == Path(".")
            else Path.cwd() / requested_pdf
        )
    requested_pdf = requested_pdf.resolve()
    if (
        requested_pdf.parent != pdf_directory.resolve()
        or requested_pdf.suffix.lower() != ".pdf"
    ):
        parser.error("--pdf must identify a PDF located directly in ./my_pdfs")
    if not requested_pdf.is_file():
        parser.error(f"PDF not found in ./my_pdfs: {requested_pdf.name}")
    selected_pdf = requested_pdf


def report(message: str) -> None:
    print(f"[{datetime.now().astimezone().strftime('%H:%M:%S')}] {message}", flush=True)

# 1. Connect to Neo4j Aura Free Tier
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
GROUP_NAME = os.getenv("STUDY_GROUP", "Group 1")
TEXTBOOK_OVERRIDE = os.getenv("TEXTBOOK_NAME", "").strip()

report(f"Connecting to Neo4j database '{NEO4J_DATABASE}'")
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    database=NEO4J_DATABASE,
    password=NEO4J_PASSWORD
)
report("Connected to Neo4j")


def textbook_for_pdf(pdf_path: Path) -> str:
    filename = pdf_path.stem.lower()
    if all(term in filename for term in ("environment", "agriculture")) or "disaster" in filename:
        return "Environment, Agriculture and Disaster Management"
    if "science" in filename and "technology" in filename:
        return "Science and Technology"
    if TEXTBOOK_OVERRIDE:
        return TEXTBOOK_OVERRIDE
    return " ".join(pdf_path.stem.replace("_", " ").replace("-", " ").split()).strip()


def create_study_context(pdf_path: Path) -> None:
    textbook_name = textbook_for_pdf(pdf_path)
    textbook_id = textbook_name.strip().lower().replace(" ", "-").replace(",", "")
    document_name = pdf_path.name
    graph.query(
        """
        MERGE (g:StudyGroup {name: $group_name})
        MERGE (t:Textbook {id: $textbook_id})
        SET t.name = $textbook_name
        MERGE (g)-[:USES_TEXTBOOK]->(t)
        MERGE (d:Document {name: $document_name})
        SET d.source = $document_name
        MERGE (t)-[:CONTAINS_DOCUMENT]->(d)
        MERGE (d)-[:BELONGS_TO_GROUP]->(g)
        """,
        params={
            "group_name": GROUP_NAME,
            "textbook_id": textbook_id,
            "textbook_name": textbook_name,
            "document_name": document_name,
        },
    )
    report(f"Linked {document_name} to {GROUP_NAME} / {textbook_name}")

# 2. Function to extract text from a single PDF in chunks
def extract_text_from_pdf(
    pdf_path: str,
    chunk_size: int,
    max_pages_per_chunk: int = 3,
    page_range: tuple[int, int] | None = None,
):
    doc = fitz.open(pdf_path)
    chunks = []
    file_name = os.path.basename(pdf_path)
    page_buffer: list[str] = []
    first_page = 1
    last_page = 0

    if page_range and page_range[1] > len(doc):
        raise ValueError(
            f"--pages {page_range[0]}-{page_range[1]} is outside the physical "
            f"PDF page range 1-{len(doc)}"
        )

    def flush_buffer(last_page: int) -> None:
        if not page_buffer:
            return
        chunks.append(
            Document(
                page_content="\n\n".join(page_buffer),
                metadata={
                    "source": file_name,
                    "page": f"{first_page}-{last_page}" if first_page != last_page else first_page,
                },
            )
        )

    first_requested = page_range[0] - 1 if page_range else 0
    last_requested = min(page_range[1], len(doc)) if page_range else len(doc)
    for page_num in range(first_requested, last_requested):
        text = doc[page_num].get_text()
        if not text.strip():
            continue
        if not page_buffer:
            first_page = page_num + 1
        page_buffer.append(text.strip())
        last_page = page_num + 1
        if len("\n\n".join(page_buffer)) >= chunk_size or len(page_buffer) >= max_pages_per_chunk:
            flush_buffer(last_page)
            page_buffer = []
    flush_buffer(last_page)
    return chunks


def chunk_size_for_file(size_bytes: int) -> int:
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > 25:
        return 3500
    if size_mb > 10:
        return 4500
    return 6000

# 3. Define ingestion settings
tracking_file = Path(".ingested_pdfs.json")
chunk_tracking_file = Path(".ingested_pdf_chunks.json")
max_pdf_size = int(os.getenv("MAX_PDF_SIZE_MB", "50")) * 1024 * 1024
ollama_selected = use_model == "OLLAMA"
batch_size = max(1, int(os.getenv("INGEST_BATCH_SIZE", "1" if ollama_selected else "1")))
ollama_timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
request_interval = float(os.getenv("GEMINI_REQUEST_INTERVAL_SECONDS", "15"))
if ollama_selected:
    request_interval = 0
last_ai_request = 0.0


def wait_for_ai_rate_limit() -> None:
    global last_ai_request
    elapsed = time.monotonic() - last_ai_request
    remaining = request_interval - elapsed
    if remaining > 0:
        report(f"Rate limit pacing: waiting {remaining:.1f}s before next AI request")
        time.sleep(remaining)
    last_ai_request = time.monotonic()


def file_hash(pdf_path: Path) -> str:
    digest = hashlib.sha256()
    with pdf_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ingestion_state() -> dict[str, str]:
    if not tracking_file.exists():
        return {}
    try:
        return json.loads(tracking_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"Warning: could not read {tracking_file}: {error}. Reprocessing PDFs.")
        return {}


def save_ingestion_state(state: dict[str, str]) -> None:
    temporary_file = tracking_file.with_suffix(".tmp")
    temporary_file.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    temporary_file.replace(tracking_file)


if not pdf_directory.is_dir():
    raise FileNotFoundError(f"PDF directory does not exist: {pdf_directory}")

ingestion_state = load_ingestion_state()
chunk_progress = load_chunk_progress(chunk_tracking_file)
if selected_pdf is not None:
    pdf_files = [selected_pdf]
else:
    pdf_files = sorted(
        path
        for path in pdf_directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
pending_files = []
for pdf_path in pdf_files:
    size_bytes = pdf_path.stat().st_size
    if size_bytes > max_pdf_size:
        report(
            f"Skipping oversized PDF: {pdf_path} "
            f"({size_bytes / (1024 * 1024):.1f} MB > {max_pdf_size / (1024 * 1024):.0f} MB limit)"
        )
        continue
    create_study_context(pdf_path)
    page_range = None
    if args.pages:
        match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", args.pages)
        if not match:
            parser.error("--pages must be a range such as 5-6")
        first_page, last_page = (int(value) for value in match.groups())
        if first_page < 1 or last_page < first_page:
            parser.error("--pages must contain a valid ascending page range")
        page_range = (first_page, last_page)
    state_key = str(pdf_path)
    if page_range:
        state_key = f"{state_key}#pages={page_range[0]}-{page_range[1]}"
    current_hash = file_hash(pdf_path)
    state_value = f"{current_hash}#pages={page_range[0]}-{page_range[1]}" if page_range else current_hash
    if ingestion_state.get(state_key) == state_value:
        print(f"Skipping unchanged PDF sample: {pdf_path} pages {page_range[0]}-{page_range[1]}")
    else:
        pending_files.append((pdf_path, state_key, state_value, page_range))

if not pending_files:
    print("No new or changed PDFs to ingest.")
    raise SystemExit(0)

# 4. Initialize the LLM selected for PDF-to-graph extraction.
if use_model == "GROK":
    if not os.getenv("GROK_API_KEY") or not os.getenv("GROK_MODEL"):
        raise RuntimeError("USE_MODEL=GROK requires GROK_API_KEY and GROK_MODEL in .env")

    from langchain_openai import ChatOpenAI

    llm_provider = "grok"
    llm_model = os.environ["GROK_MODEL"].strip()
    grok_base_url = (
        os.getenv("GROK_ENDPOINT", "").strip()
        or "https://api.groq.com/openai/v1"
    ).rstrip("/")
    if grok_base_url.endswith("/chat/completions"):
        grok_base_url = grok_base_url.removesuffix("/chat/completions")
    llm = ChatOpenAI(
        api_key=os.environ["GROK_API_KEY"],
        base_url=grok_base_url,
        model=llm_model,
        temperature=0,
        max_retries=0,
        timeout=120,
    )
elif use_model == "OLLAMA":
    if not os.getenv("OLLAMA_MODEL"):
        raise RuntimeError("USE_MODEL=OLLAMA requires OLLAMA_MODEL in .env")

    from langchain_ollama import ChatOllama

    llm_provider = "ollama"
    llm_model = os.environ["OLLAMA_MODEL"]
    llm = ChatOllama(
        base_url=os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"),
        model=os.environ["OLLAMA_MODEL"],
        temperature=0,
        num_ctx=int(os.getenv("OLLAMA_CONTEXT_TOKENS", "8192")),
        num_predict=int(os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "1024")),
        sync_client_kwargs={"timeout": ollama_timeout},
    )
else:
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("USE_MODEL=GEMINI requires GEMINI_API_KEY in .env")

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm_provider = "gemini"
    llm_model = os.getenv(
        "GEMINI_MODEL", "models/gemini-3.8-flash"
    ).removeprefix("models/")
    llm = ChatGoogleGenerativeAI(
        google_api_key=os.environ["GEMINI_API_KEY"],
        model=llm_model,
        temperature=0,
        max_retries=0,
        timeout=120,
    )

llm_transformer = LLMGraphTransformer(
    llm=llm,
    ignore_tool_usage=use_model == "GROK",
    allowed_nodes=[
        "StudyGroup",
        "Textbook",
        "Document",
        "Chapter",
        "Concept",
        "Topic",
        "Person",
        "Organization",
        "Location",
        "Event",
    ],
    allowed_relationships=[
        "USES_TEXTBOOK",
        "CONTAINS_DOCUMENT",
        "BELONGS_TO_GROUP",
        "CONTAINS",
        "RELATED_TO",
        "PART_OF",
        "CAUSES",
        "OCCURS_IN",
        "AUTHORED_BY",
    ],
)
report(f"Using {llm_provider}:{llm_model} for graph extraction")


def convert_batch_with_tracing(
    batch_docs: list[Document],
    batch_number: int,
    total_batches: int,
):
    source_text = [document.page_content for document in batch_docs]
    with trace_operation(
        "extract-pdf-graph",
        feature="pdf-ingestion",
        input=source_text,
        metadata={"batch": batch_number, "chunk_count": len(batch_docs)},
    ) as operation:
        callback = langchain_callback_handler()
        for attempt in range(1, 4):
            try:
                wait_for_ai_rate_limit()
                started_at = time.monotonic()
                report(
                    f"Batch {batch_number}/{total_batches}: "
                    f"{llm_provider.capitalize()} is generating..."
                )
                if callback is not None:
                    graph_documents = llm_transformer.convert_to_graph_documents(
                        batch_docs,
                        config={"callbacks": [callback]},
                    )
                else:
                    with trace_generation(
                        "extract-graph-from-text",
                        provider=llm_provider,
                        model=llm_model,
                        input=source_text,
                        metadata={"attempt": attempt, "chunk_count": len(batch_docs)},
                    ) as generation:
                        graph_documents = llm_transformer.convert_to_graph_documents(
                            batch_docs
                        )
                        node_count = sum(len(item.nodes) for item in graph_documents)
                        relationship_count = sum(
                            len(item.relationships) for item in graph_documents
                        )
                        result_summary = {
                            "node_count": node_count,
                            "relationship_count": relationship_count,
                        }
                        update_generation(
                            generation,
                            output=result_summary,
                            summary=result_summary,
                        )
                node_count = sum(len(item.nodes) for item in graph_documents)
                relationship_count = sum(
                    len(item.relationships) for item in graph_documents
                )
                result_summary = {
                    "node_count": node_count,
                    "relationship_count": relationship_count,
                }
                update_operation(
                    operation,
                    output=result_summary,
                    summary=result_summary,
                )
                report(
                    f"Batch {batch_number}/{total_batches}: AI finished in "
                    f"{time.monotonic() - started_at:.1f}s"
                )
                return graph_documents
            except Exception as error:
                status_code = getattr(error, "status_code", None)
                if is_daily_token_quota_error(error):
                    report(
                        f"Batch {batch_number}: {llm_provider} daily token quota "
                        "exhausted; stopping this ingestion run without retrying "
                        "remaining batches"
                    )
                    raise DailyTokenQuotaError(
                        f"{llm_provider} daily token quota exhausted; "
                        "retry the PDF ingestion after the quota resets"
                    ) from None
                is_permanent_client_error = (
                    isinstance(status_code, int)
                    and 400 <= status_code < 500
                    and status_code not in {408, 429}
                )
                if is_permanent_client_error:
                    error_body = getattr(error, "body", None)
                    if isinstance(error_body, dict):
                        error_message = error_body.get("message", "request rejected")
                        error_code = error_body.get("code")
                        detail = (
                            f"code={error_code}: {error_message}"
                            if error_code
                            else str(error_message)
                        )
                    else:
                        detail = "request rejected"
                    report(
                        f"Batch {batch_number}: non-retryable {llm_provider} "
                        f"HTTP {status_code}: {detail[:300]}"
                    )
                    raise
                if attempt == 3:
                    raise
                backoff = max(request_interval, min(60, attempt * 15))
                report(
                    f"Batch {batch_number}: attempt {attempt} failed "
                    f"({type(error).__name__}); retrying in {backoff}s"
                )
                time.sleep(backoff)
    raise RuntimeError("PDF graph extraction exhausted retries without a result")

# 5. Process each PDF incrementally
report(
    f"Starting ingestion: {len(pending_files)} PDF(s), "
    f"batch size {batch_size}, AI interval {request_interval:.1f}s, "
    f"max file size {max_pdf_size / (1024 * 1024):.0f} MB"
)

for pdf_path, state_key, state_value, page_range in pending_files:
    size_bytes = pdf_path.stat().st_size
    chunk_size = chunk_size_for_file(size_bytes)
    report(
        f"Processing {pdf_path.name} "
        f"({size_bytes / (1024 * 1024):.1f} MB, {chunk_size} characters/chunk)"
    )
    documents = extract_text_from_pdf(
        str(pdf_path),
        chunk_size,
        page_range=page_range,
    )
    if not documents:
        page_description = (
            f" for physical pages {page_range[0]}-{page_range[1]}"
            if page_range
            else ""
        )
        raise RuntimeError(
            f"No text chunks extracted from {pdf_path.name}{page_description}; "
            "refusing to mark this ingestion complete"
        )
    total_batches = (len(documents) + batch_size - 1) // batch_size
    report(f"Extracted {len(documents)} chunks across {total_batches} batch(es)")
    stored_documents = graph.query(
        """
        MATCH (d:Document)
        WHERE d.source = $source AND d.text IS NOT NULL
        RETURN d.id AS id
        """,
        params={"source": pdf_path.name},
    )
    stored_chunk_ids = {
        record["id"] for record in stored_documents if isinstance(record.get("id"), str)
    }
    locally_completed = completed_chunk_ids(chunk_progress, state_key, state_value)
    if locally_completed:
        report(
            f"Resuming {pdf_path.name}: {len(locally_completed)} chunk(s) recorded "
            "as completed locally"
        )
    if stored_chunk_ids:
        report(
            f"Found {len(stored_chunk_ids)} previously stored source chunk(s) "
            f"for {pdf_path.name}"
        )
    completed_ids = stored_chunk_ids | locally_completed
    file_succeeded = True
    
    # Process chunks in batches to manage token usage and memory
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_number = i // batch_size + 1
        batch_chunk_ids = [
            hashlib.md5(document.page_content.encode("utf-8")).hexdigest()
            for document in batch_docs
        ]
        unprocessed_batch = [
            (document, chunk_id)
            for document, chunk_id in zip(batch_docs, batch_chunk_ids, strict=True)
            if chunk_id not in completed_ids
        ]
        if not unprocessed_batch:
            report(f"Batch {batch_number}/{total_batches}: all chunks already stored; skipping")
            continue
        batch_docs = [document for document, _ in unprocessed_batch]
        batch_chunk_ids = [chunk_id for _, chunk_id in unprocessed_batch]
        report(f"Batch {batch_number}/{total_batches}: sending {len(batch_docs)} chunk(s) to AI")
        
        try:
            # Transform text chunk batch into graph nodes and relationships
            graph_documents = convert_batch_with_tracing(
                batch_docs,
                batch_number,
                total_batches,
            )
            
            # Store in Neo4j Aura (include_source links nodes back to the source PDF)
            graph.add_graph_documents(graph_documents, include_source=True)
            report(f"Batch {batch_number}/{total_batches}: stored in Neo4j")
            completed_ids.update(batch_chunk_ids)
            record_completed_chunks(
                chunk_progress,
                state_key,
                state_value,
                set(batch_chunk_ids),
            )
            save_chunk_progress(chunk_tracking_file, chunk_progress)
        except Exception as e:
            if isinstance(e, DailyTokenQuotaError):
                raise
            file_succeeded = False
            report(f"Batch {batch_number}/{total_batches} failed: {type(e).__name__}: {e}")
    if file_succeeded:
        ingestion_state[state_key] = state_value
        save_ingestion_state(ingestion_state)
        chunk_progress.pop(state_key, None)
        save_chunk_progress(chunk_tracking_file, chunk_progress)
        report(f"Completed and recorded: {pdf_path.name}")
    else:
        report(f"Not marked complete; rerun will retry failed batches: {pdf_path.name}")

report("Ingestion run finished")
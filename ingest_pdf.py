import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
import fitz  # PyMuPDF

load_dotenv()


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
def extract_text_from_pdf(pdf_path: str, chunk_size: int, max_pages_per_chunk: int = 3):
    doc = fitz.open(pdf_path)
    chunks = []
    file_name = os.path.basename(pdf_path)
    page_buffer: list[str] = []
    first_page = 1
    last_page = 0

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

    for page_num in range(len(doc)):
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

# 3. Define directory containing your large PDFs
pdf_directory = Path("./my_pdfs")
tracking_file = Path(".ingested_pdfs.json")
max_pdf_size = int(os.getenv("MAX_PDF_SIZE_MB", "50")) * 1024 * 1024
ollama_selected = bool(os.getenv("OLLAMA_MODEL"))
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
pdf_files = sorted(
    path for path in pdf_directory.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"
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
    current_hash = file_hash(pdf_path)
    if ingestion_state.get(str(pdf_path)) == current_hash:
        print(f"Skipping unchanged PDF: {pdf_path}")
    else:
        pending_files.append((pdf_path, current_hash))

if not pending_files:
    print("No new or changed PDFs to ingest.")
    raise SystemExit(0)

# 4. Initialize the LLM used only for PDF-to-graph extraction.
if os.getenv("OLLAMA_MODEL"):
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        base_url=os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"),
        model=os.environ["OLLAMA_MODEL"],
        temperature=0,
        num_ctx=int(os.getenv("OLLAMA_CONTEXT_TOKENS", "8192")),
        num_predict=int(os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "1024")),
        sync_client_kwargs={"timeout": ollama_timeout},
    )
elif os.getenv("GEMINI_API_KEY"):
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        google_api_key=os.environ["GEMINI_API_KEY"],
        model=os.getenv("GEMINI_MODEL", "models/gemini-3.8-flash").removeprefix("models/"),
        temperature=0,
        max_retries=0,
        timeout=120,
    )
elif os.getenv("OPENAI_API_KEY"):
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
else:
    raise RuntimeError(
        "PDF graph extraction requires OLLAMA_MODEL, GEMINI_API_KEY, or OPENAI_API_KEY in .env. "
        "The chatbot is not required to run ingestion."
    )

llm_transformer = LLMGraphTransformer(
    llm=llm,
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
report(f"Using {llm.__class__.__name__} for graph extraction")

# 5. Process each PDF incrementally
report(
    f"Starting ingestion: {len(pending_files)} PDF(s), "
    f"batch size {batch_size}, AI interval {request_interval:.1f}s, "
    f"max file size {max_pdf_size / (1024 * 1024):.0f} MB"
)

for pdf_path, current_hash in pending_files:
    size_bytes = pdf_path.stat().st_size
    chunk_size = chunk_size_for_file(size_bytes)
    report(
        f"Processing {pdf_path.name} "
        f"({size_bytes / (1024 * 1024):.1f} MB, {chunk_size} characters/chunk)"
    )
    documents = extract_text_from_pdf(str(pdf_path), chunk_size)
    total_batches = (len(documents) + batch_size - 1) // batch_size
    report(f"Extracted {len(documents)} chunks across {total_batches} batch(es)")
    file_succeeded = True
    
    # Process chunks in batches to manage token usage and memory
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_number = i // batch_size + 1
        report(f"Batch {batch_number}/{total_batches}: sending {len(batch_docs)} chunk(s) to AI")
        
        try:
            # Transform text chunk batch into graph nodes and relationships
            graph_documents = None
            for attempt in range(1, 4):
                try:
                    wait_for_ai_rate_limit()
                    started_at = time.monotonic()
                    report(f"Batch {batch_number}/{total_batches}: Ollama is generating...")
                    graph_documents = llm_transformer.convert_to_graph_documents(batch_docs)
                    report(
                        f"Batch {batch_number}/{total_batches}: AI finished in "
                        f"{time.monotonic() - started_at:.1f}s"
                    )
                    break
                except Exception as error:
                    if attempt == 3:
                        raise
                    backoff = max(request_interval, min(60, attempt * 15))
                    report(
                        f"Batch {batch_number}: attempt {attempt} failed "
                        f"({type(error).__name__}); retrying in {backoff}s"
                    )
                    time.sleep(backoff)
            
            # Store in Neo4j Aura (include_source links nodes back to the source PDF)
            graph.add_graph_documents(graph_documents, include_source=True)
            report(f"Batch {batch_number}/{total_batches}: stored in Neo4j")
        except Exception as e:
            file_succeeded = False
            report(f"Batch {batch_number}/{total_batches} failed: {type(e).__name__}: {e}")
    if file_succeeded:
        ingestion_state[str(pdf_path)] = current_hash
        save_ingestion_state(ingestion_state)
        report(f"Completed and recorded: {pdf_path.name}")
    else:
        report(f"Not marked complete; rerun will retry failed batches: {pdf_path.name}")

report("Ingestion run finished")
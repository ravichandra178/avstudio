# Learning Studio

A FastAPI service that turns uploaded PDFs into a cited educational knowledge graph.

## Architecture

```mermaid
flowchart LR
    Client --> API[FastAPI]
    API --> S3[(S3 PDF storage)]
    API --> Redis[(Redis queue)]
    Redis --> Worker[Celery ingestion worker]
    Worker --> S3
    Worker --> AI[GitHub Models / Groq / OpenAI fallback]
    Worker --> Neo4j[(Neo4j knowledge graph)]
    Client --> API
    API --> Neo4j
```

## Run locally

1. Copy `.env.example` to `.env` and configure credentials.
2. Start dependencies: `docker compose up --build`.
3. Open Swagger UI at `http://localhost:8000/docs`.
4. Upload a PDF:

   ```bash
   curl -F "file=@course.pdf" http://localhost:8000/api/v1/documents
   ```

### Deploying with `learning_studio` as the Vercel Root Directory

Set Vercel's **Root Directory** to `learning_studio`. That folder contains its own
`vercel.json`, `requirements.txt`, and `api/index.py` entrypoint. The entrypoint exposes
the FastAPI application while preserving package-relative imports.

Endpoints include `GET /health`, `GET /api/v1/search?q=...`, and `GET /api/v1/graph`.

The structured study experience uses:

- A startup selector for `Group` and `Textbook`, persisted with each study session.
- `POST /api/v1/sessions` to create a learning session.
- `POST /api/v1/sessions/{session_id}/chat` for context-aware, textbook-grounded chat.
- `POST /api/v1/sessions/{session_id}/questions` with `mode` set to `textbook`, `previous`, or
  `repeated` to play one question on demand.

Conversation messages and generated questions are persisted in Neo4j, so the tutor can reference
earlier questions and revisit repeated concepts with page-level citations.

To pre-generate textbook questions, ensure the configured Ollama model is installed and run:

```bash
.venv/bin/python ingest_question.py --textbook "Courseware on Science and Technology.pdf"
```

The script reads direct `MENTIONS` entities from Neo4j, generates one grounded MCQ per entity,
and stores reusable `StudyQuestion` nodes. Provider order is Gemini, Grok, then local Ollama.
Set `USE_MODEL` to `GEMINI`, `GROK`, or `OLLAMA` to select the primary ingestion provider.
The default and this workspace's `.env` select `OLLAMA`, using the local
`qwen2.5-1.5b-instruct:local` alias copied from the downloaded Qwen model. Set `OLLAMA_MODEL` to an
installed Ollama model name. Generation runs through the local Ollama API at `OLLAMA_ENDPOINT`;
ingestion does not call Hugging Face. When selected, Ollama is used directly. Gemini falls back to
Grok and then Ollama;
Grok falls back to Ollama. Configure
`GROK_API_KEY` for Grok; `GROK_MODEL` defaults to
`grok-4.7`. Grok HTTP
failures log the status and a truncated, API-key-redacted response body. Gemini requests time out after
8 seconds by default, return at most 256 tokens, and a Gemini failure disables further Gemini calls
for that run. Grok requests default to a 15-second timeout and also stop being retried after a
failure. Adjust `GEMINI_QUESTION_TIMEOUT_SECONDS`, `GEMINI_QUESTION_MAX_OUTPUT_TOKENS`,
`GROK_QUESTION_TIMEOUT_SECONDS`, `GROK_QUESTION_MAX_OUTPUT_TOKENS`, `QUESTION_INGEST_WORKERS`, and
`QUESTION_INGEST_BATCH_SIZE` as needed. Generation uses four concurrent workers, but Ollama calls
are limited to one at a time by default (`QUESTION_INGEST_OLLAMA_WORKERS`) to avoid overloading a
local model after Gemini failures. Logs distinguish Ollama queue wait from model request duration.
Neo4j reads and writes are batched. Use `--limit` to test a small batch first. The question player
serves stored questions immediately. Ingestion logs configuration (excluding credentials), entity
loading, existing-question skips, per-provider attempts/results and durations, batch
storage/results, and run totals; failures include exception details for troubleshooting.

All generated questions are four-option multiple-choice questions. Set `QUESTION_LEVEL` in `.env`
to control the target standard; the default is `Default`.

When a question has no matching local graph material, configure `GEMINI_API_KEY` and the tutor
uses Gemini as the knowledge fallback. If the graph only has a bare entity mention, Gemini Search
grounding is used with a 12-second timeout (`CHAT_FALLBACK_TIMEOUT_SECONDS` can adjust it). Local
graph results remain the preferred source.

Learning-step summaries are extracted from matching textbook passages stored on `HAS_CHUNK` nodes
or legacy source-document text. Entity mentions without a passage are not presented as summaries.

## Vector search prerequisites

Vector search is a post-extraction ingestion step: PDF text and concepts are extracted first,
then chunks are embedded and stored as `Chunk` nodes in Neo4j. Configure `OPENAI_API_KEY` (or an
OpenAI-compatible embedding endpoint) plus `EMBEDDING_MODEL` and `EMBEDDING_DIMENSIONS`. Neo4j 5
with vector-index support is required. Without embedding configuration, ingestion still completes
and search falls back to concept matching; configure embeddings before relying on semantic search.

The UI includes contextual suggestion chips and a session progress bar. Progress is based on
persisted chat messages and question-player activity rather than a hard-coded percentage.

For production, use AWS IAM roles for S3 access, private S3/Neo4j networking, TLS, a secrets manager, authentication, rate limiting, malware scanning, and ECS task definitions with separate API and worker services.

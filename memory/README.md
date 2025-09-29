# Mem0-Compatible Memory Service

FastAPI + SQLite implementation of the Mem0 memory API. It accepts the same request/response shapes as the official service so the `mem0ai` clients can target it by switching their base URL.

## Prerequisites

- Python 3.12 (via `uv` or system interpreter)
- SQLite (bundled with Python)

## Local Setup

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env  # set MEMORY_DB_PATH and MEM0_API_KEY
```

## Running the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

All authenticated requests must include the header `Authorization: Token <MEM0_API_KEY>`.

## Endpoint Cheat Sheet

```bash
# Health check (no auth required)
curl http://localhost:8000/health

# Add memories (v1)
curl -X POST http://localhost:8000/v1/memories/ \
  -H "Authorization: Token changeme" \
  -H "Content-Type: application/json" \
  -d '{
        "messages": [{"role": "user", "content": "My name is Alice"}],
        "user_id": "alice",
        "agent_id": "agent-123",
        "metadata": {"source": "chat"}
      }'

# Fetch a single memory (v1)
curl -H "Authorization: Token changeme" http://localhost:8000/v1/memories/<memory_id>/

# Update a memory (v1)
curl -X PUT http://localhost:8000/v1/memories/<memory_id>/ \
  -H "Authorization: Token changeme" \
  -H "Content-Type: application/json" \
  -d '{"text": "user: Updated fact", "metadata": {"source": "chat", "updated": true}}'

# Delete a memory (v1)
curl -X DELETE -H "Authorization: Token changeme" http://localhost:8000/v1/memories/<memory_id>/

# List memories with filters (v2)
curl -X POST http://localhost:8000/v2/memories/ \
  -H "Authorization: Token changeme" \
  -H "Content-Type: application/json" \
  -d '{"filters": {"user_id": "alice"}, "page": 1, "page_size": 100}'

# Semantic search (v2)
curl -X POST http://localhost:8000/v2/memories/search/ \
  -H "Authorization: Token changeme" \
  -H "Content-Type: application/json" \
  -d '{"query": "favorite color", "filters": {"user_id": "alice"}, "limit": 5}'
```

## Manual Smoke Test

1. Start the API with `uvicorn app.main:app --reload`.
2. Use `POST /v1/memories/` to ingest sample messages and confirm `results` include generated IDs.
3. Call `GET /v1/memories/{memory_id}/` to verify persisted fields (user, agent, metadata, timestamps).
4. Exercise `PUT /v1/memories/{memory_id}/` and `DELETE /v1/memories/{memory_id}/`, confirming updated text/metadata and the delete acknowledgement.
5. Try `POST /v2/memories/search/` and `POST /v2/memories/` with filters to validate scoring and pagination payloads.

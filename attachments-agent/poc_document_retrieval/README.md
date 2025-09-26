# Document Chunk Service POC

A FastAPI-based service for document chunking and retrieval without LLM generation. This service extracts text from documents, splits it into chunks, and retrieves relevant chunks based on user queries.

## Usage

### Starting the Server

Run the server with a configuration file:

```bash
export CHUNK_SERVICE_CONFIG="./config/bge_m3_400.yaml"
uv run uvicorn modules.main:app
```

### API Endpoints

Once running, the API will be available at `http://localhost:8000` with the following endpoints:

#### 1. Upload and Process Document
`POST /chunk-upload`

Upload a document file and retrieve relevant chunks based on chat history.

**Form Data:**
- `file`: Document file (PDF or TXT)
- `chat_history`: JSON string of chat history
- `max_tokens`: Optional maximum tokens. Default: -1 (None)
- `clear_retrieval_history`: Remove previously retrieved chunks in the conversation. Default: False

**Example Request:**
```bash
curl -X POST "http://localhost:8000/chunk-upload" \
     -F "file=@document.pdf" \
     -F 'chat_history=[{"role": "user", "content": "What is this document about?"}]'
```

**Response:**
```json
{
  "chat_history": [
    {
      "role": "user", 
      "content": "Retrieved relevant information from the document:\n\nChunk 1:\n[retrieved content]"
    },
    {"role": "user", "content": "What is this document about?"}
  ],
  "chunks_count": 25,
  "retrieved_chunks_count": 5,
  "document_info": {
    "file_path": "/tmp/document.pdf",
    "file_size": 1024000,
    "text_length": 50000,
    "chunks_created": 25
  }
}
```

#### 2. Health Check
`GET /health`

Check service health status.

#### 3. API Documentation
`GET /docs`

Interactive API documentation (Swagger UI).

## Configuration

The service uses YAML configuration files compatible with the existing attachments-agent configs. Example configuration:

```yaml
name: "BAAI BGE-M3 (400 chunks)"
chunk_size: 400
retrieval_method: "sentence-transformers"
retrieval_kwargs:
  model_name: "BAAI/bge-m3"
reranker_method: null
processor: "pypdf2"
max_tokens: 4000
top_k_retrieval: 10  # Number of chunks to retrieve initially
top_k_rerank: 5      # Number of chunks to keep after reranking
```
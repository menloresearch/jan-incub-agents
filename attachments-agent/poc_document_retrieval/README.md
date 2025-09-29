# Document Chunk Service POC

A FastAPI-based service for document chunking and retrieval without LLM generation. This service extracts text from documents, splits it into chunks, and retrieves relevant chunks based on user queries.

## Setup

### Prerequisites

- Python 3.12+
- Virtual environment (recommended)

### Installation

1. Navigate to the POC directory:
```bash
cd poc_document_retrieval
```

2. Create and activate a virtual environment:

Generally we recommend using `uv` to manage virtual environments and install packages.

```bash
uv venv --python=3.12 --managed-python
source .venv/bin/activate
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

### Optional Dependencies
For advanced features, uncomment the relevant lines in `requirements.txt` and reinstall:
- `tiktoken` - Enhanced tokenization
- `markitdown` - MarkItDown document processor
- `docling` - Docling document processor
- `marker-pdf` - Marker PDF processor
- `transformers` - Qwen reranker support

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

## Process Flow

The document extraction and retrieval process follows these steps:

```
1. Input Validation & Parsing
   ├── Parse JSON chat_history
   ├── Validate file upload
   └── Save file to temporary location

2. Document Processing
   ├── Extract text using configured processor (PyPDF2, MarkItDown, etc.)
   ├── Validate extracted text is not empty
   └── Optional: Clear previous retrieval chunks if enabled

3. Chunking Strategy Decision
   ├── If max_tokens > 0 and document fits within limit
   │   └── Use entire document as single chunk
   └── Else: Split document into semantic chunks
       ├── Split by sentences (maintaining coherence)
       ├── Apply configured chunk_size
       └── Add overlap between chunks (default: 50 words)

4. Query Extraction
   ├── Find last user message in chat_history
   └── Use as search query for retrieval

5. Chunk Retrieval
   ├── Initial retrieval using configured method:
   │   ├── BM25 (keyword-based)
   │   ├── SentenceTransformers (embedding-based)
   │   └── InstructSentenceTransformers (instruction-guided)
   ├── Retrieve top_k_retrieval candidates (default: 10)
   └── Optional: Rerank using configured reranker:
       ├── CrossEncoder reranker
       └── Qwen decoder-based reranker

6. Response Preparation
   ├── Format selected chunks with numbering
   ├── Insert as new user message in chat_history
   ├── Prepare document metadata
   └── Return enhanced chat_history with chunks
```

### Key Features

- **Hybrid Processing**: Automatically uses full document if within token limit, otherwise chunks
- **Semantic Chunking**: Splits by sentences to maintain context coherence
- **Multiple Retrieval Methods**: BM25, sentence transformers, or instruction-based
- **Optional Reranking**: Cross-encoder or Qwen-based reranking for better relevance
- **History Management**: Optional clearing of previous retrieval chunks to prevent context overflow

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
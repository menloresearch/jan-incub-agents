# Document Chunk Service POC

A FastAPI-based service for document chunking and retrieval without LLM generation. This service extracts text from documents, splits it into chunks, and retrieves relevant chunks based on user queries.

## Features

- **Document Processing**: Supports PDF and text files using multiple processors (PyPDF2, MarkItDown, Docling, Marker)
- **Text Chunking**: Intelligent text splitting with configurable chunk sizes and overlap
- **Retrieval Methods**: BM25, Sentence Transformers, and Instruct-based embeddings
- **Reranking**: Optional reranking using Cross-Encoders or Qwen Reranker
- **FastAPI Integration**: RESTful API with automatic documentation
- **Configuration-driven**: YAML-based configuration system

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For optional processors, install additional dependencies:
```bash
# For MarkItDown processor
pip install markitdown[pdf]

# For Docling processor  
pip install docling

# For Marker processor
pip install marker-pdf
```

## Usage

### Starting the Server

Run the server with a configuration file:

```bash
python main.py --config ../attachments-agent/config/bge_m3_400.yaml
```

Options:
- `--config`: Path to configuration YAML file (required)
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)
- `--reload`: Enable auto-reload for development

### API Endpoints

Once running, the API will be available at `http://localhost:8000` with the following endpoints:

#### 1. Upload and Process Document
`POST /chunk-upload`

Upload a document file and retrieve relevant chunks based on chat history.

**Form Data:**
- `file`: Document file (PDF or TXT)
- `chat_history`: JSON string of chat history
- `max_tokens`: Optional maximum tokens

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
      "role": "system", 
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

### Configuration Options

- `chunk_size`: Size of text chunks (in words)
- `retrieval_method`: "bm25", "sentence-transformers", or "instruct-sentence-transformers"
- `retrieval_kwargs`: Parameters for the retrieval method
- `reranker_method`: "cross-encoder", "qwen-reranker", or null
- `reranker_kwargs`: Parameters for the reranker
- `processor`: "pypdf2", "markitdown", "docling", or "marker"
- `max_tokens`: Maximum tokens to consider
- `top_k_retrieval`: Number of chunks to retrieve initially (default: 10)
- `top_k_rerank`: Number of chunks to keep after reranking (default: 5)

## Example Usage

### Python Client Example

```python
import requests
import json

# Start the server first
# python main.py --config ../attachments-agent/config/bge_m3_400.yaml

# Upload and process a document
chat_history = [
    {"role": "user", "content": "What are the main topics in this document?"}
]

with open("document.pdf", "rb") as f:
    files = {"file": ("document.pdf", f, "application/pdf")}
    data = {"chat_history": json.dumps(chat_history)}
    
    response = requests.post("http://localhost:8000/chunk-upload", files=files, data=data)
    result = response.json()

print("Enhanced chat history:")
for message in result["chat_history"]:
    print(f"{message['role']}: {message['content'][:100]}...")
```

### curl Example

```bash
# Upload and process document
curl -X POST "http://localhost:8000/chunk-upload" \
     -F "file=@document.pdf" \
     -F 'chat_history=[{"role": "user", "content": "Summarize this document"}]'
```

## Output Format

The service returns an enhanced chat history where relevant document chunks are inserted as system messages before user queries. This allows downstream LLM services to have access to the relevant context without needing to implement their own retrieval logic.

## Architecture

The POC is built with the following components:

1. **ChunkService**: Core service handling document processing and retrieval
2. **FastAPI App**: REST API layer with request/response models
3. **Configuration System**: YAML-based configuration compatible with existing configs
4. **Document Processors**: Multiple processors for different file types
5. **Retrieval Methods**: Various retrieval algorithms (BM25, embeddings)
6. **Reranking**: Optional reranking for improved relevance

## Development

For development with auto-reload:

```bash
python main.py --config ../attachments-agent/config/bge_m3_400.yaml --reload
```

The service will automatically restart when code changes are detected.

# Attachments Agent - Document Q&A System

RAG-based document Q&A system with evaluation suite. Tests 20+ configurations across retrieval methods, embeddings, and processors.

## Setup

```bash
cp .env.example .env  # Add your OPENAI_API_KEY
uv sync
```

## Usage

### Web Demo
```bash
uv run streamlit run web_demo.py
```

### Dataset & Evaluation
```bash
uv run python load_dataset.py                                    # Download MMLongBench-Doc dataset
uv run python retrieval_eval.py config/evaluation_suite.yaml    # Run evaluation (requires load_dataset.py first)
```

### Performance Analysis
```bash
uv run python map_time_tradeoff_plot.py --data output/combined_result.csv --config config/plot/config_mapping.yaml --output-dir ./plots
```

**CLI args:**
- `--data`: CSV with evaluation results
- `--config`: YAML mapping file for plot labels  
- `--output-dir`: Output directory for plots

## Architecture

**Core Components:**
- `processors.py`: Document-to-text (PyPDF2, MarkItDown, Docling, Native)
- `utils.py`: Retrieval (BM25, sentence-transformers) + reranking (cross-encoder)
- `document_agents.py`: RAG pipeline (SimpleRAGDocumentAgent)
- `retrieval_eval.py`: Evaluation framework

**FastAPI Service (POC):**
```bash
cd poc && export CHUNK_SERVICE_CONFIG="bge_m3_400.yaml" && uv run uvicorn main:app
```

## Evaluation

20 configurations tested:
- **Retrieval**: BM25, dense embeddings, instruction-tuned embeddings
- **Models**: MiniLM, Qwen, Gemma, BGE-M3
- **Chunk sizes**: 400/800/1200 tokens
- **Reranking**: MS-MARCO, Qwen cross-encoders
- **Processors**: PyPDF2, MarkItDown, Docling

**Metrics**: MAP@10, MRR, Success Rate, Processing Time

See [experiments/](experiments/) for detailed results and analysis.

## Config Example

```yaml
name: "Qwen + Cross-Encoder"
chunk_size: 400
retrieval_method: "sentence-transformers"
retrieval_kwargs:
  model_name: "Alibaba-NLP/gte-Qwen2-0.6B-instruct"
reranker_method: "cross-encoder"
reranker_kwargs:
  model_name: "cross-encoder/qwen2-0.6b-reranker"
processor_class: "PyPDF2Processor"
top_k_retrieval: 10
top_k_rerank: 5
```
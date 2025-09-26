from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Type, Optional
import yaml
import sys
import os

from .processors import (
    ProcessorBase,
    PyPDF2Processor,
    MarkItDownProcessor,
    DoclingProcessor,
    MarkerProcessor,
)
from .utils import split_into_multi_chunks, load_retrieval, load_reranker, Tokenizer


class ChunkService:
    """Service for document chunking and retrieval without LLM generation."""

    def __init__(
        self,
        chunk_size: int,
        retrieval_method: str = "bm25",
        retrieval_kwargs: Dict = None,
        reranker_method: Optional[str] = None,
        reranker_kwargs: Dict = None,
        processor_class: Type[ProcessorBase] = None,
        processor_kwargs: Dict = None,
        top_k_retrieval: int = 10,
        top_k_rerank: int = 10,
        **kwargs,
    ):
        self.chunk_size = chunk_size
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank

        # Initialize document processor
        if processor_class:
            self.document_processor = processor_class(**(processor_kwargs or {}))
        else:
            # Default to PyPDF2 processor
            self.document_processor = PyPDF2Processor()

        default_model_id = "gpt-4o"
        self.tokenizer = Tokenizer(model_id=default_model_id)

        # Initialize retrieval and reranking
        self.retrieval = load_retrieval(retrieval_method, **(retrieval_kwargs or {}))
        self.reranker = (
            load_reranker(reranker_method, **(reranker_kwargs or {}))
            if reranker_method
            else None
        )

    def process_document_and_retrieve_chunks(
        self,
        file_path: Optional[str] = None,
        document_text: Optional[str] = None,
        chat_history: List[Dict[str, str]] = None,
        max_tokens: Optional[int] = -1,
    ) -> Dict[str, Any]:
        """
        Process document and retrieve relevant chunks based on chat history.

        Args:
            file_path: Path to the document file (optional if document_text provided)
            document_text: Document text directly (optional if file_path provided)
            chat_history: List of chat messages with 'role' and 'content' keys
            max_tokens: Maximum tokens to consider (optional override)

        Returns:
            Dictionary containing chat_history with retrieved chunks inserted as context
            and selected_chunks for direct access
        """
        try:
            # Validate inputs
            if not file_path and not document_text:
                raise ValueError("Either file_path or document_text must be provided")

            if not chat_history:
                chat_history = []

            # Get document text
            if document_text:
                # Use provided document text directly
                pass  # document_text is already set
            else:
                # Process document from file path
                document_file = Path(file_path)
                if not document_file.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                document_text = self.document_processor.file_to_text(document_file)

            if not document_text.strip():
                return {
                    "error": "Could not extract any text from the document.",
                    "chat_history": chat_history,
                }

            # Hybrid logic: Check if document is small enough to use without chunking
            should_use_full_document = False
            if max_tokens > 0 and self.tokenizer:
                try:
                    # Tokenize the document to check its length
                    document_tokens = self.tokenizer.tokenize(document_text)
                    num_tokens = len(document_tokens)

                    if num_tokens <= max_tokens:
                        should_use_full_document = True
                        print(
                            f"Document has {num_tokens} tokens (<= {max_tokens}), using full document without chunking"
                        )
                    else:
                        print(
                            f"Document has {num_tokens} tokens (> {max_tokens}), using chunking"
                        )
                except Exception as e:
                    print(f"Warning: Could not tokenize document for hybrid logic: {e}")
                    # Fall back to chunking
                    should_use_full_document = False

            if should_use_full_document:
                # Use the entire document as a single "chunk"
                selected_chunks = [document_text]
                chunks_text = [document_text]  # For consistency in return values
            else:
                # Split into chunks
                chunks_text = split_into_multi_chunks(
                    document_text, chunk_size=self.chunk_size
                )

                if not chunks_text:
                    return {
                        "error": "Could not create any chunks from the document.",
                        "chat_history": chat_history,
                    }

                # Get query from last message in chat history for retrieval
                if not chat_history:
                    return {
                        "error": "No query provided in chat history.",
                        "chat_history": chat_history,
                    }

                # Find the last user message as the query
                query = None
                for message in reversed(chat_history):
                    if message.get("role") == "user":
                        query = message.get("content", "")
                        break

                if not query:
                    return {
                        "error": "No user message found in chat history.",
                        "chat_history": chat_history,
                    }

                # Use provided parameters or defaults
                retrieval_k = self.top_k_retrieval
                rerank_k = self.top_k_rerank

                # Retrieve relevant chunks
                candidates = self.retrieval.retrieve(
                    query=query, documents=chunks_text, top_k=retrieval_k
                )

                # Rerank if reranker is available
                if self.reranker:
                    selected_chunks = self.reranker.rerank(
                        query=query,
                        documents=candidates,
                        top_k=min(rerank_k, len(candidates)),
                    )
                else:
                    selected_chunks = candidates[:rerank_k]

            # Prepare context from selected chunks
            retrieved_context = "\n\n".join(
                [f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(selected_chunks)]
            )

            # Insert context into chat history
            # We'll add the retrieved context as the latest user message
            enhanced_chat_history = list(chat_history)  # Copy existing chat history

            # Add retrieved context as the latest user message
            context_message = {
                "role": "user",
                "content": f"Retrieved relevant information from the document:\n\n{retrieved_context}",
            }
            enhanced_chat_history.append(context_message)

            # Prepare document info
            if file_path:
                document_file = Path(file_path)
                document_info = {
                    "file_path": str(document_file),
                    "file_size": document_file.stat().st_size,
                    "text_length": len(document_text),
                    "chunks_created": len(chunks_text),
                }
            else:
                document_info = {
                    "file_path": None,
                    "file_size": None,
                    "text_length": len(document_text),
                    "chunks_created": len(chunks_text),
                }

            return {
                "chat_history": enhanced_chat_history,
                "selected_chunks": selected_chunks,  # Add selected chunks for direct access
                "chunks_count": len(chunks_text),
                "retrieved_chunks_count": len(selected_chunks),
                "document_info": document_info,
            }

        except Exception as e:
            return {"error": str(e), "chat_history": chat_history}

    def cleanup_models(self):
        """Clean up models and free memory."""
        import torch
        import gc

        # Move models to CPU first
        if hasattr(self.retrieval, "model") and hasattr(self.retrieval.model, "to"):
            try:
                self.retrieval.model.to("cpu")
            except Exception as e:
                print(f"Warning: Could not move retrieval model to CPU: {e}")

        # Move reranker model to CPU if it exists
        if (
            self.reranker
            and hasattr(self.reranker, "model")
            and hasattr(self.reranker.model, "to")
        ):
            try:
                self.reranker.model.to("cpu")
            except Exception as e:
                print(f"Warning: Could not move reranker model to CPU: {e}")

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def create_chunk_service_from_config(config_path: str) -> ChunkService:
    """Create ChunkService from configuration file."""
    config = load_config(config_path)

    # Map processor names to classes
    processor_map = {
        "pypdf2": PyPDF2Processor,
        "markitdown": MarkItDownProcessor,
        "docling": DoclingProcessor,
        "marker": MarkerProcessor,
    }

    # Get processor class from config (default to PyPDF2)
    processor_name = config.get("processor", "pypdf2")
    processor_class = processor_map.get(processor_name, PyPDF2Processor)

    return ChunkService(
        chunk_size=config.get("chunk_size", 400),
        retrieval_method=config.get("retrieval_method", "bm25"),
        retrieval_kwargs=config.get("retrieval_kwargs", {}),
        reranker_method=config.get("reranker_method"),
        reranker_kwargs=config.get("reranker_kwargs", {}),
        processor_class=processor_class,
        processor_kwargs=config.get("processor_kwargs", {}),
        top_k_retrieval=config.get("top_k_retrieval", 10),
        top_k_rerank=config.get("top_k_rerank", 5),
    )

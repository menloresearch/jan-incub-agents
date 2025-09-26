from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Type, Optional
import sys
import os

# Add poc directory to path to import ChunkService
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "poc"))

from processors import ProcessorBase
from llm_wrapper import OpenAIApiWrapper
from utils import Tokenizer
from chunk_service import ChunkService


class DocumentAgentBase(ABC):
    """Base class for document processing agents."""

    def __init__(
        self,
        model_id: str,
        base_url_llm: str = "https://api.openai.com/v1",
        processor_class: Type[ProcessorBase] = None,
        processor_kwargs: Dict = None,
        system_prompt: str = "",
        **kwargs,
    ):
        self.llm = OpenAIApiWrapper(
            model_id=model_id,
            base_url=base_url_llm,
            sampling_params=kwargs.get("sampling_params", {}),
        )

        if processor_class:
            self.document_processor = processor_class(**(processor_kwargs or {}))
        else:
            self.document_processor = None

        self.system_prompt = system_prompt

    @abstractmethod
    def chat_with_document(
        self,
        chat_history: List[Dict],
        document_file: Path,
    ) -> str:
        """Chat with document using the specific agent implementation."""
        pass


def generate_from_document_text(
    document_text: str,
    chat_history: List[Dict],
    system_prompt: str,
    llm: OpenAIApiWrapper,
) -> str:
    """Generate response using document text directly."""

    # Create context with document
    context_prompt = f"{system_prompt}\n\nDocument content:\n{document_text}"

    # Prepare messages for the LLM
    messages = [{"role": "system", "content": context_prompt}]
    messages.extend(chat_history)

    return llm.generate(messages)


def generate_from_rag(
    chunks_text: List[str],
    chat_history: List[Dict],
    system_prompt: str,
    llm: OpenAIApiWrapper,
) -> str:
    """Generate response using RAG-retrieved chunks."""

    # Combine retrieved chunks
    retrieved_context = "\n\n".join(
        [f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks_text)]
    )

    # Create context with retrieved chunks
    context_prompt = (
        f"{system_prompt}\n\nRetrieved relevant information:\n{retrieved_context}"
    )

    # Prepare messages for the LLM
    messages = [{"role": "system", "content": context_prompt}]
    messages.extend(chat_history)

    return llm.generate(messages)


class SimpleRAGDocumentAgent(DocumentAgentBase):
    """RAG-based document agent with retrieval and reranking - MIGRATED to use ChunkService."""

    def __init__(
        self,
        chunk_size: int,
        retrieval_method: str = "bm25",
        retrieval_kwargs: Dict = None,
        reranker_method: Optional[str] = None,
        reranker_kwargs: Dict = None,
        **kwargs,
    ):
        self.chunk_size = chunk_size

        # Call parent constructor first
        super().__init__(**kwargs)

        # Create ChunkService with the same configuration
        # We need to extract top_k values from retrieval_kwargs and reranker_kwargs
        retrieval_kwargs = retrieval_kwargs or {}
        reranker_kwargs = reranker_kwargs or {}

        # Extract top_k values, with backward compatibility
        top_k_retrieval = retrieval_kwargs.pop(
            "top_k", 10
        )  # Remove from kwargs to avoid conflicts
        top_k_rerank = reranker_kwargs.pop(
            "top_k", 10
        )  # Remove from kwargs to avoid conflicts

        self._chunk_service = ChunkService(
            chunk_size=chunk_size,
            retrieval_method=retrieval_method,
            retrieval_kwargs=retrieval_kwargs,
            reranker_method=reranker_method,
            reranker_kwargs=reranker_kwargs,
            processor_class=type(self.document_processor),
            processor_kwargs={},
            top_k_retrieval=top_k_retrieval,
            top_k_rerank=top_k_rerank,
        )

        # Expose retrieval and reranker for backward compatibility
        # This ensures existing code that accesses agent.retrieval and agent.reranker still works
        self.retrieval = self._chunk_service.retrieval
        self.reranker = self._chunk_service.reranker

    def chat_with_document(
        self,
        chat_history: List[Dict],
        document_file: Path,
        max_tokens: Optional[int] = -1,
    ) -> str:
        """Chat with document using RAG pipeline - MIGRATED to use ChunkService."""

        # Use ChunkService to process document and retrieve chunks
        result = self._chunk_service.process_document_and_retrieve_chunks(
            file_path=str(document_file),
            chat_history=chat_history,
            max_tokens=max_tokens,
        )

        # Handle errors
        if "error" in result:
            return f"Error: {result['error']}"

        # Extract the retrieved chunks directly from the result
        # The ChunkService now returns selected_chunks directly
        enhanced_chat_history = result["chat_history"]

        # Get chunks directly from the result (more efficient)
        retrieved_chunks = result.get("selected_chunks", [])

        # If no chunks were found, return error
        if not retrieved_chunks:
            return "Error: Could not extract any text from the document."

        original_chat_history = chat_history  # Keep original for generation

        # Generate answer with retrieved context using the same function as before
        return generate_from_rag(
            chunks_text=retrieved_chunks,
            chat_history=original_chat_history,
            system_prompt=self.system_prompt,
            llm=self.llm,
        )

    def move_models_to_cpu(self):
        """Move all models from GPU to CPU to free GPU memory."""
        # Delegate to ChunkService
        if hasattr(self._chunk_service, "cleanup_models"):
            # Use the move_models_to_cpu equivalent from ChunkService
            import torch

            # Move retrieval model to CPU if it has a model attribute
            if hasattr(self.retrieval, "model") and hasattr(self.retrieval.model, "to"):
                try:
                    self.retrieval.model.to("cpu")
                except Exception as e:
                    print(f"Warning: Could not move retrieval model to CPU: {e}")

            # Move reranker model to CPU if it exists and has a model attribute
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

    def cleanup_models(self):
        """Clean up models and free memory."""
        # Delegate to ChunkService
        self._chunk_service.cleanup_models()

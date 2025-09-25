from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Type, Optional
from processors import ProcessorBase
from llm_wrapper import OpenAIApiWrapper
from utils import Tokenizer, split_into_multi_chunks, load_retrieval, load_reranker


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
            sampling_params=kwargs.get("sampling_params", {})
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
    llm: OpenAIApiWrapper
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
    retrieved_context = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks_text)])
    
    # Create context with retrieved chunks
    context_prompt = f"{system_prompt}\n\nRetrieved relevant information:\n{retrieved_context}"
    
    # Prepare messages for the LLM
    messages = [{"role": "system", "content": context_prompt}]
    messages.extend(chat_history)
    
    return llm.generate(messages)


class SimpleRAGDocumentAgent(DocumentAgentBase):
    """RAG-based document agent with retrieval and reranking."""
    
    def __init__(
        self,
        chunk_size: int,
        retrieval_method: str = "bm25",
        retrieval_kwargs: Dict = None,
        reranker_method: Optional[str] = None,
        reranker_kwargs: Dict = None,
        **kwargs,
    ):
        self.llm_tokenizer = Tokenizer(model_id=kwargs['model_id'])
        self.chunk_size = chunk_size
        
        # Initialize retrieval and reranking
        self.retrieval = load_retrieval(retrieval_method, **(retrieval_kwargs or {}))
        self.reranker = (
            load_reranker(reranker_method, **(reranker_kwargs or {}))
            if reranker_method else None
        )
        
        super().__init__(**kwargs)
    
    def chat_with_document(
        self,
        chat_history: List[Dict],
        document_file: Path,
    ) -> str:
        """Chat with document using RAG pipeline."""
        
        # Process document to text
        document_text = self.document_processor.file_to_text(document_file)
        
        # Split into chunks
        chunks_text = split_into_multi_chunks(document_text, chunk_size=self.chunk_size)
        
        if not chunks_text:
            return "Error: Could not extract any text from the document."
        
        # Get query from last message
        if not chat_history:
            return "Error: No query provided in chat history."
        
        query = chat_history[-1]["content"]
        
        # Retrieve relevant chunks
        candidates = self.retrieval.retrieve(
            query=query,
            documents=chunks_text,
        )
        
        # Rerank if reranker is available
        if self.reranker:
            selected_chunks = self.reranker.rerank(query=query, documents=candidates)
        else:
            selected_chunks = candidates
        
        # Generate answer with retrieved context
        return generate_from_rag(
            chunks_text=selected_chunks,
            chat_history=chat_history,
            system_prompt=self.system_prompt,
            llm=self.llm,
        )
    
    def move_models_to_cpu(self):
        """Move all models from GPU to CPU to free GPU memory."""
        import torch
        
        # Move retrieval model to CPU if it has a model attribute
        if hasattr(self.retrieval, 'model') and hasattr(self.retrieval.model, 'to'):
            try:
                self.retrieval.model.to('cpu')
            except Exception as e:
                print(f"Warning: Could not move retrieval model to CPU: {e}")
        
        # Move reranker model to CPU if it exists and has a model attribute
        if self.reranker and hasattr(self.reranker, 'model') and hasattr(self.reranker.model, 'to'):
            try:
                self.reranker.model.to('cpu')
            except Exception as e:
                print(f"Warning: Could not move reranker model to CPU: {e}")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def cleanup_models(self):
        """Clean up models and free memory."""
        import torch
        import gc
        
        # Move models to CPU first
        self.move_models_to_cpu()
        
        # Delete model references if possible
        try:
            if hasattr(self.retrieval, 'model'):
                del self.retrieval.model
        except Exception:
            pass
            
        try:
            if self.reranker and hasattr(self.reranker, 'model'):
                del self.reranker.model
        except Exception:
            pass
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
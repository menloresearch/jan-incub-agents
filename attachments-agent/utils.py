import re
from typing import List, Dict, Any, Optional
import math
from abc import ABC, abstractmethod


def split_into_multi_chunks(text: str, chunk_size: int, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks of specified size."""
    
    if not text.strip():
        return []
    
    # Split by sentences first to maintain semantic coherence
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        # If adding this sentence would exceed chunk size, finalize current chunk
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap from previous chunk
            if overlap > 0 and chunks:
                overlap_words = current_chunk.split()[-overlap:]
                current_chunk = " ".join(overlap_words) + " " + sentence
                current_length = len(overlap_words) + sentence_length
            else:
                current_chunk = sentence
                current_length = sentence_length
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_length += sentence_length
    
    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


class RetrievalBase(ABC):
    """Base class for retrieval methods."""
    
    @abstractmethod
    def retrieve(self, query: str, documents: List[str], top_k: int = 10) -> List[str]:
        """Retrieve relevant documents for the query."""
        pass


class BM25Retrieval(RetrievalBase):
    """Simple BM25-based retrieval implementation."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.idf_cache = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _compute_idf(self, term: str, documents: List[str]) -> float:
        """Compute IDF for a term."""
        if term in self.idf_cache:
            return self.idf_cache[term]
        
        df = sum(1 for doc in documents if term in self._tokenize(doc))
        idf = math.log((len(documents) - df + 0.5) / (df + 0.5) + 1)
        self.idf_cache[term] = idf
        return idf
    
    def _compute_bm25_score(self, query_terms: List[str], document: str, documents: List[str], avg_doc_len: float) -> float:
        """Compute BM25 score for a document."""
        doc_terms = self._tokenize(document)
        doc_len = len(doc_terms)
        score = 0.0
        
        for term in query_terms:
            tf = doc_terms.count(term)
            if tf > 0:
                idf = self._compute_idf(term, documents)
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / avg_doc_len))
        
        return score
    
    def retrieve(self, query: str, documents: List[str], top_k: int = 10) -> List[str]:
        """Retrieve documents using BM25."""
        if not documents:
            return []
        
        query_terms = self._tokenize(query)
        if not query_terms:
            return documents[:top_k]
        
        # Compute average document length
        avg_doc_len = sum(len(self._tokenize(doc)) for doc in documents) / len(documents)
        
        # Score all documents
        scores = []
        for i, doc in enumerate(documents):
            score = self._compute_bm25_score(query_terms, doc, documents, avg_doc_len)
            scores.append((score, i, doc))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, _, doc in scores[:top_k]]


class SentenceTransformerRetrieval(RetrievalBase):
    """Retrieval using sentence transformers for embedding-based similarity."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            self.model = SentenceTransformer(model_name)
            self.np = np
            self.cosine_similarity = cosine_similarity
        except ImportError:
            raise ImportError("sentence-transformers and scikit-learn required for SentenceTransformerRetrieval")
    
    def retrieve(self, query: str, documents: List[str], top_k: int = 10) -> List[str]:
        """Retrieve documents using sentence transformer embeddings."""
        if not documents:
            return []
        
        # Encode query and documents
        query_embedding = self.model.encode([query])
        doc_embeddings = self.model.encode(documents)
        
        # Compute similarities
        similarities = self.cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Get top-k most similar documents
        top_indices = self.np.argsort(similarities)[::-1][:top_k]
        
        return [documents[i] for i in top_indices]


class InstructSentenceTransformerRetrieval(RetrievalBase):
    """Retrieval using sentence transformers with instruction-based embedding."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", instruction: str = "Given a web search query, retrieve relevant passages that answer the query"):
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            self.model = SentenceTransformer(model_name)
            self.instruction = instruction
            self.np = np
            self.cosine_similarity = cosine_similarity
        except ImportError:
            raise ImportError("sentence-transformers and scikit-learn required for InstructSentenceTransformerRetrieval")
    
    def _format_instruction(self, query: str) -> str:
        """Format query with instruction according to Qwen embedding format."""
        return f"Instruct: {self.instruction}\nQuery:{query}"
    
    def retrieve(self, query: str, documents: List[str], top_k: int = 10) -> List[str]:
        """Retrieve documents using instruction-based sentence transformer embeddings."""
        if not documents:
            return []
        
        # Format query with instruction
        instructed_query = self._format_instruction(query)
        
        # Encode query and documents
        query_embedding = self.model.encode([instructed_query])
        doc_embeddings = self.model.encode(documents)
        
        # Compute similarities
        similarities = self.cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Get top-k most similar documents
        top_indices = self.np.argsort(similarities)[::-1][:top_k]
        
        return [documents[i] for i in top_indices]


class RerankerBase(ABC):
    """Base class for reranking methods."""
    
    @abstractmethod
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        """Rerank documents based on query relevance."""
        pass


class CrossEncoderReranker(RerankerBase):
    """Cross-encoder based reranking."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except ImportError:
            raise ImportError("sentence-transformers required for CrossEncoderReranker")
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        """Rerank documents using cross-encoder."""
        if not documents:
            return []
        
        if len(documents) <= top_k:
            return documents
        
        # Score query-document pairs
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        
        # Sort by score and return top-k
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in scored_docs[:top_k]]


class QwenReranker(RerankerBase):
    """Qwen decoder-based reranking."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-4B", instruction: str = "Determine if this document is relevant to the query."):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.model_name = model_name
            self.instruction = instruction
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto").eval()
            
            # Get token IDs for "Yes" and "No" - these are used for scoring
            self.true_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
            self.false_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
            
        except ImportError:
            raise ImportError("transformers and torch required for QwenReranker")
    
    def _format_instruction(self, query: str, doc: str) -> str:
        """Format query-document pair according to Qwen reranker format."""
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        """Rerank documents using Qwen decoder model."""
        if not documents:
            return []
        
        if len(documents) <= top_k:
            return documents
        
        import torch
        
        # Format all query-document pairs
        pairs = [self._format_instruction(query, doc) for doc in documents]
        
        scores = []
        for pair in pairs:
            # Tokenize input
            inputs = self.tokenizer(pair, return_tensors="pt", truncation=True, max_length=2048)
            
            with torch.no_grad():
                # Get model outputs
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]  # Get last token logits
                
                # Get probabilities for "Yes" and "No" tokens
                true_prob = torch.softmax(logits[[self.true_token_id, self.false_token_id]], dim=0)[0]
                scores.append(true_prob.cpu().item())
        
        # Sort by score and return top-k
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in scored_docs[:top_k]]


def load_retrieval(retrieval_method: str, **kwargs) -> RetrievalBase:
    """Load retrieval method by name."""
    
    if retrieval_method == "bm25":
        return BM25Retrieval(**kwargs)
    elif retrieval_method == "sentence-transformers":
        model_name = kwargs.get("model_name", "all-MiniLM-L6-v2")
        return SentenceTransformerRetrieval(model_name=model_name)
    elif retrieval_method == "instruct-sentence-transformers":
        model_name = kwargs.get("model_name", "Qwen/Qwen3-Embedding-0.6B")
        instruction = kwargs.get("instruction", "Given a web search query, retrieve relevant passages that answer the query")
        return InstructSentenceTransformerRetrieval(model_name=model_name, instruction=instruction)
    else:
        raise ValueError(f"Unknown retrieval method: {retrieval_method}")


def load_reranker(reranker_method: str, **kwargs) -> Optional[RerankerBase]:
    """Load reranker method by name."""
    
    if reranker_method is None:
        return None
    elif reranker_method == "cross-encoder":
        model_name = kwargs.get("model_name", "cross-encoder/ms-marco-MiniLM-L-2-v2")
        return CrossEncoderReranker(model_name=model_name)
    elif reranker_method == "qwen-reranker":
        model_name = kwargs.get("model_name", "Qwen/Qwen3-Reranker-4B")
        instruction = kwargs.get("instruction", "Determine if this document is relevant to the query.")
        return QwenReranker(model_name=model_name, instruction=instruction)
    else:
        raise ValueError(f"Unknown reranker method: {reranker_method}")


class Tokenizer:
    """Simple tokenizer wrapper for token counting and truncation."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        try:
            import tiktoken
            self.encoding = tiktoken.encoding_for_model(model_id)
        except ImportError:
            # Fallback to simple word-based tokenization
            self.encoding = None
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens."""
        if self.encoding:
            return self.encoding.encode(text)
        else:
            # Simple word-based tokenization as fallback
            return text.split()
    
    def detokenize(self, tokens: List) -> str:
        """Convert tokens back to text."""
        if self.encoding and isinstance(tokens[0], int):
            return self.encoding.decode(tokens)
        else:
            # Simple word-based detokenization as fallback
            return " ".join(str(token) for token in tokens)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenize(text))
# Jan Attachment Understanding Agent

Jan Attachment eval contains two main components: 1. Processor and 2. DocumentAgent

## Processors

### Pseudo Code
```python
class ProcessorBase:
    def __init__(self, ):
        # shared args

    def file_to_text(file: Path) -> str:
        pass

class AdvancedProcessor(ProcessorBase):

    def __init__(self, **kwargs):
        # Process additional args
        super().__init__(**kwargs)  

    def file_to_text(file: Path) -> str:
        # .... Load and process Document
```

### Available Processors

#### 1. Native Processor
Send unprocessed PDF into LLM and let the provider automatically handle it.
1. [OpenAI PDF](https://platform.openai.com/docs/guides/pdf-files?api-mode=chat)
2. [Gemini PDF](https://ai.google.dev/gemini-api/docs/document-processing)

We will use [OpenRouter's native pdf feature](https://openrouter.ai/docs/features/multimodal/pdfs) to standardize different providers' native pdf functionality.

#### 2. PyPDF2 Processor
Use [PyPDF2](https://pypdf2.readthedocs.io/en/3.x/) to process PDF
Supports only PDF.

#### 3. MarkItDown Processor
Use [MarkItDown](https://github.com/microsoft/markitdown) to process PDF
Supports PDF, Word, HTML, Excel, and more.

#### 4. Docling Processor
Use [Docling](https://github.com/docling-project/docling) to process PDF
Supports PDF, Word, HTML, Excel, and more.


## Document Agent Adapter

Document Agent Adapter will be used to encapsulate all kinds of Document processing method into same multi-turn conversation agent format.

### Pseudo code

```python
class DocumentAgentBase:
    def __init__(
        self,
        model_id: str,
        base_url_llm: str = "https://api.openai.com/v1",
        processor_class: Type[ProcessorBase],
        processor_kwargs: Dict = {},
        system_prompt: str,
        **kwargs,
    ):
        self.llm = OpenAIApiWrapper(
            model_id=model_id, 
            base_url=base_url,
            sampling_params=kwargs.get("sampling_params", {})
        ) # can be point to VLLM, OpenRouter, or OpenAI
        self.document_processor = processor_class(**processor_kwargs)
        self.system_prompt = system_prompt
        
    
    def chat_with_document(
        chat_history: List[Dict],
        document_file: File,
    ) -> str:
        pass

```

### Available Agents

#### 1. SimpleDocumentAgent
We simply pass entire Document into LLM and truncate to specified context length

**pseudo code:**

```python
class SimpleDocumentAgent(DocumentAgentBase):
    
    def __init__(
        self,
        max_context_length: int,
        **kwargs,
    ):
        self.max_context_length = max_context_length
        self.llm_tokenizer = Tokenizer(model_id=kwargs['model_id'])
        super().__init__(
            **kwargs
        )
    
    def chat_with_document(
        chat_history: List[Dict],
        document_file: File,
    ) -> str:
        document_text = self.document_processor.file_to_text(document_file)
        document_text_tokenized = self.llm_tokenizer.tokenize(document_text)
        document_text_trimmed = self.llm_tokenizer.detokenize(document_text_tokenized[:max_context_length])

        return generate_from_document_text(
            document_text=document_text_trimmed, 
            chat_history=chat_history, 
            system_prompt=self.system_prompt,
            llm=self.llm
        )
```

#### 2. SimpleRAGDocumentAgent

We handle very large document with RAG pipeline.

**pseudo code:**

```python
class SimpleRAGDocumentAgent(DocumentAgentBase):
    
    def __init__(
        self,
        chunk_size: int,
        retrieval_method: str = "bm25",  # e.g. "bm25", "sentence-transformers", etc.
        retrieval_kwargs: Dict = {},
        reranker_method: Optional[str] = None,  # e.g. "cross-encoder"
        reranker_kwargs: Dict = {},
        **kwargs,
    ):
        self.llm_tokenizer = Tokenizer(model_id=kwargs['model_id'])

        self.chunk_size = chunk_size

        self.retrieval = load_retrieval(retrieval_method, **retrieval_kwargs)
        self.reranker = (
            load_reranker(reranker_method, **reranker_kwargs)
            if reranker_method else None
        )

        super().__init__(**kwargs)


    def chat_with_document(
        self,
        chat_history: List[Dict],
        document_file: File,
    ) -> str:
        document_text = self.document_processor.file_to_text(document_file)
        chunks_text = split_into_multi_chunks(document_text, chunk_size=self.chunk_size)


        query = chat_history[-1]["content"]
        candidates = self.retrieval.retrieve(
            query=query,
            documents=chunks_text,
        )

        if self.reranker:
            reranked = self.reranker.rerank(query=query, documents=candidates)
            selected_chunks = reranked
        else:
            selected_chunks = candidates

        # 4) generate answer with retrieved context
        return generate_from_rag(
            chunks_text=selected_chunks,
            chat_history=chat_history,
            system_prompt=self.system_prompt,
            llm=self.llm,
        )
```


#### 3. MCP_DocumentAgent

A single class that can operate in two MCP modes selected via the `mode` parameter:

* `RAG MCP` – classic RAG flow. LLM can calls `search` which returns ranked chunks included directly in answer generation.
* `Search MCP` – Jan-Nano style search-augmented flow. LLM should call `search` to obtain `(chunk_id, preview)` pairs and then `visit` on chosen `chunk_id`s to fetch full chunks.

****pseudo code:****

```python
def init_rag_mcp_server(...):
    """
    Register the document-specific RAG tools with a FastMCP server.

    Tools:
        1. `search(query: str) -> str`                   # retrieve relevant chunks
        2. `index_doc(doc_text: str) -> str`             # returns a temporary document ID
        3. `clear_index() -> bool`                       # reset index after use

    Need to allow LLM to only use `search`.
    """
    pass

def init_mcp_search_server(..., preview_chunk_size: int):
    """
    Register the search-only tools with a FastMCP server.

    Tools:
        (**Modified) 1. `search(query: str) -> List[Dict[str, str]]`   # returns [{"chunk_id": str, "preview": str}, ...]
        (**New)      2. `visit(chunk_id: str) -> str`                  # returns full chunk text
        3. `index_doc(doc_text: str) -> str`                           # returns a temporary document ID
        4. `clear_index() -> bool`                                     # reset index after use

    Need to allow LLM to only use `search` and 'visit'.
    """
    pass

@contextmanager
def indexed_document(mcp_server_url: str, document_text: str):
    """
    Context manager that temporarily indexes a document on the FastMCP RAG server
    and guarantees cleanup afterwards.

    Args:
        mcp_server_url (str): URL or IP address of the FastMCP server.
        document_text (str): Raw text of the document to index.
    """
    doc_id = index_doc(document_text)
    try:
        # Inside the `with` block the LLM can call `search(query)` which
        # implicitly scopes the query to this doc_id
        yield
    finally:
        clear_index()

class MCP_RAGDocumentAgent(DocumentAgentBase):
    def __init__(
            self, 
            chunk_size: int,
            retrieval_method: str = "bm25",  # e.g. "bm25", "sentence-transformers", etc.
            retrieval_kwargs: Dict = {},
            reranker_method: Optional[str] = None,  # e.g. "cross-encoder"
            reranker_kwargs: Dict = {},
            mcp_server_ip: str,
            allowed_tools: list = ['search'], # ['search'] for RAG MCP, ['search', 'visit'] for Search MCP
            **kwargs
        ):
        self.mcp_server_ip = mcp_server_ip
        mcp_servers = [
            {
                "type": "http",
                "server_url": self.mcp_server_ip,
                "allowed_tools": allowed_tools,
            }
        ]

        self.agent = Agent(
            base_url=kwargs.get("base_url_llm", "https://api.openai.com/v1"),
            api_key=os.environ.get("OPENAI_API_KEY),
            model=kwargs["model_id"],
            system_prompt=kwargs["system_prompt"],
            mcp_servers=mcp_servers,
            sampling_params=kwargs.get("sampling_params", {}),
        )
        super().__init__(**kwargs)

    async def chat_with_document(
        self,
        chat_history: List[Dict],
        document_file: File,
    ) -> str:
        doc_text = self.document_processor.file_to_text(document_file)
        history = [{"role": "system", "content": system_msg}] + chat_history
        
        # Use a context manager so the temporary index is removed automatically
        with indexed_document(self.mcp_server_ip, doc_text):
            async with self.agent as agent:
                _, answer = await agent.generate_response(
                    prompt=history[-1]["content"],
                    tool_use=True,
                    history=history[:-1],
                )
            
        return answer
```

#### 4. HybridDocumentAgent

If text of pdf is shorter than `self.max_context_length` we use `SimpleDocumentAgent` otherwise fallback to long context agent ex. `RAGAgent`

**pseudo code:**

```python
class HybridDocumentAgent(DocumentAgentBase):
    
    def __init__(
        self,
        max_context_length: int,
        retrieval_agent_class: Type[DocumentAgentBase] = SimpleRAGDocumentAgent,
        retrieval_agent_kwargs: Dict = {},
        **kwargs,
    ):
        self.max_context_length = max_context_length
        self.llm_tokenizer = Tokenizer(model_id=model_id)
        
        self.simple_agent = SimpleDocumentAgent(
            max_context_length=max_context_length,
            **kwargs,
        )
        self.retrieval_agent = load_retrieval_agent(retrieval_agent_class, **retrieval_agent_kwargs)
        
        super().__init__(**kwargs)
    
    def chat_with_document(
        self,
        chat_history: List[Dict],
        document_file: File,
    ) -> str:
        # Check document length first
        document_text = self.document_processor.file_to_text(document_file)
        document_tokens = self.llm_tokenizer.tokenize(document_text)
        num_tokens = len(document_tokens)
        
        # Route to appropriate strategy based on length
        if num_tokens <= self.max_context_length:
            # Use simple approach for short documents
            return self.simple_agent.chat_with_document(
                chat_history=chat_history,
                document_file=document_file,
            )
        else:
            return self.retrieval_agent.chat_with_document(
                chat_history=chat_history,
                document_file=document_file,
            )
```

## Usage Examples

### SimpleDocumentAgent

```python
simple_agent = SimpleDocumentAgent(
    model_id="gpt-4o",
    processor_class=SimpleProcessor,  # converts non-PDF text files directly
    processor_kwargs={},
    system_prompt="You are a concise summarizer.",
    max_context_length=8000,
)

response = simple_agent.chat_with_document(
    chat_history=[{"role": "user", "content": "Give me a summary."}],
    document_file=Path("short_report.txt"),
)
```

### SimpleRAGDocumentAgent

```python
rag_agent = SimpleRAGDocumentAgent(
    model_id="gpt-4",
    processor_class=Py2PDFProcessor,  # convert pdf pages to text
    processor_kwargs={},
    system_prompt="You are a detailed analyst.",
    
    chunk_size=512,
    retrieval_method="sentence-transformers",
    retrieval_kwargs={"embedding_model_id": "text-embedding-ada-002", "top_k": 20},
    reranker_method="cross-encoder",
    reranker_kwargs={"model_id": "cross-encoder/ms-marco-MiniLM-L-2-v2", "top_k": 5},
)

response = rag_agent.chat_with_document(
    chat_history=[{"role": "user", "content": "List the key findings."}],
    document_file=Path("long_whitepaper.pdf"),
)
```

### MCP_DocumentAgent (RAG MCP mode)

```python
# 1) Spin up the Fast-MCP server in RAG mode
rag_server_ip = init_rag_mcp_server(
    retrieval_config={"method": "sentence-transformers"},
    retrieval_kwargs={"embedding_model_id": "text-embedding-ada-002", "top_k": 20},
    reranker_method="cross-encoder",
    reranker_kwargs={"model_id": "cross-encoder/ms-marco-MiniLM-L-2-v2", "top_k": 10},
)

# 2) Instantiate the agent pointing to that server
rag_mcp_agent = MCP_DocumentAgent(
    mcp_server_ip=rag_server_ip,
    allowed_tools=["search"],  # RAG mode only needs 'search'

    model_id="gpt-4",
    processor_class=Py2PDFProcessor,
    processor_kwargs={},
    system_prompt="You are a helpful research assistant. You can assess document info by search function.",
    
    chunk_size=512,
    retrieval_method="sentence-transformers",
    retrieval_kwargs={"embedding_model_id": "text-embedding-ada-002", "top_k": 20},
    reranker_method="cross-encoder",
    reranker_kwargs={"model_id": "cross-encoder/ms-marco-MiniLM-L-2-v2", "top_k": 10},
)

response = rag_mcp_agent.chat_with_document(
    chat_history=[{"role": "user", "content": "Summarise key contributions."}],
    document_file=Path("conference_paper.pdf"),
)
```

### MCP_DocumentAgent (Search MCP mode)

```python
# 1) Spin up the Fast-MCP server in Search mode (search + visit)
search_server_ip = init_mcp_search_server(
    preview_chunk_size=128,
    retrieval_config={"method": "sentence-transformers"},
    retrieval_kwargs={"embedding_model_id": "text-embedding-ada-002", "top_k": 40},
    reranker_method="cross-encoder",
    reranker_kwargs={"model_id": "cross-encoder/ms-marco-MiniLM-L-2-v2", "top_k": 20},
)

# 2) Instantiate the agent pointing to that server
search_mcp_agent = MCP_DocumentAgent(
    mcp_server_ip=search_server_ip,
    allowed_tools=["search", "visit"],

    model_id="gpt-4",
    processor_class=Py2PDFProcessor,
    processor_kwargs={},
    system_prompt="You are a search-augmented assistant. Use `search` for previews and `visit` for full chunks.",
    chunk_size=2048,
)

response = search_mcp_agent.chat_with_document(
    chat_history=[{"role": "user", "content": "Explain the dataset used."}],
    document_file=Path("dataset_description.pdf"),
)
```


### Basic HybridDocumentAgent with SimpleRAGDocumentAgent

```python
hybrid_agent = HybridDocumentAgent(
    model_id="gpt-4",
    processor_class=Py2PDFProcessor,
    processor_kwargs={},
    system_prompt="You are a helpful document assistant.",
    max_context_length=8000,
    
    retrieval_agent_class=SimpleRAGDocumentAgent,
    retrieval_agent_kwargs={
        "model_id": "gpt-4",
        "processor_class": Py2PDFProcessor,  # convert pdf pages to text
        "processor_kwargs": {},
        "system_prompt": "You are a detailed analyst.",
        "chunk_size": 512,
        "retrieval_method": "sentence-transformers",
        "retrieval_kwargs": {"embedding_model_id": "text-embedding-ada-002", "top_k": 20},
        "reranker_method": "cross-encoder",
        "reranker_kwargs": {"model_id": "cross-encoder/ms-marco-MiniLM-L-2-v2", "top_k": 5},
    }
)

response = hybrid_agent.chat_with_document(
    chat_history=[{"role": "user", "content": "What is this document about?"}],
    document_file=Path("research_paper.pdf")
)
```
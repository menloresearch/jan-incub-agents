from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(
        ..., pattern=r"^(user|assistant)$", description="Sender role for the message"
    )
    content: str = Field(..., min_length=1, description="Message content")


class AddMemoriesRequest(BaseModel):
    messages: List[Message]
    user_id: str = Field(..., min_length=1)
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryEvent(BaseModel):
    id: str
    memory: str
    event: str = Field(..., pattern=r"^(ADD|UPDATE|DELETE)$")


class ResultsWrapper(BaseModel):
    results: List[MemoryEvent]


class MemoryItem(BaseModel):
    id: str
    memory: str
    user_id: str
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class MemorySearchResult(MemoryItem):
    score: float


class SearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    app_id: Optional[str] = None
    run_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = 100


class SearchResponse(BaseModel):
    results: List[MemorySearchResult]


class ListRequest(BaseModel):
    filters: Dict[str, Any]
    page: int = 1
    page_size: int = 100


class PaginatedResponse(BaseModel):
    count: int
    next: Optional[str]
    previous: Optional[str]
    results: List[MemoryItem]


class UpdateMemoryRequest(BaseModel):
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DeleteResponse(BaseModel):
    message: str

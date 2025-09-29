from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.deps import verify_api_key
from app.models.schemas import (
    AddMemoriesRequest,
    DeleteResponse,
    ListRequest,
    MemoryItem,
    MemorySearchResult,
    PaginatedResponse,
    ResultsWrapper,
    SearchRequest,
    SearchResponse,
    UpdateMemoryRequest,
)
from app.services.memory_service import MemoryService


def get_memory_service(request: Request) -> MemoryService:
    service = getattr(request.app.state, "memory_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Memory service not configured",
        )
    return service


router_v1 = APIRouter(
    prefix="/memories", tags=["memories"], dependencies=[Depends(verify_api_key)]
)
router_v2 = APIRouter(
    prefix="/memories", tags=["memories"], dependencies=[Depends(verify_api_key)]
)


@router_v1.post("/", response_model=ResultsWrapper)
def add_memories(
    payload: AddMemoriesRequest, service: MemoryService = Depends(get_memory_service)
) -> ResultsWrapper:
    results = service.add_memories(
        messages=[message.model_dump() for message in payload.messages],
        user_id=payload.user_id,
        agent_id=payload.agent_id,
        run_id=payload.run_id,
        metadata=payload.metadata,
    )
    return ResultsWrapper(results=results)


@router_v1.get("/{memory_id}/", response_model=MemoryItem)
def get_memory(
    memory_id: str, service: MemoryService = Depends(get_memory_service)
) -> MemoryItem:
    record = service.get_memory(memory_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found"
        )
    return MemoryItem(**service.serialize(record))


@router_v1.put("/{memory_id}/", response_model=MemoryItem)
def update_memory(
    memory_id: str,
    payload: UpdateMemoryRequest,
    service: MemoryService = Depends(get_memory_service),
) -> MemoryItem:
    updated = service.update_memory(
        memory_id, text=payload.text, metadata=payload.metadata
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found"
        )
    return MemoryItem(**service.serialize(updated))


@router_v1.delete("/{memory_id}/", response_model=DeleteResponse)
def delete_memory(
    memory_id: str, service: MemoryService = Depends(get_memory_service)
) -> DeleteResponse:
    removed = service.delete_memory(memory_id)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found"
        )
    return DeleteResponse(message="Memory deleted successfully")


@router_v1.post("/search/", response_model=SearchResponse)
def search_memories_v1(
    payload: SearchRequest, service: MemoryService = Depends(get_memory_service)
) -> SearchResponse:
    filters = payload.filters or {}
    if payload.user_id:
        filters["user_id"] = payload.user_id
    if payload.agent_id:
        filters["agent_id"] = payload.agent_id
    if payload.app_id:
        filters["app_id"] = payload.app_id
    if payload.run_id:
        filters["run_id"] = payload.run_id
    results = service.search_memories(
        query=payload.query,
        filters=filters,
        limit=payload.limit or 100,
    )
    items: List[MemorySearchResult] = [MemorySearchResult(**item) for item in results]
    return SearchResponse(results=items)


@router_v2.post("/search/", response_model=SearchResponse)
def search_memories(
    payload: SearchRequest, service: MemoryService = Depends(get_memory_service)
) -> SearchResponse:
    results = service.search_memories(
        query=payload.query,
        filters=payload.filters,
        limit=payload.limit or 100,
    )
    items: List[MemorySearchResult] = [MemorySearchResult(**item) for item in results]
    return SearchResponse(results=items)


@router_v2.post("/", response_model=PaginatedResponse)
def list_memories(
    payload: ListRequest, service: MemoryService = Depends(get_memory_service)
) -> PaginatedResponse:
    records = service.list_memories(
        filters=payload.filters, page=payload.page, page_size=payload.page_size
    )
    results = [MemoryItem(**item) for item in records["results"]]
    return PaginatedResponse(
        count=records["count"],
        next=records["next"],
        previous=records["previous"],
        results=results,
    )


def health() -> Dict[str, str]:
    return {"status": "ok"}


@router_v1.get("/ping/")
def ping() -> Dict[str, Any]:
    return {"org_id": "local", "project_id": "local"}

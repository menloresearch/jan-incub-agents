import logging
import os
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api.v1.endpoints import health, router_v1, router_v2
from app.services.memory_service import MemoryService

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="Mem0-Compatible Memory Service",
    description="Drop-in replacement API for mem0-compatible clients",
    version="0.2.0",
)

memory_db_path = os.getenv("MEMORY_DB_PATH", "memory.db")


def _init_memory_service() -> MemoryService:
    logger.info("Initializing memory service with db path %s", memory_db_path)
    return MemoryService(db_path=memory_db_path)


@app.on_event("startup")
async def on_startup() -> None:
    app.state.memory_service = _init_memory_service()
    logger.info("Memory service ready")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    service: MemoryService = getattr(app.state, "memory_service", None)
    if service:
        service.close()
        logger.info("Memory service closed")


app.include_router(router_v1, prefix="/v1")
app.include_router(router_v2, prefix="/v2")


@app.get("/health")
async def health_check() -> dict:
    return health()


@app.get("/v1/ping/")
async def ping() -> Dict[str, str]:
    return {"org_id": "local", "project_id": "local"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

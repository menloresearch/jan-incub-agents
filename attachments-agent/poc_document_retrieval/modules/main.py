from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import tempfile
import os
import argparse
from pathlib import Path
import uvicorn
import sys

from modules.chunk_service import (
    ChunkService,
    create_chunk_service_from_config,
)


# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str = Field(
        ..., description="Role of the message sender (user, assistant, system)"
    )
    content: str = Field(..., description="Content of the message")


class ChunkResponse(BaseModel):
    chat_history: List[ChatMessage] = Field(
        ..., description="Enhanced chat history with retrieved chunks"
    )
    chunks_count: Optional[int] = Field(
        None, description="Total number of chunks created from document"
    )
    retrieved_chunks_count: Optional[int] = Field(
        None, description="Number of chunks retrieved"
    )
    document_info: Optional[Dict[str, Any]] = Field(
        None, description="Information about the processed document"
    )
    error: Optional[str] = Field(None, description="Error message if processing failed")


# Global service instance
chunk_service: Optional[ChunkService] = None


def initialize_service(config_path: str):
    """Initialize the global chunk service."""
    global chunk_service
    try:
        chunk_service = create_chunk_service_from_config(config_path)
        print(f"✅ Chunk service initialized with config: {config_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize chunk service: {e}")
        return False


# Removed get_chunk_service dependency function - using global service directly


# FastAPI app
app = FastAPI(
    title="Document Chunk Service",
    description="API for document chunking and retrieval without LLM generation",
    version="1.0.0",
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Document Chunk Service API",
        "version": "1.0.0",
        "endpoints": {
            "/chunk-upload": "POST - Upload document and retrieve relevant chunks",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "chunk_service"}


@app.post("/chunk-upload", response_model=ChunkResponse)
async def upload_and_process(
    file: UploadFile = File(...),
    chat_history: str = Form(..., description="JSON string of chat history"),
    max_tokens: Optional[int] = Form(-1),
    clear_retrieval_history: bool = Form(False, description="Remove previously retrieved chunks from conversation"),
):
    """
    Upload a document file and retrieve relevant chunks based on chat history.

    Input:
    - file: PDF document or .txt file
    - chat_history: JSON string of chat messages with 'role' and 'content' keys
    - max_tokens: Optional maximum tokens to consider
    - clear_retrieval_history: Remove previously retrieved chunks from conversation. Default: False

    Output:
    - Enhanced chat_history with retrieved chunks inserted as context
    """
    try:
        import json

        # Parse chat history from JSON string
        try:
            chat_history_data = json.loads(chat_history)
            chat_messages = [ChatMessage(**msg) for msg in chat_history_data]
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid chat_history JSON: {str(e)}"
            )

        # Upload file to temporary location
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Convert Pydantic models to dict for service
            chat_history_dict = [
                {"role": msg.role, "content": msg.content} for msg in chat_messages
            ]

            # Check if service is initialized
            if chunk_service is None:
                raise HTTPException(
                    status_code=500, detail="Chunk service not initialized"
                )

            # Process document and retrieve chunks
            result = chunk_service.process_document_and_retrieve_chunks(
                file_path=tmp_file_path,
                chat_history=chat_history_dict,
                max_tokens=max_tokens,
                clear_retrieval_history=clear_retrieval_history,
            )

            # Convert back to Pydantic models
            if "error" in result:
                return ChunkResponse(
                    chat_history=[ChatMessage(**msg) for msg in result["chat_history"]],
                    error=result["error"],
                )
            else:
                return ChunkResponse(
                    chat_history=[ChatMessage(**msg) for msg in result["chat_history"]],
                    chunks_count=result.get("chunks_count"),
                    retrieved_chunks_count=result.get("retrieved_chunks_count"),
                    document_info=result.get("document_info"),
                )

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Upload and processing failed: {str(e)}"
        )


@app.on_event("startup")
async def startup_event():
    """Initialize chunk service on startup."""
    global chunk_service
    config_path = os.getenv("CHUNK_SERVICE_CONFIG")
    if config_path:
        success = initialize_service(config_path)
        if not success:
            print("❌ Failed to initialize chunk service on startup")
    else:
        print("⚠️ No CHUNK_SERVICE_CONFIG environment variable set")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global chunk_service
    if chunk_service:
        chunk_service.cleanup_models()


def main():
    """
    Main function to run the FastAPI server with CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Document Chunk Service API")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file (e.g., ../config/bge_m3_400.yaml)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    # Set environment variable for startup event
    import os

    os.environ["CHUNK_SERVICE_CONFIG"] = args.config

    print(f"🔧 Setting CHUNK_SERVICE_CONFIG to: {args.config}")
    print(f"🚀 Starting server on {args.host}:{args.port}")
    print("💡 Recommended: Use 'uvicorn main:app --host 0.0.0.0 --port 8000' instead")

    # Run the FastAPI server
    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

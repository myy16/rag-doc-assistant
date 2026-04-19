import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import CHROMA_TOP_K
from app.core.rag_service import get_rag_service

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    file_id: Optional[str] = None
    source_file: Optional[str] = None
    top_k: int = Field(default=CHROMA_TOP_K, ge=1, le=20)


@router.post("/chat")
def chat(request: ChatRequest):
    try:
        service = get_rag_service()
        return service.answer_question(
            question=request.question,
            top_k=request.top_k,
            file_id=request.file_id,
            source_file=request.source_file,
        )
    except RuntimeError as exc:
        logger.warning("Chat request failed due to runtime dependency issue: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while handling /api/chat")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat response with Server-Sent Events (SSE).
    
    Response format (newline-delimited SSE):
    data: {"type": "token", "content": "word"}
    
    data: {"type": "sources", "content": [{...}, ...]}
    """
    try:
        service = get_rag_service()
        def safe_stream():
            try:
                yield from service.answer_question_stream(
                    question=request.question,
                    top_k=request.top_k,
                    file_id=request.file_id,
                    source_file=request.source_file,
                )
            except RuntimeError as exc:
                logger.warning("Streaming chat failed due to runtime dependency issue: %s", exc)
                payload = json.dumps({"type": "error", "detail": str(exc)})
                yield f"data: {payload}\n\n"
            except Exception:
                logger.exception("Unexpected error while streaming /api/chat/stream")
                payload = json.dumps({"type": "error", "detail": "Internal server error."})
                yield f"data: {payload}\n\n"

        return StreamingResponse(safe_stream(), media_type="text/event-stream")
    except RuntimeError as exc:
        logger.warning("Unable to initialize streaming chat: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while initializing /api/chat/stream")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc
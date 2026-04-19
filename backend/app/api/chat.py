from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import CHROMA_TOP_K
from app.core.rag_service import get_rag_service

router = APIRouter()


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
        raise HTTPException(status_code=503, detail=str(exc)) from exc
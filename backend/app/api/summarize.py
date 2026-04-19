import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.rag_service import get_rag_service

router = APIRouter()
logger = logging.getLogger(__name__)


class SummarizeRequest(BaseModel):
    file_id: Optional[str] = None
    source_file: Optional[str] = None
    max_chunks: int = Field(default=8, ge=1, le=20)


@router.post("/summarize")
def summarize(request: SummarizeRequest):
    try:
        service = get_rag_service()
        return service.summarize_documents(
            file_id=request.file_id,
            source_file=request.source_file,
            max_chunks=request.max_chunks,
        )
    except RuntimeError as exc:
        logger.warning("Summarize request failed due to runtime dependency issue: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while handling /api/summarize")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc
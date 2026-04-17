import json
import os
import uuid
from typing import AsyncGenerator, List

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, UPLOAD_DIR
from app.core.parser import parse_document
from app.core.cleaner import clean_text
from app.core.chunker import chunk_text

router = APIRouter()


def _validate_file(file: UploadFile) -> str:
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"'{file.filename}' is not supported. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    return ext


async def _process_file(file: UploadFile) -> dict:
    """Validate, save, parse, clean and chunk a single uploaded file."""
    ext = _validate_file(file)

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"'{file.filename}' exceeds the {MAX_FILE_SIZE_MB} MB size limit.",
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}.{ext}")
    with open(save_path, "wb") as f:
        f.write(content)

    raw_text = parse_document(save_path, ext)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(
        text=cleaned_text,
        metadata={
            "file_id": file_id,
            "source_file": file.filename,
            "file_type": ext,
        },
    )

    return {
        "file_id": file_id,
        "original_name": file.filename,
        "file_type": ext,
        "size_mb": round(size_mb, 3),
        "saved_path": save_path,
        "extracted_text": cleaned_text,
        "chunks": chunks,
        "chunk_count": len(chunks),
    }


@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload one or more documents (PDF, DOCX, TXT, DOC).
    Returns all results at once after processing every file.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    uploaded = []
    for file in files:
        try:
            result = await _process_file(file)
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))
        uploaded.append(result)

    return JSONResponse(
        status_code=200,
        content={"uploaded_files": uploaded, "count": len(uploaded)},
    )


@router.post("/upload/stream")
async def upload_documents_stream(files: List[UploadFile] = File(...)):
    """
    Upload one or more documents with Server-Sent Events (SSE) streaming.

    Emits one JSON event per file as it finishes processing, then a final
    'done' event. Frontend can consume with EventSource or fetch + ReadableStream.

    Event format:
        data: {"event": "file_done",  "file": {...}}
        data: {"event": "error",      "filename": "...", "detail": "..."}
        data: {"event": "done",       "count": N}
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    async def event_generator() -> AsyncGenerator[str, None]:
        count = 0
        for file in files:
            try:
                result = await _process_file(file)
                payload = json.dumps({"event": "file_done", "file": result}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                count += 1
            except (HTTPException, RuntimeError) as exc:
                detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
                payload = json.dumps(
                    {"event": "error", "filename": file.filename, "detail": detail},
                    ensure_ascii=False,
                )
                yield f"data: {payload}\n\n"

        yield f"data: {json.dumps({'event': 'done', 'count': count})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
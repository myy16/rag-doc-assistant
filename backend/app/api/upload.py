import asyncio
import json
import os
import uuid
from typing import AsyncGenerator, Dict, Iterator, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional

from app.core.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, UPLOAD_DIR
from app.core.rag_service import get_rag_service
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


async def _read_upload(file: UploadFile) -> tuple:
    """Read file content eagerly before any streaming starts."""
    ext = _validate_file(file)
    content = await file.read()
    return file.filename, ext, content


def _process_content(filename: str, ext: str, content: bytes, username: str = "") -> dict:
    """Save, parse, clean and chunk already-read file content."""
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"'{filename}' exceeds the {MAX_FILE_SIZE_MB} MB size limit.",
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Aynı kullanıcıdan aynı isimli dosya varsa önce sil
    service = get_rag_service()
    if username:
        store = service.vector_store
        existing = store.fetch_all(filters={"$and": [
            {"username": {"$eq": username}},
            {"source_file": {"$eq": filename}},
        ]})
        old_ids = existing.get("ids") or []
        if old_ids:
            store.collection().delete(ids=old_ids)

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
            "source_file": filename,
            "file_type": ext,
            "username": username,
        },
    )

    if chunks:
        service.index_chunks(chunks)

    return {
        "file_id": file_id,
        "original_name": filename,
        "file_type": ext,
        "size_mb": round(size_mb, 3),
        "saved_path": save_path,
        "extracted_text": cleaned_text,
        "chunks": chunks,
        "chunk_count": len(chunks),
    }


def _stream_process_content_events(filename: str, ext: str, content: bytes, username: str = "") -> Iterator[Dict[str, object]]:
    """Yield progress events for upload/parse/index stages."""
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"'{filename}' exceeds the {MAX_FILE_SIZE_MB} MB size limit.",
        )

    yield {"event": "stage", "stage": "Dosya kaydediliyor", "filename": filename}

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    service = get_rag_service()
    if username:
        store = service.vector_store
        existing = store.fetch_all(filters={"$and": [
            {"username": {"$eq": username}},
            {"source_file": {"$eq": filename}},
        ]})
        old_ids = existing.get("ids") or []
        if old_ids:
            store.collection().delete(ids=old_ids)

    file_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}.{ext}")
    with open(save_path, "wb") as f:
        f.write(content)

    yield {"event": "stage", "stage": "Metin çıkarılıyor", "filename": filename}
    raw_text = parse_document(save_path, ext)

    yield {"event": "stage", "stage": "Temizleniyor", "filename": filename}
    cleaned_text = clean_text(raw_text)

    yield {"event": "stage", "stage": "Chunking yapılıyor", "filename": filename}
    chunks = chunk_text(
        text=cleaned_text,
        metadata={
            "file_id": file_id,
            "source_file": filename,
            "file_type": ext,
            "username": username,
        },
    )

    if chunks:
        yield {"event": "stage", "stage": "Embedding ve indeksleme", "filename": filename}
        service.index_chunks(chunks)

    result = {
        "file_id": file_id,
        "original_name": filename,
        "file_type": ext,
        "size_mb": round(size_mb, 3),
        "saved_path": save_path,
        "extracted_text": cleaned_text,
        "chunks": chunks,
        "chunk_count": len(chunks),
    }
    yield {"event": "file_done", "file": result, "filename": filename}


@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    username: Optional[str] = Form(default=""),
):
    """
    Upload one or more documents (PDF, DOCX, TXT, DOC).
    Returns all results at once after processing every file.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    uploaded = []
    for file in files:
        try:
            filename, ext, content = await _read_upload(file)
            result = _process_content(filename, ext, content, username=username or "")
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))
        uploaded.append(result)

    return JSONResponse(
        status_code=200,
        content={"uploaded_files": uploaded, "count": len(uploaded)},
    )


@router.post("/upload/stream")
async def upload_documents_stream(
    files: List[UploadFile] = File(...),
    username: Optional[str] = Form(default=""),
):
    """
    Upload one or more documents with Server-Sent Events (SSE) streaming.

    Reads all file contents eagerly before streaming begins to avoid
    closed-file errors inside the async generator.

    Emits one JSON event per file as it finishes processing, then a final
    'done' event. Frontend can consume with EventSource or fetch + ReadableStream.

    Event format:
        data: {"event": "file_done",  "file": {...}}
        data: {"event": "error",      "filename": "...", "detail": "..."}
        data: {"event": "done",       "count": N}
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # Read all file contents before entering the generator
    file_payloads = []
    for file in files:
        try:
            filename, ext, content = await _read_upload(file)
            file_payloads.append((filename, ext, content))
        except HTTPException as exc:
            raise exc

    async def event_generator() -> AsyncGenerator[str, None]:
        count = 0
        for filename, ext, content in file_payloads:
            try:
                for event in _stream_process_content_events(filename, ext, content, username=username or ""):
                    payload = json.dumps(event, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                    if event.get("event") == "file_done":
                        count += 1
                    await asyncio.sleep(0)
            except (HTTPException, RuntimeError) as exc:
                detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
                payload = json.dumps(
                    {"event": "error", "filename": filename, "detail": detail},
                    ensure_ascii=False,
                )
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0)
            except Exception as exc:
                payload = json.dumps(
                    {"event": "error", "filename": filename, "detail": f"Unexpected error: {exc}"},
                    ensure_ascii=False,
                )
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0)

        yield f"data: {json.dumps({'event': 'done', 'count': count})}\n\n"
        await asyncio.sleep(0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class DeleteRequest(BaseModel):
    file_id: str


@router.delete("/upload")
def delete_document(request: DeleteRequest):
    """Remove a document's chunks from the vector store and delete the saved file."""
    service = get_rag_service()
    deleted_chunks = service.vector_store.delete_by_file_id(request.file_id)

    # Remove the physical file (any extension)
    import glob
    pattern = os.path.join(UPLOAD_DIR, f"{request.file_id}.*")
    for path in glob.glob(pattern):
        try:
            os.remove(path)
        except OSError:
            pass

    return {"deleted_chunks": deleted_chunks, "file_id": request.file_id}
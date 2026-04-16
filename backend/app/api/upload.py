import os
import uuid
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from app.core.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, UPLOAD_DIR

router = APIRouter()


def _validate_file(file: UploadFile) -> str:
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"'{file.filename}' is not supported. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    return ext


@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload one or more documents (PDF, DOCX, TXT, DOC).
    Use the file picker — do not type string values directly.
    Returns metadata for each uploaded file.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    uploaded = []
    for file in files:
        ext = _validate_file(file)

        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"'{file.filename}' exceeds the {MAX_FILE_SIZE_MB} MB size limit.",
            )

        file_id = str(uuid.uuid4())
        save_path = os.path.join(UPLOAD_DIR, f"{file_id}.{ext}")
        with open(save_path, "wb") as f:
            f.write(content)

        uploaded.append({
            "file_id": file_id,
            "original_name": file.filename,
            "file_type": ext,
            "size_mb": round(size_mb, 3),
            "saved_path": save_path,
        })

    return JSONResponse(
        status_code=200,
        content={"uploaded_files": uploaded, "count": len(uploaded)},
    )

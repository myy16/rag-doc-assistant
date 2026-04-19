import logging

from fastapi import FastAPI
from app.api.upload import router as upload_router
from app.api.chat import router as chat_router
from app.api.summarize import router as summarize_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(
    title="P2P YZTA - Chat with Your Documents",
    version="1.0.0",
)

app.include_router(upload_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(summarize_router, prefix="/api")


@app.get("/")
def health_check():
    return {"status": "ok"}

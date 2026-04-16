from fastapi import FastAPI
from app.api.upload import router as upload_router

app = FastAPI(
    title="P2P YZTA - Chat with Your Documents",
    version="1.0.0",
)

app.include_router(upload_router, prefix="/api")


@app.get("/")
def health_check():
    return {"status": "ok"}

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import CHROMA_COLLECTION, CHROMA_DIR


class ChromaStore:
    def __init__(self, persist_directory: str = CHROMA_DIR, collection_name: str = CHROMA_COLLECTION):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _load_client(self):
        if self._client is None:
            try:
                import chromadb  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "chromadb is not installed. Install backend dependencies first."
                ) from exc

            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        return self._client

    def collection(self):
        if self._collection is None:
            client = self._load_client()
            self._collection = client.get_or_create_collection(name=self.collection_name)
        return self._collection

    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        clean_metadata: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean_metadata[key] = value
            else:
                clean_metadata[key] = str(value)
        return clean_metadata

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError("Chunk count and embedding count must match.")

        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            metadata = {
                "file_id": chunk.get("file_id", ""),
                "source_file": chunk.get("source_file", ""),
                "file_type": chunk.get("file_type", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", len(chunks)),
                "char_count": chunk.get("char_count", len(chunk.get("text", ""))),
            }
            metadatas.append(self._sanitize_metadata(metadata))

        self.collection().upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunks)

    def query(self, query_embedding: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            query_kwargs["where"] = filters
        return self.collection().query(**query_kwargs)

    def fetch_all(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "include": ["documents", "metadatas"],
        }
        if filters:
            kwargs["where"] = filters
        return self.collection().get(**kwargs)


_STORE: Optional[ChromaStore] = None


def get_vector_store() -> ChromaStore:
    global _STORE
    if _STORE is None:
        _STORE = ChromaStore()
    return _STORE
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.config import CHROMA_TOP_K
from app.core.embeddings import get_embedding_service
from app.core.vector_store import get_vector_store


class Retriever:
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store()

    @staticmethod
    def _build_filters(file_id: Optional[str] = None, source_file: Optional[str] = None) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}
        if file_id:
            filters["file_id"] = file_id
        if source_file:
            filters["source_file"] = source_file
        return filters

    def retrieve(self, query: str, top_k: int = CHROMA_TOP_K, file_id: Optional[str] = None, source_file: Optional[str] = None) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_service.embed_query(query)
        if not query_embedding:
            return []

        filters = self._build_filters(file_id=file_id, source_file=source_file)
        results = self.vector_store.query(query_embedding=query_embedding, top_k=top_k, filters=filters or None)
        return self._format_results(results)

    def fetch_documents(self, file_id: Optional[str] = None, source_file: Optional[str] = None) -> List[Dict[str, Any]]:
        filters = self._build_filters(file_id=file_id, source_file=source_file)
        results = self.vector_store.fetch_all(filters=filters or None)
        return self._format_fetched_documents(results)

    @staticmethod
    def _format_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        formatted: List[Dict[str, Any]] = []
        for index, text in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            formatted.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "distance": distances[index] if index < len(distances) else None,
                    "chunk_index": metadata.get("chunk_index"),
                    "source_file": metadata.get("source_file"),
                    "file_id": metadata.get("file_id"),
                }
            )
        return formatted

    @staticmethod
    def _format_fetched_documents(results: Dict[str, Any]) -> List[Dict[str, Any]]:
        documents = results.get("documents", []) or []
        metadatas = results.get("metadatas", []) or []
        ids = results.get("ids", []) or []

        formatted: List[Dict[str, Any]] = []
        for index, text in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            formatted.append(
                {
                    "id": ids[index] if index < len(ids) else None,
                    "text": text,
                    "metadata": metadata,
                    "chunk_index": metadata.get("chunk_index"),
                    "source_file": metadata.get("source_file"),
                    "file_id": metadata.get("file_id"),
                }
            )
        return formatted


def get_retriever() -> Retriever:
    return Retriever()
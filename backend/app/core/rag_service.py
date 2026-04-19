from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List, Optional

from app.core.config import CHROMA_TOP_K, GROQ_API_KEY, GROQ_MODEL_NAME
from app.core.embeddings import get_embedding_service
from app.core.retriever import get_retriever
from app.core.vector_store import get_vector_store


class RagService:
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.retriever = get_retriever()
        self.vector_store = get_vector_store()
        self.model_name = GROQ_MODEL_NAME
        self.api_key = GROQ_API_KEY

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)
        return self.vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)

    def answer_question(
        self,
        question: str,
        top_k: int = CHROMA_TOP_K,
        file_id: Optional[str] = None,
        source_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        chunks = self.retriever.retrieve(question, top_k=top_k, file_id=file_id, source_file=source_file)
        if not chunks:
            return {
                "answer": "İlgili bağlam bulunamadı.",
                "sources": [],
                "context": [],
                "model": self.model_name,
            }

        context_text = self._build_context(chunks)
        answer = self._generate_answer(question=question, context_text=context_text)
        sources = self._build_sources(chunks)
        return {
            "answer": answer,
            "sources": sources,
            "context": chunks,
            "model": self.model_name,
        }

    def answer_question_stream(
        self,
        question: str,
        top_k: int = CHROMA_TOP_K,
        file_id: Optional[str] = None,
        source_file: Optional[str] = None,
    ):
        """Answer question with streaming response. Yields SSE events.
        
        Stream format (one JSON per line):
        data: {"type": "token", "content": "text"}
        data: {"type": "sources", "content": [...]}
        """
        chunks = self.retriever.retrieve(question, top_k=top_k, file_id=file_id, source_file=source_file)
        if not chunks:
            event = {"type": "message", "content": "İlgili bağlam bulunamadı.", "model": self.model_name}
            yield f"data: {json.dumps(event)}\n\n"
            return

        context_text = self._build_context(chunks)
        prompt = (
            "You are a helpful assistant answering questions only from the provided context. "
            "If the answer is not in the context, say you could not find it. "
            "Cite sources inline with [source_file:chunk_index].\n\n"
            f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
        )
        
        # Stream tokens from Groq
        for token in self._call_groq_stream(prompt=prompt):
            event = {"type": "token", "content": token}
            yield f"data: {json.dumps(event)}\n\n"
        
        # Send sources after streaming completes
        sources = self._build_sources(chunks)
        event = {"type": "sources", "content": sources}
        yield f"data: {json.dumps(event)}\n\n"

    def summarize_documents(
        self,
        file_id: Optional[str] = None,
        source_file: Optional[str] = None,
        max_chunks: int = 8,
    ) -> Dict[str, Any]:
        chunks = self.retriever.fetch_documents(file_id=file_id, source_file=source_file)
        if not chunks:
            return {
                "summary": "Özetlenecek doküman bulunamadı.",
                "sources": [],
                "context": [],
                "model": self.model_name,
            }

        ordered_chunks = sorted(chunks, key=lambda item: (item.get("source_file") or "", item.get("chunk_index") or 0))
        selected_chunks = ordered_chunks[:max_chunks]
        context_text = self._build_context(selected_chunks)
        summary = self._generate_summary(context_text=context_text)
        sources = self._build_sources(selected_chunks)
        return {
            "summary": summary,
            "sources": sources,
            "context": selected_chunks,
            "model": self.model_name,
        }

    def _generate_answer(self, question: str, context_text: str) -> str:
        prompt = (
            "You are a helpful assistant answering questions only from the provided context. "
            "If the answer is not in the context, say you could not find it. "
            "Cite sources inline with [source_file:chunk_index].\n\n"
            f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
        )
        return self._call_groq(prompt=prompt)

    def _generate_summary(self, context_text: str) -> str:
        prompt = (
            "Summarize the provided document chunks in Turkish. "
            "Keep the summary concise, factual, and grounded in the text.\n\n"
            f"Context:\n{context_text}\n\nSummary:"
        )
        return self._call_groq(prompt=prompt)

    def _call_groq(self, prompt: str) -> str:
        try:
            from groq import Groq  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("groq is not installed. Install backend dependencies first.") from exc

        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        client = Groq(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You answer strictly using the given context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def _call_groq_stream(self, prompt: str):
        """Stream tokens from Groq API. Yields text chunks."""
        try:
            from groq import Groq  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("groq is not installed. Install backend dependencies first.") from exc

        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        client = Groq(api_key=self.api_key)
        stream = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You answer strictly using the given context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @staticmethod
    def _build_context(chunks: List[Dict[str, Any]]) -> str:
        sections = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            source_file = chunk.get("source_file") or metadata.get("source_file") or "unknown"
            chunk_index = chunk.get("chunk_index") if chunk.get("chunk_index") is not None else metadata.get("chunk_index", 0)
            sections.append(f"[{source_file}:{chunk_index}]\n{chunk.get('text', '')}")
        return "\n\n".join(sections)

    @staticmethod
    def _build_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            source_file = chunk.get("source_file") or metadata.get("source_file")
            chunk_index = chunk.get("chunk_index") if chunk.get("chunk_index") is not None else metadata.get("chunk_index")
            sources.append(
                {
                    "source_file": source_file,
                    "file_id": chunk.get("file_id") or metadata.get("file_id"),
                    "chunk_index": chunk_index,
                    "char_count": metadata.get("char_count"),
                }
            )
        return sources


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    return RagService()
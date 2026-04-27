from __future__ import annotations

import json
import logging
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

from app.core.config import (
    CHROMA_TOP_K,
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    LLM_CIRCUIT_BREAKER_THRESHOLD,
    LLM_MAX_RETRIES,
    LLM_RETRY_BASE_DELAY_SECONDS,
    LLM_RETRY_MAX_DELAY_SECONDS,
    LLM_TIMEOUT_SECONDS,
)
from app.core.embeddings import get_embedding_service
from app.core.evaluator import evaluate_rag
from app.core.retriever import get_retriever
from app.core.vector_store import get_vector_store


logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are an advanced RAG system with retrieval, reasoning, and self-evaluation capabilities.

Your job is to ANSWER the question using the provided context — even if confidence scores are low.

---

## CRITICAL OVERRIDE RULE

If the answer is explicitly present in the context,
YOU MUST answer the question.

DO NOT refuse to answer due to:
* low confidence
* low coverage
* low relevance scores

These scores are ONLY hints, not hard constraints.

---

## CONTEXT TRUST LOGIC

Treat the context as the source of truth.

If ANY of the following is true:
* A keyword from the question appears in the context
* A named entity (event, person, project) is present
* The answer can be directly extracted from a sentence

→ THEN the context is SUFFICIENT.

---

## FAILURE CONDITION (STRICT)

ONLY say "insufficient context" IF:
* The answer is completely absent
* OR there is zero mention of relevant entities

---

## QUESTION EXAMPLE

Question:
"Yusuf hangi yarışmaya katılmış?"

Context:
"... TEKNOFEST gibi yarışmalarda ..."

CORRECT BEHAVIOR:
* Detect "TEKNOFEST"
* Return it as the answer

INCORRECT BEHAVIOR:
* Saying "coverage low"
* Saying "confidence low"
* Refusing to answer

---

## OUTPUT RULES

* Give a direct, short answer
* Do NOT mention evaluation scores
* Do NOT mention confidence
* Do NOT mention retrieval metrics

---

## FINAL INSTRUCTION

Even if evaluation signals are weak,
IF the answer is visible → RETURN IT.

Never hide a correct answer due to scoring heuristics.
"""


class RagService:
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.retriever = get_retriever()
        self.vector_store = get_vector_store()
        self.model_name = GROQ_MODEL_NAME
        self.api_key = GROQ_API_KEY
        self.llm_timeout_seconds = LLM_TIMEOUT_SECONDS
        self.llm_max_retries = max(0, LLM_MAX_RETRIES)
        self.llm_retry_base_delay_seconds = max(0.0, LLM_RETRY_BASE_DELAY_SECONDS)
        self.llm_retry_max_delay_seconds = max(
            self.llm_retry_base_delay_seconds,
            LLM_RETRY_MAX_DELAY_SECONDS,
        )
        self.llm_circuit_breaker_threshold = max(1, LLM_CIRCUIT_BREAKER_THRESHOLD)
        self.llm_circuit_breaker_cooldown_seconds = max(1.0, LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS)
        self._llm_consecutive_failures = 0
        self._llm_circuit_open_until = 0.0

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            logger.debug("No chunks received for indexing.")
            return 0

        try:
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_service.embed_texts(texts)
            count = self.vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)
            logger.info("Indexed %s chunks into vector store.", count)
            return count
        except Exception as exc:
            logger.exception("Failed to index chunks into vector store.")
            raise RuntimeError("Failed to index document chunks.") from exc

    def answer_question(
        self,
        question: str,
        top_k: int = CHROMA_TOP_K,
        file_id: Optional[str] = None,
        source_file: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            if hasattr(self.retriever, "retrieve_with_diagnostics"):
                retrieval = self.retriever.retrieve_with_diagnostics(
                    query=question,
                    top_k=top_k,
                    file_id=file_id,
                    source_file=source_file,
                    username=username or None,
                )
                chunks = retrieval.get("chunks", [])
            else:
                chunks = self.retriever.retrieve(
                    question,
                    top_k=top_k,
                    file_id=file_id,
                    source_file=source_file,
                    username=username or None,
                )
                retrieval = {
                    "confidence_score": 0.0,
                    "context_coverage": 0.0,
                    "retrieval_quality": bool(chunks),
                    "query_variants": [question],
                }
        except Exception as exc:
            logger.exception("Retriever failed while answering question.")
            raise RuntimeError("Failed to retrieve context for question.") from exc

        if not chunks:
            logger.info("No context found for question.")
            answer = "İlgili bağlam bulunamadı. Daha spesifik bir soru sorabilir veya farklı dokümanlar yükleyebilirsin."
            evaluation = evaluate_rag(
                question=question,
                chunks=[],
                answer=answer,
                retrieval_confidence=retrieval.get("confidence_score", 0.0),
                retrieval_quality=False,
            )
            return {
                "answer": answer,
                "sources": [],
                "context": [],
                "model": self.model_name,
                "evaluation": evaluation,
                "retrieval": retrieval,
            }


        context_text = self._build_context(chunks)
        try:
            answer = self._generate_answer(question=question, context_text=context_text)
        except RuntimeError as exc:
            logger.warning("LLM unavailable during answer generation, using fallback: %s", exc)
            answer = self._build_fallback_answer(question=question, chunks=chunks)
        sources = self._build_sources(chunks)
        evaluation = evaluate_rag(
            question=question,
            chunks=chunks,
            answer=answer,
            retrieval_confidence=retrieval.get("confidence_score", 0.0),
            retrieval_quality=retrieval.get("retrieval_quality", False),
        )
        logger.info("Generated answer using %s retrieved chunks.", len(chunks))
        return {
            "answer": answer,
            "sources": sources,
            "context": chunks,
            "model": self.model_name,
            "evaluation": evaluation,
            "retrieval": retrieval,
        }

    def answer_question_stream(
        self,
        question: str,
        top_k: int = CHROMA_TOP_K,
        file_id: Optional[str] = None,
        source_file: Optional[str] = None,
        username: Optional[str] = None,
    ):
        """Answer question with streaming response. Yields SSE events.

        Stream format (one JSON per line):
        data: {"type": "token", "content": "text"}
        data: {"type": "sources", "content": [...]}
        """
        try:
            if hasattr(self.retriever, "retrieve_with_diagnostics"):
                retrieval = self.retriever.retrieve_with_diagnostics(
                    query=question,
                    top_k=top_k,
                    file_id=file_id,
                    source_file=source_file,
                    username=username or None,
                )
                chunks = retrieval.get("chunks", [])
            else:
                chunks = self.retriever.retrieve(
                    question,
                    top_k=top_k,
                    file_id=file_id,
                    source_file=source_file,
                    username=username or None,
                )
                retrieval = {
                    "confidence_score": 0.0,
                    "context_coverage": 0.0,
                    "retrieval_quality": bool(chunks),
                    "query_variants": [question],
                }
        except Exception as exc:
            logger.exception("Retriever failed while preparing streaming answer.")
            raise RuntimeError("Failed to retrieve context for streaming question.") from exc

        if not chunks:
            answer = "İlgili bağlam bulunamadı. Daha spesifik bir soru sorabilir veya farklı dokümanlar yükleyebilirsin."
            event = {"type": "message", "content": answer, "model": self.model_name}
            yield f"data: {json.dumps(event)}\n\n"
            evaluation = evaluate_rag(
                question=question,
                chunks=[],
                answer=answer,
                retrieval_confidence=retrieval.get("confidence_score", 0.0),
                retrieval_quality=False,
            )
            yield f"data: {json.dumps({'type': 'retrieval', 'content': retrieval})}\n\n"
            yield f"data: {json.dumps({'type': 'evaluation', 'content': evaluation})}\n\n"
            return


        context_text = self._build_context(chunks)
        prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"

        answer_parts: List[str] = []
        # Stream tokens from Groq; if it fails, degrade gracefully to context-based fallback text.
        try:
            for token in self._call_groq_stream(prompt=prompt, system_prompt=RAG_SYSTEM_PROMPT):
                answer_parts.append(token)
                event = {"type": "token", "content": token}
                yield f"data: {json.dumps(event)}\n\n"
        except RuntimeError as exc:
            logger.warning("LLM streaming unavailable, using fallback response: %s", exc)
            fallback = self._build_fallback_answer(question=question, chunks=chunks)
            answer_parts = [fallback]
            event = {"type": "token", "content": fallback}
            yield f"data: {json.dumps(event)}\n\n"
        
        # Send sources after streaming completes
        sources = self._build_sources(chunks)
        event = {"type": "sources", "content": sources}
        yield f"data: {json.dumps(event)}\n\n"

        evaluation = evaluate_rag(
            question=question,
            chunks=chunks,
            answer="".join(answer_parts),
            retrieval_confidence=retrieval.get("confidence_score", 0.0),
            retrieval_quality=retrieval.get("retrieval_quality", False),
        )
        yield f"data: {json.dumps({'type': 'retrieval', 'content': retrieval})}\n\n"
        yield f"data: {json.dumps({'type': 'evaluation', 'content': evaluation})}\n\n"

    def summarize_documents(
        self,
        file_id: Optional[str] = None,
        source_file: Optional[str] = None,
        max_chunks: int = 8,
        username: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            chunks = self.retriever.fetch_documents(file_id=file_id, source_file=source_file, username=username or None)
        except Exception as exc:
            logger.exception("Retriever failed while summarizing documents.")
            raise RuntimeError("Failed to fetch documents for summarization.") from exc

        if not chunks:
            logger.info("No documents found for summarization request.")
            return {
                "summary": "Özetlenecek doküman bulunamadı.",
                "sources": [],
                "context": [],
                "model": self.model_name,
            }

        ordered_chunks = sorted(chunks, key=lambda item: (item.get("source_file") or "", item.get("chunk_index") or 0))
        selected_chunks = ordered_chunks[:max_chunks]
        context_text = self._build_context(selected_chunks)
        try:
            summary = self._generate_summary(context_text=context_text)
        except RuntimeError as exc:
            logger.warning("LLM unavailable during summarization, using fallback: %s", exc)
            summary = self._build_fallback_summary(chunks=selected_chunks, error_msg=str(exc))
        sources = self._build_sources(selected_chunks)
        logger.info("Generated summary using %s chunks.", len(selected_chunks))
        return {
            "summary": summary,
            "sources": sources,
            "context": selected_chunks,
            "model": self.model_name,
        }

    def _generate_answer(self, question: str, context_text: str) -> str:
        prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
        return self._call_groq(prompt=prompt, system_prompt=RAG_SYSTEM_PROMPT)

    def _generate_summary(self, context_text: str) -> str:
        prompt = (
            "Aşağıdaki doküman parçalarını Türkçe olarak özetle. "
            "Özet kısa, net ve yalnızca metne dayalı olsun.\n\n"
            f"Bağlam:\n{context_text}\n\nÖzet:"
        )
        return self._call_groq(prompt=prompt)

    def _call_groq(self, prompt: str, system_prompt: str = "You answer strictly using the given context.") -> str:
        try:
            from groq import Groq  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("groq is not installed. Install backend dependencies first.") from exc

        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        if self._is_llm_circuit_open():
            remaining = max(0.0, self._llm_circuit_open_until - time.time())
            raise RuntimeError(f"LLM circuit breaker is open for {remaining:.1f}s.")

        attempts = self.llm_max_retries + 1
        for attempt in range(1, attempts + 1):
            started_at = time.perf_counter()
            try:
                client = Groq(api_key=self.api_key, timeout=self.llm_timeout_seconds, max_retries=0)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                elapsed = time.perf_counter() - started_at
                self._record_llm_success()
                logger.info(
                    "Groq completion succeeded (attempt=%s/%s, elapsed=%.2fs, model=%s).",
                    attempt,
                    attempts,
                    elapsed,
                    self.model_name,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                elapsed = time.perf_counter() - started_at
                self._record_llm_failure(exc=exc, operation="completion")
                if attempt >= attempts:
                    logger.exception(
                        "Groq completion failed after %s attempts (elapsed=%.2fs, model=%s).",
                        attempts,
                        elapsed,
                        self.model_name,
                    )
                    raise RuntimeError(f"LLM generation failed: {str(exc)}") from exc

                delay = min(
                    self.llm_retry_max_delay_seconds,
                    self.llm_retry_base_delay_seconds * (2 ** (attempt - 1)),
                )
                logger.warning(
                    "Groq completion failed (attempt=%s/%s, elapsed=%.2fs). Retrying in %.2fs. Error: %s",
                    attempt,
                    attempts,
                    elapsed,
                    delay,
                    exc,
                )
                if delay > 0:
                    time.sleep(delay)

    def _call_groq_stream(self, prompt: str, system_prompt: str = "You answer strictly using the given context."):
        """Stream tokens from Groq API. Yields text chunks."""
        try:
            from groq import Groq  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("groq is not installed. Install backend dependencies first.") from exc

        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        if self._is_llm_circuit_open():
            remaining = max(0.0, self._llm_circuit_open_until - time.time())
            raise RuntimeError(f"LLM circuit breaker is open for {remaining:.1f}s.")

        attempts = self.llm_max_retries + 1
        for attempt in range(1, attempts + 1):
            started_at = time.perf_counter()
            emitted_tokens = 0
            try:
                client = Groq(api_key=self.api_key, timeout=self.llm_timeout_seconds, max_retries=0)
                stream = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    stream=True,
                )
                for chunk in stream:
                    token = chunk.choices[0].delta.content
                    if token:
                        emitted_tokens += 1
                        yield token

                elapsed = time.perf_counter() - started_at
                self._record_llm_success()
                logger.info(
                    "Groq stream succeeded (attempt=%s/%s, tokens=%s, elapsed=%.2fs, model=%s).",
                    attempt,
                    attempts,
                    emitted_tokens,
                    elapsed,
                    self.model_name,
                )
                return
            except Exception as exc:
                elapsed = time.perf_counter() - started_at
                self._record_llm_failure(exc=exc, operation="stream")

                # If streaming already started, avoid retry to prevent duplicate partial output.
                if emitted_tokens > 0:
                    logger.exception(
                        "Groq stream failed after partial output (tokens=%s, elapsed=%.2fs, model=%s).",
                        emitted_tokens,
                        elapsed,
                        self.model_name,
                    )
                    raise RuntimeError(f"LLM streaming failed: {str(exc)}") from exc

                if attempt >= attempts:
                    logger.exception(
                        "Groq stream failed after %s attempts (elapsed=%.2fs, model=%s).",
                        attempts,
                        elapsed,
                        self.model_name,
                    )
                    raise RuntimeError(f"LLM streaming failed: {str(exc)}") from exc

                delay = min(
                    self.llm_retry_max_delay_seconds,
                    self.llm_retry_base_delay_seconds * (2 ** (attempt - 1)),
                )
                logger.warning(
                    "Groq stream failed before token emission (attempt=%s/%s, elapsed=%.2fs). Retrying in %.2fs. Error: %s",
                    attempt,
                    attempts,
                    elapsed,
                    delay,
                    exc,
                )
                if delay > 0:
                    time.sleep(delay)

    def _is_llm_circuit_open(self) -> bool:
        return time.time() < self._llm_circuit_open_until

    def _record_llm_success(self) -> None:
        self._llm_consecutive_failures = 0
        self._llm_circuit_open_until = 0.0

    def _record_llm_failure(self, exc: Exception, operation: str) -> None:
        self._llm_consecutive_failures += 1
        logger.warning(
            "LLM %s failure #%s: %s",
            operation,
            self._llm_consecutive_failures,
            exc,
        )

        if self._llm_consecutive_failures >= self.llm_circuit_breaker_threshold:
            self._llm_circuit_open_until = time.time() + self.llm_circuit_breaker_cooldown_seconds
            logger.warning(
                "LLM circuit breaker opened for %.1fs after %s consecutive failures.",
                self.llm_circuit_breaker_cooldown_seconds,
                self._llm_consecutive_failures,
            )

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

    @staticmethod
    def _build_fallback_answer(question: str, chunks: List[Dict[str, Any]], max_items: int = 3) -> str:
        snippets: List[str] = []
        for chunk in chunks[:max_items]:
            source = chunk.get("source_file") or chunk.get("metadata", {}).get("source_file") or "bilinmeyen"
            idx = chunk.get("chunk_index")
            if idx is None:
                idx = chunk.get("metadata", {}).get("chunk_index", 0)
            text = (chunk.get("text") or "").strip().replace("\n", " ")
            if len(text) > 220:
                text = text[:220].rstrip() + "..."
            snippets.append(f"[{source}:{idx}] {text}")

        if not snippets:
            return "LLM servisine şu an ulaşılamıyor ve kullanılabilir bağlam da bulunamadı."

        joined = "\n- " + "\n- ".join(snippets)
        return (
            "LLM servisine şu an ulaşılamıyor. Aşağıdaki ilgili doküman parçalarını paylaşıyorum:\n"
            f"Soru: {question}\n"
            f"{joined}\n"
            "İstersen bu parçalara göre daha dar bir soru sorabilirsin."
        )

    @staticmethod
    def _build_fallback_summary(chunks: List[Dict[str, Any]], max_items: int = 5, error_msg: str = "") -> str:
        lines: List[str] = []
        for chunk in chunks[:max_items]:
            source = chunk.get("source_file") or chunk.get("metadata", {}).get("source_file") or "bilinmeyen"
            idx = chunk.get("chunk_index")
            if idx is None:
                idx = chunk.get("metadata", {}).get("chunk_index", 0)
            text = (chunk.get("text") or "").strip().replace("\n", " ")
            if len(text) > 180:
                text = text[:180].rstrip() + "..."
            lines.append(f"- [{source}:{idx}] {text}")

        err_prefix = f" ({error_msg})" if error_msg else ""
        if not lines:
            return f"LLM servisine ulaşılamadığı için özet oluşturulamadı.{err_prefix}"
        return f"LLM servisine ulaşılamadı{err_prefix}. Ham özet parçaları:\n" + "\n".join(lines)

    @staticmethod
    def _build_insufficient_context_answer(question: str, retrieval: Dict[str, Any]) -> str:
        coverage = retrieval.get("context_coverage", 0.0)
        confidence = retrieval.get("confidence_score", 0.0)
        return (
            "Bu soruya guvenilir bir yanit uretecek kadar saglam baglam bulunamadi. "
            f"(coverage={coverage}, confidence={confidence}). "
            f"Lutfen soruyu daha daralt veya ilgili dokumanlari yeniden yukle. Soru: {question}"
        )


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    return RagService()
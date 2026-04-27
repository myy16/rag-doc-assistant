import sys
import os

# Allow imports from backend/ without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.rag_service import RagService


class FakeEmbeddingService:
    def __init__(self):
        self.seen_texts = []

    def embed_texts(self, texts):
        self.seen_texts = list(texts)
        return [[float(len(text))] for text in texts]

    def embed_query(self, text):
        return [float(len(text))]


class FakeRetriever:
    def __init__(self):
        self.retrieve_calls = []
        self.fetch_calls = []

    def retrieve(self, query, top_k=5, file_id=None, source_file=None, username=None):
        self.retrieve_calls.append((query, top_k, file_id, source_file, username))
        return [
            {
                "text": "Chunk 1 içerik",
                "metadata": {"source_file": "dosya.pdf", "chunk_index": 0, "char_count": 14},
                "chunk_index": 0,
                "source_file": "dosya.pdf",
                "file_id": "file-1",
            }
        ]

    def fetch_documents(self, file_id=None, source_file=None, username=None):
        self.fetch_calls.append((file_id, source_file, username))
        return [
            {
                "text": "Chunk 1 içerik",
                "metadata": {"source_file": "dosya.pdf", "chunk_index": 0, "char_count": 14},
                "chunk_index": 0,
                "source_file": "dosya.pdf",
                "file_id": "file-1",
            },
            {
                "text": "Chunk 2 içerik",
                "metadata": {"source_file": "dosya.pdf", "chunk_index": 1, "char_count": 14},
                "chunk_index": 1,
                "source_file": "dosya.pdf",
                "file_id": "file-1",
            },
        ]


class FakeVectorStore:
    def __init__(self):
        self.upsert_payloads = []

    def upsert_chunks(self, chunks, embeddings):
        self.upsert_payloads.append((chunks, embeddings))
        return len(chunks)


def build_service(monkeypatch):
    fake_embedding = FakeEmbeddingService()
    fake_retriever = FakeRetriever()
    fake_store = FakeVectorStore()

    monkeypatch.setattr("app.core.rag_service.get_embedding_service", lambda: fake_embedding)
    monkeypatch.setattr("app.core.rag_service.get_retriever", lambda: fake_retriever)
    monkeypatch.setattr("app.core.rag_service.get_vector_store", lambda: fake_store)

    service = RagService()
    return service, fake_embedding, fake_retriever, fake_store


def test_index_chunks_uses_embeddings_and_store(monkeypatch):
    service, fake_embedding, _, fake_store = build_service(monkeypatch)

    chunks = [
        {
            "chunk_id": "c1",
            "text": "Birinci chunk",
            "file_id": "f1",
            "source_file": "rapor.pdf",
            "file_type": "pdf",
            "chunk_index": 0,
            "total_chunks": 2,
            "char_count": 13,
        },
        {
            "chunk_id": "c2",
            "text": "İkinci chunk",
            "file_id": "f1",
            "source_file": "rapor.pdf",
            "file_type": "pdf",
            "chunk_index": 1,
            "total_chunks": 2,
            "char_count": 12,
        },
    ]

    assert service.index_chunks(chunks) == 2
    assert fake_embedding.seen_texts == ["Birinci chunk", "İkinci chunk"]
    assert len(fake_store.upsert_payloads) == 1
    stored_chunks, stored_embeddings = fake_store.upsert_payloads[0]
    assert stored_chunks == chunks
    assert stored_embeddings == [[13.0], [12.0]]


def test_answer_question_returns_sources_without_llm_when_no_context(monkeypatch):
    service, _, fake_retriever, _ = build_service(monkeypatch)
    fake_retriever.retrieve = lambda *args, **kwargs: []

    result = service.answer_question("Soru nedir?")

    assert result["answer"].startswith("İlgili bağlam bulunamadı.")
    assert result["sources"] == []
    assert result["context"] == []
    assert result["model"]


def test_answer_question_builds_context_and_sources(monkeypatch):
    service, _, _, _ = build_service(monkeypatch)
    monkeypatch.setattr(service, "_call_groq", lambda prompt, system_prompt=None: "Yanıt üretildi")

    result = service.answer_question("Belge ne anlatıyor?")

    assert result["answer"] == "Yanıt üretildi"
    assert result["sources"][0]["source_file"] == "dosya.pdf"
    assert result["context"][0]["text"] == "Chunk 1 içerik"


def test_summarize_documents_uses_retrieved_chunks(monkeypatch):
    service, _, fake_retriever, _ = build_service(monkeypatch)
    monkeypatch.setattr(service, "_call_groq", lambda prompt, system_prompt=None: "Özet üretildi")

    result = service.summarize_documents(source_file="dosya.pdf", max_chunks=1)

    assert fake_retriever.fetch_calls == [(None, "dosya.pdf", None)]
    assert result["summary"] == "Özet üretildi"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source_file"] == "dosya.pdf"


def test_llm_circuit_breaker_opens_after_threshold(monkeypatch):
    service, _, _, _ = build_service(monkeypatch)
    service.llm_circuit_breaker_threshold = 2
    service.llm_circuit_breaker_cooldown_seconds = 30

    service._record_llm_failure(RuntimeError("timeout-1"), operation="completion")
    assert not service._is_llm_circuit_open()

    service._record_llm_failure(RuntimeError("timeout-2"), operation="completion")
    assert service._is_llm_circuit_open()


def test_llm_circuit_breaker_resets_after_success(monkeypatch):
    service, _, _, _ = build_service(monkeypatch)
    service.llm_circuit_breaker_threshold = 1
    service.llm_circuit_breaker_cooldown_seconds = 30

    service._record_llm_failure(RuntimeError("timeout"), operation="stream")
    assert service._is_llm_circuit_open()

    service._record_llm_success()
    assert not service._is_llm_circuit_open()
    assert service._llm_consecutive_failures == 0

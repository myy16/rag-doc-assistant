from functools import lru_cache
from typing import List

from app.core.config import EMBEDDING_MODEL_NAME


class EmbeddingService:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. Install backend dependencies first."
                ) from exc

            # Load HF token if set (for private models)
            import os as _os
            from app.core.config import HF_TOKEN
            if HF_TOKEN:
                _os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
            
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        model = self._load_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return vectors.tolist() if hasattr(vectors, "tolist") else list(vectors)

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings else []


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
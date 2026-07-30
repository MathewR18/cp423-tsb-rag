"""Dense retrieval using a pretrained Sentence Transformer."""

from __future__ import annotations

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval_common import INDEX_DIR, corpus_fingerprint, load_chunks, load_config, make_results


class DenseRetriever:
    def __init__(self) -> None:
        self.config = load_config()
        self.chunks = load_chunks()
        self.model_name = self.config["dense_model"]
        self.embedding_path = INDEX_DIR / "dense_embeddings.npy"
        self.metadata_path = INDEX_DIR / "dense_index.json"
        self.model: SentenceTransformer | None = None
        self.embeddings: np.ndarray | None = None

    def _load_model(self) -> SentenceTransformer:
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name, local_files_only=True)
            except OSError:
                print(f"Model {self.model_name} is not cached; downloading it once...")
                self.model = SentenceTransformer(self.model_name)
        return self.model

    def build(self, force: bool = False) -> None:
        fingerprint = corpus_fingerprint()
        if not force and self.embedding_path.exists() and self.metadata_path.exists():
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("corpus_fingerprint") == fingerprint
                and metadata.get("model_name") == self.model_name
                and metadata.get("chunk_count") == len(self.chunks)
            ):
                self.embeddings = np.load(self.embedding_path, mmap_mode="r")
                return

        model = self._load_model()
        passages = [f"{chunk['title']}\n{chunk['text']}" for chunk in self.chunks]
        embeddings = model.encode(
            passages,
            batch_size=int(self.config["dense_batch_size"]),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.save(self.embedding_path, embeddings)
        self.metadata_path.write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "embedding_dimension": int(embeddings.shape[1]),
                    "chunk_count": len(self.chunks),
                    "normalized": True,
                    "similarity": "cosine via normalized dot product",
                    "corpus_fingerprint": fingerprint,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.embeddings = embeddings

    def search(self, query: str, top_k: int = 5):
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if self.embeddings is None:
            self.build()
        assert self.embeddings is not None

        model = self._load_model()
        query_text = f"{self.config['dense_query_prefix']}{query}"
        query_embedding = model.encode(
            [query_text], convert_to_numpy=True, normalize_embeddings=True
        )[0].astype(np.float32)
        scores = np.asarray(self.embeddings @ query_embedding)
        count = min(top_k, len(scores))
        candidates = np.argpartition(-scores, count - 1)[:count]
        ranked = candidates[np.argsort(-scores[candidates], kind="stable")]
        return make_results(
            self.chunks,
            ranked.tolist(),
            scores[ranked].tolist(),
        )

"""Shared data loading, fingerprints, and result formatting for retrieval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    chunk_id: str
    document_id: str
    title: str
    report_url: str
    text: str

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "score": round(self.score, 6),
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "report_url": self.report_url,
            "text": self.text,
        }


def load_config() -> dict:
    return json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing chunks file: {CHUNKS_PATH}")
    return [
        json.loads(line)
        for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
        if line
    ]


def corpus_fingerprint() -> str:
    digest = hashlib.sha256()
    with CHUNKS_PATH.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_results(chunks: list[dict], indices: list[int], scores: list[float]) -> list[SearchResult]:
    results: list[SearchResult] = []
    for rank, (index, score) in enumerate(zip(indices, scores), start=1):
        chunk = chunks[index]
        results.append(
            SearchResult(
                rank=rank,
                score=float(score),
                chunk_id=chunk["chunk_id"],
                document_id=chunk["document_id"],
                title=chunk["title"],
                report_url=chunk["report_url"],
                text=chunk["text"],
            )
        )
    return results

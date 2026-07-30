"""Classical BM25 retrieval over the prepared TSB chunks."""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

from retrieval_common import INDEX_DIR, corpus_fingerprint, load_chunks, load_config, make_results


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


class BM25Retriever:
    def __init__(self, index_path: Path | None = None) -> None:
        self.config = load_config()
        self.chunks = load_chunks()
        self.index_path = index_path or INDEX_DIR / "bm25_index.pkl"
        self.index: dict | None = None

    def build(self, force: bool = False) -> None:
        fingerprint = corpus_fingerprint()
        if not force and self.index_path.exists():
            with self.index_path.open("rb") as source:
                candidate = pickle.load(source)
            if candidate.get("corpus_fingerprint") == fingerprint:
                self.index = candidate
                return

        document_lengths: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, chunk in enumerate(self.chunks):
            terms = tokenize(f"{chunk['document_id']} {chunk['title']} {chunk['text']}")
            document_lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                postings[term].append((index, frequency))

        count = len(self.chunks)
        idf = {
            term: math.log(1.0 + (count - len(items) + 0.5) / (len(items) + 0.5))
            for term, items in postings.items()
        }
        self.index = {
            "corpus_fingerprint": fingerprint,
            "document_count": count,
            "average_document_length": sum(document_lengths) / count,
            "document_lengths": document_lengths,
            "postings": dict(postings),
            "idf": idf,
            "k1": float(self.config["bm25_k1"]),
            "b": float(self.config["bm25_b"]),
        }
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("wb") as output:
            pickle.dump(self.index, output, protocol=pickle.HIGHEST_PROTOCOL)

    def search(self, query: str, top_k: int = 5):
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if self.index is None:
            self.build()
        assert self.index is not None

        scores: dict[int, float] = defaultdict(float)
        query_terms = Counter(tokenize(query))
        k1 = self.index["k1"]
        b = self.index["b"]
        average_length = self.index["average_document_length"]
        lengths = self.index["document_lengths"]

        for term, query_frequency in query_terms.items():
            term_idf = self.index["idf"].get(term)
            if term_idf is None:
                continue
            for document_index, frequency in self.index["postings"][term]:
                normalization = frequency + k1 * (
                    1.0 - b + b * lengths[document_index] / average_length
                )
                scores[document_index] += (
                    term_idf * (frequency * (k1 + 1.0) / normalization) * query_frequency
                )

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return make_results(
            self.chunks,
            [index for index, _ in ranked],
            [score for _, score in ranked],
        )

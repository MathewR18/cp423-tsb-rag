"""Small retrieval smoke test using known corpus facts."""

from __future__ import annotations

from bm25_retriever import BM25Retriever
from dense_retriever import DenseRetriever


BM25_CASES = [
    ("What aircraft was involved in occurrence A23Q0145?", "A23Q0145"),
    ("Where did occurrence A23P0091 take place?", "A23P0091"),
    ("Which company operated the aircraft in occurrence A22P0067?", "A22P0067"),
]

DENSE_CASES = [
    (
        "Which report describes a Beech King Air striking a snow windrow during landing at Wemindji?",
        "A23Q0145",
    ),
    (
        "Which report describes Air Nootka DHC-2 Beaver C-FZVP colliding with terrain at Gold River?",
        "A23P0091",
    ),
    (
        "Which report describes a Conair Air Tractor losing engine power and making a forced landing near Cranbrook?",
        "A22P0067",
    ),
]


def verify(name: str, retriever, cases) -> None:
    for query, expected_document in cases:
        results = retriever.search(query, top_k=5)
        retrieved = [result.document_id for result in results]
        assert expected_document in retrieved, (
            f"{name} missed {expected_document} for {query!r}; got {retrieved}"
        )
        rank = retrieved.index(expected_document) + 1
        print(f"PASS {name}: {expected_document} at rank {rank}")


def main() -> None:
    verify("BM25", BM25Retriever(), BM25_CASES)
    verify("dense", DenseRetriever(), DENSE_CASES)


if __name__ == "__main__":
    main()

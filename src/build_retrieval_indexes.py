"""Build BM25 and dense retrieval indexes."""

from __future__ import annotations

import argparse

from bm25_retriever import BM25Retriever
from dense_retriever import DenseRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild existing indexes")
    args = parser.parse_args()

    print("Building BM25 index...")
    BM25Retriever().build(force=args.force)
    print("BM25 index ready.")

    print("Building dense index...")
    DenseRetriever().build(force=args.force)
    print("Dense index ready.")


if __name__ == "__main__":
    main()

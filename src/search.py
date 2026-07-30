"""Search the prepared corpus with BM25 or dense retrieval."""

from __future__ import annotations

import argparse
import json
import sys

from bm25_retriever import BM25Retriever
from dense_retriever import DenseRetriever
from retrieval_common import load_config


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["bm25", "dense"])
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=int(config["default_top_k"]))
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    retriever = BM25Retriever() if args.method == "bm25" else DenseRetriever()
    results = retriever.search(args.query, args.top_k)

    if args.json:
        print(json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2))
        return

    print(f"\nMethod: {args.method}\nQuery: {args.query}\n")
    for result in results:
        preview = " ".join(result.text.split())[:450]
        print(
            f"#{result.rank}  score={result.score:.4f}  {result.chunk_id}\n"
            f"Document: {result.document_id} — {result.title}\n"
            f"Source: {result.report_url}\n"
            f"Preview: {preview}\n"
        )


if __name__ == "__main__":
    main()

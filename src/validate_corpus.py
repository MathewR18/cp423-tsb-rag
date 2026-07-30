"""Validate the prepared corpus before retrieval experiments."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    config = json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))
    documents = read_jsonl(PROJECT_ROOT / "data" / "processed" / "documents.jsonl")
    chunks = read_jsonl(PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl")
    target_count = int(config["document_limit"])

    assert len(documents) == target_count, (len(documents), target_count)
    document_ids = [document["document_id"] for document in documents]
    assert len(document_ids) == len(set(document_ids)), "Duplicate document IDs"
    assert all(document["text"].strip() for document in documents), "Empty document text"
    assert all(document["report_url"].startswith("https://www.tsb.gc.ca/") for document in documents)

    for document in documents:
        html_path = PROJECT_ROOT / document["raw_html_path"]
        assert html_path.is_file(), f"Missing raw HTML: {html_path}"
        digest = hashlib.sha256(html_path.read_bytes()).hexdigest()
        assert digest == document["raw_html_sha256"], f"Hash mismatch: {document['document_id']}"

    known_ids = set(document_ids)
    chunk_counts: Counter[str] = Counter()
    chunk_ids: set[str] = set()
    for chunk in chunks:
        assert chunk["document_id"] in known_ids, f"Orphan chunk: {chunk['chunk_id']}"
        assert chunk["chunk_id"] not in chunk_ids, f"Duplicate chunk: {chunk['chunk_id']}"
        assert chunk["text"].strip(), f"Empty chunk: {chunk['chunk_id']}"
        chunk_ids.add(chunk["chunk_id"])
        chunk_counts[chunk["document_id"]] += 1

    assert set(chunk_counts) == known_ids, "One or more documents have no chunks"
    print(
        f"VALID: {len(documents)} documents, {len(chunks)} unique chunks, "
        f"{sum(document['word_count'] for document in documents):,} source words"
    )


if __name__ == "__main__":
    main()

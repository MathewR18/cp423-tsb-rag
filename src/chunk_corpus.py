"""Split cleaned TSB reports into deterministic retrieval chunks."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.json"
DOCUMENT_PATH = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"
CHUNK_DIR = PROJECT_ROOT / "data" / "chunks"


def paragraph_windows(text: str, target_words: int, overlap_words: int) -> list[str]:
    paragraphs = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\n\s*\n", text)]
    paragraphs = [part for part in paragraphs if part]
    chunks: list[str] = []
    current: list[str] = []
    current_count = 0

    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) > target_words:
            if current:
                chunks.append("\n\n".join(current))
                current, current_count = [], 0
            step = max(1, target_words - overlap_words)
            for start in range(0, len(words), step):
                window = words[start : start + target_words]
                if window:
                    chunks.append(" ".join(window))
                if start + target_words >= len(words):
                    break
            continue

        if current and current_count + len(words) > target_words:
            chunks.append("\n\n".join(current))
            overlap: list[str] = []
            overlap_count = 0
            for previous in reversed(current):
                previous_count = len(previous.split())
                if overlap and overlap_count + previous_count > overlap_words:
                    break
                overlap.insert(0, previous)
                overlap_count += previous_count
            current = overlap
            current_count = overlap_count

        current.append(paragraph)
        current_count += len(words)

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    target_words = int(config["chunk_size_words"])
    overlap_words = int(config["chunk_overlap_words"])
    if overlap_words >= target_words:
        raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")

    documents = [json.loads(line) for line in DOCUMENT_PATH.read_text(encoding="utf-8").splitlines() if line]
    chunks: list[dict] = []
    document_chunk_counts: Counter[str] = Counter()

    for document in documents:
        windows = paragraph_windows(document["text"], target_words, overlap_words)
        for index, text in enumerate(windows, start=1):
            chunk_id = f"{document['document_id']}_chunk_{index:04d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document["document_id"],
                    "chunk_index": index,
                    "title": document["title"],
                    "occurrence_date": document["occurrence_date"],
                    "release_date": document["release_date"],
                    "report_url": document["report_url"],
                    "text": text,
                    "word_count": len(text.split()),
                }
            )
            document_chunk_counts[document["document_id"]] += 1

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    chunk_path = CHUNK_DIR / "chunks.jsonl"
    with chunk_path.open("w", encoding="utf-8", newline="\n") as output:
        for chunk in chunks:
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    stats = {
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "total_document_words": sum(document["word_count"] for document in documents),
        "total_chunk_words_with_overlap": sum(chunk["word_count"] for chunk in chunks),
        "configured_chunk_size_words": target_words,
        "configured_chunk_overlap_words": overlap_words,
        "minimum_chunks_per_document": min(document_chunk_counts.values()),
        "maximum_chunks_per_document": max(document_chunk_counts.values()),
        "average_chunks_per_document": round(len(chunks) / len(documents), 2),
    }
    (CHUNK_DIR / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

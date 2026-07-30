"""Validate the manually verified gold-standard evaluation set."""

from __future__ import annotations

import json
from collections import Counter

from retrieval_common import CHUNKS_PATH, PROJECT_ROOT


def main() -> None:
    evaluation_path = PROJECT_ROOT / "data" / "evaluation" / "gold_questions.jsonl"
    questions = [json.loads(line) for line in evaluation_path.read_text(encoding="utf-8").splitlines() if line]
    chunks = [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if line]
    known_chunk_ids = {chunk["chunk_id"] for chunk in chunks}

    assert len(questions) >= 10, "At least 10 questions are required"
    assert len({item["question_id"] for item in questions}) == len(questions), "Duplicate question IDs"
    assert all(item["manually_verified"] is True for item in questions), "Every question must be verified"
    counts = Counter(item["type"] for item in questions)
    assert counts["multi-hop"] >= 2, "At least 2 multi-hop questions are required"
    assert counts["unanswerable"] >= 2, "At least 2 unanswerable questions are required"

    for item in questions:
        ground_truth = item["ground_truth_chunks"]
        if item["type"] == "unanswerable":
            assert not ground_truth, f"{item['question_id']} should have no ground-truth chunks"
            assert item["reference_answer"].rstrip(".").casefold() == "i don't know"
        else:
            assert ground_truth, f"{item['question_id']} needs ground-truth chunks"
            missing = set(ground_truth) - known_chunk_ids
            assert not missing, f"{item['question_id']} references missing chunks: {missing}"
        if item["type"] == "multi-hop":
            documents = {chunk_id.split("_chunk_")[0] for chunk_id in ground_truth}
            assert len(ground_truth) >= 2 or len(documents) >= 2

    print(f"VALID: {len(questions)} manually verified questions — {dict(counts)}")


if __name__ == "__main__":
    main()

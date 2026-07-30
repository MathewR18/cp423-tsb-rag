"""Run the gold evaluation set against BM25-RAG and dense-RAG."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

from bm25_retriever import BM25Retriever
from dense_retriever import DenseRetriever
from rag import build_prompt, generate, normalize_abstention, validate_citations
from retrieval_common import PROJECT_ROOT, load_config


EVALUATION_PATH = PROJECT_ROOT / "data" / "evaluation" / "gold_questions.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation"


def load_questions() -> list[dict]:
    return [
        json.loads(line)
        for line in EVALUATION_PATH.read_text(encoding="utf-8").splitlines()
        if line
    ]


def retrieval_metrics(retrieved_ids: list[str], ground_truth: list[str]) -> dict:
    if not ground_truth:
        return {
            "hit_at_k": None,
            "recall_at_k": None,
            "reciprocal_rank": None,
            "all_ground_truth_retrieved": None,
            "first_relevant_rank": None,
        }
    ground_set = set(ground_truth)
    retrieved_set = set(retrieved_ids)
    relevant_ranks = [
        rank for rank, chunk_id in enumerate(retrieved_ids, start=1) if chunk_id in ground_set
    ]
    first_rank = min(relevant_ranks) if relevant_ranks else None
    intersection = ground_set & retrieved_set
    return {
        "hit_at_k": bool(intersection),
        "recall_at_k": len(intersection) / len(ground_set),
        "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        "all_ground_truth_retrieved": ground_set.issubset(retrieved_set),
        "first_relevant_rank": first_rank,
    }


def run_case(question: dict, method: str, retriever, config: dict, top_k: int, generate_answers: bool) -> dict:
    retrieval_started = time.perf_counter()
    results = retriever.search(question["question"], top_k=top_k)
    retrieval_seconds = time.perf_counter() - retrieval_started
    retrieved_ids = [result.chunk_id for result in results]
    metrics = retrieval_metrics(retrieved_ids, question["ground_truth_chunks"])

    raw_answer = None
    answer = None
    citation_validation = None
    generation_seconds = None
    prompt = None
    if generate_answers:
        prompt = build_prompt(question["question"], results)
        raw_answer, _, generation_seconds = generate(config, prompt)
        answer = normalize_abstention(raw_answer)
        citation_validation = validate_citations(answer, set(retrieved_ids))

    return {
        "question_id": question["question_id"],
        "question_type": question["type"],
        "question": question["question"],
        "reference_answer": question["reference_answer"],
        "ground_truth_chunks": question["ground_truth_chunks"],
        "retrieval_method": method,
        "top_k": top_k,
        "retrieved_chunks": [result.to_dict() for result in results],
        **metrics,
        "prompt": prompt,
        "raw_ollama_answer": raw_answer,
        "answer": answer,
        "citation_validation": citation_validation,
        "retrieval_seconds": round(retrieval_seconds, 4),
        "generation_seconds": round(generation_seconds, 4) if generation_seconds is not None else None,
        "manual_answer_correct": None,
        "manual_citations_supported": None,
        "manual_unanswerable_handling_correct": None,
        "manual_notes": "",
    }


def mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def summarize(records: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    by_method: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_method[record["retrieval_method"]].append(record)

    for method, method_records in by_method.items():
        answerable = [record for record in method_records if record["question_type"] != "unanswerable"]
        unanswerable = [record for record in method_records if record["question_type"] == "unanswerable"]
        generated = [record for record in method_records if record["answer"] is not None]
        summary[method] = {
            "question_count": len(method_records),
            "answerable_question_count": len(answerable),
            "hit_rate_at_k": mean([float(record["hit_at_k"]) for record in answerable]),
            "mean_recall_at_k": mean([record["recall_at_k"] for record in answerable]),
            "mean_reciprocal_rank": mean([record["reciprocal_rank"] for record in answerable]),
            "all_ground_truth_retrieved_rate": mean(
                [float(record["all_ground_truth_retrieved"]) for record in answerable]
            ),
            "citation_format_pass_rate": mean(
                [float(record["citation_validation"]["format_valid"]) for record in generated]
            ),
            "unanswerable_abstention_rate": mean(
                [
                    float(record["citation_validation"]["abstained"])
                    for record in unanswerable
                    if record["citation_validation"] is not None
                ]
            ),
            "mean_retrieval_seconds": mean([record["retrieval_seconds"] for record in method_records]),
            "mean_generation_seconds": mean(
                [record["generation_seconds"] for record in generated]
            ),
        }
    return summary


def write_outputs(records: list[dict], summary: dict, config: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUTPUT_DIR / "evaluation_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_path = OUTPUT_DIR / "evaluation_results.csv"
    fields = [
        "question_id",
        "question_type",
        "retrieval_method",
        "question",
        "reference_answer",
        "ground_truth_chunks",
        "retrieved_chunk_ids",
        "hit_at_k",
        "recall_at_k",
        "reciprocal_rank",
        "all_ground_truth_retrieved",
        "answer",
        "citation_format_valid",
        "abstained",
        "manual_answer_correct",
        "manual_citations_supported",
        "manual_unanswerable_handling_correct",
        "manual_notes",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for record in records:
            validation = record["citation_validation"] or {}
            writer.writerow(
                {
                    "question_id": record["question_id"],
                    "question_type": record["question_type"],
                    "retrieval_method": record["retrieval_method"],
                    "question": record["question"],
                    "reference_answer": record["reference_answer"],
                    "ground_truth_chunks": "|".join(record["ground_truth_chunks"]),
                    "retrieved_chunk_ids": "|".join(
                        item["chunk_id"] for item in record["retrieved_chunks"]
                    ),
                    "hit_at_k": record["hit_at_k"],
                    "recall_at_k": record["recall_at_k"],
                    "reciprocal_rank": record["reciprocal_rank"],
                    "all_ground_truth_retrieved": record["all_ground_truth_retrieved"],
                    "answer": record["answer"],
                    "citation_format_valid": validation.get("format_valid"),
                    "abstained": validation.get("abstained"),
                    "manual_answer_correct": "",
                    "manual_citations_supported": "",
                    "manual_unanswerable_handling_correct": "",
                    "manual_notes": "",
                }
            )

    summary_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": config["ollama_model"],
        "temperature": config["generation_temperature"],
        "seed": config["random_seed"],
        "top_k": config["rag_top_k"],
        "metrics": summary,
    }
    (OUTPUT_DIR / "evaluation_summary.json").write_text(
        json.dumps(summary_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true", help="Skip Ollama generation")
    parser.add_argument("--top-k", type=int, default=int(config["rag_top_k"]))
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    questions = load_questions()
    retrievers = {"bm25": BM25Retriever(), "dense": DenseRetriever()}
    records: list[dict] = []
    total = len(questions) * len(retrievers)
    completed = 0
    for method, retriever in retrievers.items():
        for question in questions:
            completed += 1
            print(f"[{completed:02d}/{total}] {method} {question['question_id']}...", flush=True)
            record = run_case(
                question,
                method,
                retriever,
                config,
                args.top_k,
                generate_answers=not args.retrieval_only,
            )
            records.append(record)
            answer_preview = record["answer"] or "retrieval only"
            print(f"  {answer_preview[:140]}", flush=True)

    summary = summarize(records)
    write_outputs(records, summary, config)
    print("\nEvaluation complete.")
    print(json.dumps(summary, indent=2))
    print(f"Results: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

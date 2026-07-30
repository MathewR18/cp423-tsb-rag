"""Run an evidence-grounded RAG query with BM25 or dense retrieval and Ollama."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from bm25_retriever import BM25Retriever
from dense_retriever import DenseRetriever
from retrieval_common import PROJECT_ROOT, load_config


CITATION_PATTERN = re.compile(r"\[([A-Z0-9]+_chunk_\d{4})\]")


def build_prompt(question: str, results) -> str:
    context_blocks = []
    for result in results:
        context_blocks.append(
            f"[{result.chunk_id}]\n"
            f"Document: {result.document_id}\n"
            f"Title: {result.title}\n"
            f"Source: {result.report_url}\n"
            f"Passage: {result.text}"
        )
    context = "\n\n---\n\n".join(context_blocks)
    return f"""You are an evidence-grounded question-answering assistant.

Answer the question using only the provided context.

Rules:
1. Do not use outside knowledge.
2. Every answer other than "I don't know." MUST contain at least one inline citation using the exact chunk ID, for example [A23Q0145_chunk_0001].
3. Cite only chunks included in the provided context.
4. If the context does not contain enough information to answer the question, respond exactly: I don't know.
5. Keep the answer concise and direct.
6. Before responding, verify that your final answer ends with a supporting citation in square brackets.
7. A cited chunk must directly state the answer details; do not cite a chunk merely because it discusses the same event.
8. When several chunks are relevant, cite the chunk that most specifically contains the requested names, models, registrations, dates, or locations.
9. Answer every requested component that is explicitly supported by the context; do not omit the aircraft model when both model and registration are available.
10. If the answer combines identity details from one chunk with event details from another, cite both chunks.
11. Never add a citation or any other text to "I don't know." The entire response must be exactly those three words and punctuation.

Required answer format for an answerable question:
<concise answer> [exact_chunk_id]

Required answer format for an unanswerable question:
I don't know.

Context:
{context}

Question: {question}

Answer:"""


def generate(config: dict, prompt: str) -> tuple[str, dict, float]:
    payload = {
        "model": config["ollama_model"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(config["generation_temperature"]),
            "seed": int(config["random_seed"]),
            "num_predict": int(config["generation_max_tokens"]),
        },
    }
    started = time.perf_counter()
    try:
        response = requests.post(config["ollama_url"], json=payload, timeout=180)
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            "Could not contact Ollama. Confirm it is running with: ollama list"
        ) from error
    elapsed = time.perf_counter() - started
    body = response.json()
    answer = body.get("response", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty response")
    return answer, body, elapsed


def validate_citations(answer: str, retrieved_chunk_ids: set[str]) -> dict:
    citations = CITATION_PATTERN.findall(answer)
    invalid = sorted(set(citations) - retrieved_chunk_ids)
    answer_without_citations = CITATION_PATTERN.sub("", answer).strip()
    abstained = answer_without_citations.rstrip(".").casefold() == "i don't know"
    exact_abstention = answer.strip().casefold() in {"i don't know", "i don't know."}
    return {
        "citations": citations,
        "invalid_citations": invalid,
        "all_citations_were_retrieved": not invalid,
        "has_citation_when_answered": abstained or bool(citations),
        "abstained": abstained,
        "exact_abstention_format": exact_abstention if abstained else None,
        "format_valid": not invalid and (exact_abstention if abstained else bool(citations)),
    }


def normalize_abstention(raw_answer: str) -> str:
    """Enforce the required abstention string while preserving raw output in the run log."""
    without_citations = CITATION_PATTERN.sub("", raw_answer).strip()
    if without_citations.rstrip(".").casefold() == "i don't know":
        return "I don't know."
    return raw_answer


def save_run(record: dict) -> Path:
    output_dir = PROJECT_ROOT / "results" / "rag_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_method = record["retrieval_method"]
    path = output_dir / f"{timestamp}_{safe_method}.json"
    counter = 2
    while path.exists():
        path = output_dir / f"{timestamp}_{safe_method}_{counter}.json"
        counter += 1
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["bm25", "dense"])
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=int(config["rag_top_k"]))
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("question cannot be empty")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    retriever = BM25Retriever() if args.method == "bm25" else DenseRetriever()
    retrieval_started = time.perf_counter()
    results = retriever.search(args.question, top_k=args.top_k)
    retrieval_seconds = time.perf_counter() - retrieval_started
    prompt = build_prompt(args.question, results)
    raw_answer, ollama_metadata, generation_seconds = generate(config, prompt)
    answer = normalize_abstention(raw_answer)
    retrieved_ids = {result.chunk_id for result in results}
    validation = validate_citations(answer, retrieved_ids)

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "question": args.question,
        "retrieval_method": args.method,
        "top_k": args.top_k,
        "retrieved_chunks": [result.to_dict() for result in results],
        "prompt": prompt,
        "raw_ollama_answer": raw_answer,
        "answer": answer,
        "citation_validation": validation,
        "settings": {
            "model": config["ollama_model"],
            "temperature": config["generation_temperature"],
            "seed": config["random_seed"],
            "maximum_generated_tokens": config["generation_max_tokens"],
        },
        "timing_seconds": {
            "retrieval": round(retrieval_seconds, 4),
            "generation": round(generation_seconds, 4),
        },
        "ollama_metrics": {
            key: ollama_metadata.get(key)
            for key in (
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            )
        },
    }

    print(f"\nQuestion: {args.question}")
    print(f"Method: {args.method}")
    print(f"\nAnswer:\n{answer}\n")
    print(
        "Citation format check: "
        + ("PASS" if validation["format_valid"] else "REVIEW NEEDED")
    )
    print("Retrieved chunks: " + ", ".join(result.chunk_id for result in results))

    if not args.no_save:
        path = save_run(record)
        print(f"Saved run: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

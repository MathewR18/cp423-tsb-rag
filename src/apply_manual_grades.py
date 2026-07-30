"""Merge confirmed human judgments into evaluation outputs and summarize them."""

from __future__ import annotations

import csv
import json
from collections import defaultdict

from retrieval_common import PROJECT_ROOT


OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation"


def rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def main() -> None:
    result_path = OUTPUT_DIR / "evaluation_results.jsonl"
    judgments = json.loads((OUTPUT_DIR / "manual_judgments.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line]

    for record in records:
        judgment = judgments[record["retrieval_method"]][record["question_id"]]
        record["manual_answer_correct"] = judgment["answer_correct"]
        record["manual_citations_supported"] = judgment["citations_supported"]
        record["manual_unanswerable_handling_correct"] = judgment[
            "unanswerable_handling_correct"
        ]
        record["manual_notes"] = judgment["notes"]

    with result_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_path = OUTPUT_DIR / "evaluation_results.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
        fields = source.seek(0) or list(rows[0].keys())
    fieldnames = list(rows[0].keys())
    record_lookup = {
        (record["retrieval_method"], record["question_id"]): record for record in records
    }
    for row in rows:
        record = record_lookup[(row["retrieval_method"], row["question_id"])]
        row["manual_answer_correct"] = record["manual_answer_correct"]
        row["manual_citations_supported"] = record["manual_citations_supported"]
        row["manual_unanswerable_handling_correct"] = record[
            "manual_unanswerable_handling_correct"
        ]
        row["manual_notes"] = record["manual_notes"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_method: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_method[record["retrieval_method"]].append(record)
    summary: dict[str, dict] = {}
    for method, method_records in by_method.items():
        answerable = [r for r in method_records if r["question_type"] != "unanswerable"]
        citation_judged = [
            r["manual_citations_supported"]
            for r in method_records
            if r["manual_citations_supported"] is not None
        ]
        unanswerable = [
            r["manual_unanswerable_handling_correct"]
            for r in method_records
            if r["question_type"] == "unanswerable"
        ]
        summary[method] = {
            "overall_answer_accuracy": rate(
                [r["manual_answer_correct"] for r in method_records]
            ),
            "answerable_question_accuracy": rate(
                [r["manual_answer_correct"] for r in answerable]
            ),
            "citation_support_rate_when_judged": rate(citation_judged),
            "unanswerable_handling_accuracy": rate(unanswerable),
            "correct_answers": sum(r["manual_answer_correct"] for r in method_records),
            "total_questions": len(method_records),
        }

    (OUTPUT_DIR / "human_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

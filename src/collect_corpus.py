"""Download and clean a deterministic TSB aviation investigation corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = RAW_DIR / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/127.0 Safari/537.36 "
        "CP423-academic-RAG-corpus/1.0"
    ),
    "Accept-Language": "en-CA,en;q=0.9",
}


@dataclass(frozen=True)
class IndexRecord:
    document_id: str
    status: str
    occurrence_date: str
    occurrence_summary: str
    release_date: str
    index_url: str
    report_url: str


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def normalize_space(value: str) -> str:
    return re.sub(r"[\t\r\f\v ]+", " ", value).strip()


def normalize_text(value: str) -> str:
    lines = [normalize_space(line) for line in value.replace("\u00a0", " ").splitlines()]
    output: list[str] = []
    for line in lines:
        if not line:
            if output and output[-1] != "":
                output.append("")
            continue
        output.append(line)
    return "\n".join(output).strip()


def get_html(session: requests.Session, url: str, timeout: int, attempts: int = 6) -> bytes:
    """Fetch HTML with bounded exponential backoff for temporary server failures."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, timeout=timeout)
            if 400 <= response.status_code < 500 and response.status_code not in (408, 429):
                raise FileNotFoundError(f"Permanent HTTP {response.status_code} for {url}")
            response.raise_for_status()
            if "html" not in response.headers.get("Content-Type", "").lower():
                raise ValueError(
                    f"Expected HTML from {url}, got {response.headers.get('Content-Type')}"
                )
            return response.content
        except FileNotFoundError:
            raise
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt == attempts:
                break
            wait_seconds = min(30, 2 ** (attempt - 1))
            print(
                f"Temporary fetch failure ({attempt}/{attempts}) for {url}; "
                f"retrying in {wait_seconds}s: {error}",
                flush=True,
            )
            time.sleep(wait_seconds)
    raise RuntimeError(f"Unable to fetch {url} after {attempts} attempts") from last_error


def final_report_url(index_url: str) -> str:
    """Convert newer investigation-summary paths to final-report paths."""
    return index_url.replace("/eng/enquetes-investigations/", "/eng/rapports-reports/")


def parse_index(html: bytes, index_url: str, config: dict) -> list[IndexRecord]:
    soup = BeautifulSoup(html, "html.parser")
    records: list[IndexRecord] = []
    start = date.fromisoformat(config["start_date"])
    end = date.fromisoformat(config["end_date"])

    for row in soup.select("tbody tr"):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 5:
            continue
        status = normalize_space(cells[0].get_text(" ", strip=True))
        if status.casefold() != "completed":
            continue
        link = cells[1].find("a", href=True)
        time_tag = cells[2].find("time")
        if link is None or time_tag is None:
            continue
        document_id = normalize_space(link.get_text(" ", strip=True)).upper()
        if not re.fullmatch(r"A\d{2}[A-Z]\d{4}", document_id):
            continue
        occurrence_date = time_tag.get("datetime", "")[:10]
        try:
            occurrence_day = date.fromisoformat(occurrence_date)
        except ValueError:
            continue
        if not start <= occurrence_day <= end:
            continue
        source_url = urljoin(index_url, link["href"])
        records.append(
            IndexRecord(
                document_id=document_id,
                status=status,
                occurrence_date=occurrence_date,
                occurrence_summary=normalize_space(cells[3].get_text(" | ", strip=True)),
                release_date=normalize_space(cells[4].get_text(" ", strip=True)),
                index_url=source_url,
                report_url=final_report_url(source_url),
            )
        )

    records.sort(key=lambda item: (item.occurrence_date, item.document_id), reverse=True)
    if len(records) < int(config["document_limit"]):
        raise RuntimeError(
            f"Expected at least {config['document_limit']} eligible index records, found {len(records)}"
        )
    return records


def extract_report(html: bytes) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")
    if main is None:
        raise ValueError("Report page has no <main> element")

    for selector in (
        "nav",
        "script",
        "style",
        "noscript",
        "form",
        ".gc-pg-hlpfl",
        ".pagedetails",
        ".wb-share",
        ".gc-prtts",
    ):
        for node in main.select(selector):
            node.decompose()

    heading = main.find("h1")
    title = normalize_space(heading.get_text(" ", strip=True)) if heading else ""

    blocks: list[str] = []
    for node in main.find_all(["h1", "h2", "h3", "h4", "h5", "p", "li", "caption"]):
        if node.find_parent(["nav", "footer"]):
            continue
        text = normalize_space(node.get_text(" ", strip=True))
        if text and (not blocks or blocks[-1] != text):
            blocks.append(text)

    cleaned = normalize_text("\n\n".join(blocks))
    if len(cleaned.split()) < 100:
        raise ValueError(f"Extracted report is unexpectedly short ({len(cleaned.split())} words)")
    return title, cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Download files again")
    args = parser.parse_args()
    config = load_config()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    index_path = RAW_DIR / "index.html"
    timeout = int(config["request_timeout_seconds"])
    delay = float(config["request_delay_seconds"])

    session = requests.Session()
    session.headers.update(HEADERS)

    if args.refresh or not index_path.exists():
        index_html = get_html(session, config["index_url"], timeout)
        index_path.write_bytes(index_html)
    else:
        index_html = index_path.read_bytes()

    records = parse_index(index_html, config["index_url"], config)
    target_count = int(config["document_limit"])
    documents: list[dict] = []
    exclusions: list[dict] = []

    for record in records:
        if len(documents) >= target_count:
            break
        html_path = REPORT_DIR / f"{record.document_id}.html"
        try:
            if args.refresh or not html_path.exists() or html_path.stat().st_size < 1000:
                html = get_html(session, record.report_url, timeout)
                html_path.write_bytes(html)
                time.sleep(delay)
            else:
                html = html_path.read_bytes()
            title, text = extract_report(html)
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            exclusions.append(
                {
                    "document_id": record.document_id,
                    "report_url": record.report_url,
                    "reason": str(error),
                }
            )
            print(f"[SKIP] {record.document_id}: {error}", flush=True)
            continue

        documents.append(
            {
                **asdict(record),
                "title": title,
                "text": text,
                "word_count": len(text.split()),
                "raw_html_path": html_path.relative_to(PROJECT_ROOT).as_posix(),
                "raw_html_sha256": hashlib.sha256(html).hexdigest(),
            }
        )
        print(
            f"[{len(documents):03d}/{target_count}] {record.document_id}: {len(text.split()):,} words",
            flush=True,
        )

    if len(documents) != target_count:
        raise RuntimeError(f"Collected {len(documents)} valid reports; expected {target_count}")

    document_path = PROCESSED_DIR / "documents.jsonl"
    with document_path.open("w", encoding="utf-8", newline="\n") as output:
        for document in documents:
            output.write(json.dumps(document, ensure_ascii=False) + "\n")

    manifest_path = PROCESSED_DIR / "manifest.csv"
    manifest_fields = [
        "document_id",
        "status",
        "occurrence_date",
        "release_date",
        "title",
        "word_count",
        "report_url",
        "raw_html_path",
        "raw_html_sha256",
    ]
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=manifest_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(documents)

    (PROCESSED_DIR / "exclusions.json").write_text(
        json.dumps(exclusions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Saved {len(documents)} documents to {document_path}")


if __name__ == "__main__":
    main()

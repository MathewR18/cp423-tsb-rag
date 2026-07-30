# CP423 TSB Aviation RAG Corpus

Created by Mathew Rantisi for CP423.

This project builds and compares two retrieval-augmented generation systems over Transportation Safety Board of Canada aviation reports. One system uses BM25 retrieval and the other uses dense retrieval with a pretrained Sentence Transformer. Both systems send retrieved passages to Llama 3.2 through Ollama and request answers with inline chunk citations.

## Main results

The systems were tested on the same manually verified set of 10 questions, including factoid, multi-hop, and unanswerable questions. This produced 20 total RAG experiments.

| Metric | BM25-RAG | Dense-RAG |
| --- | ---: | ---: |
| Overall answer accuracy | 70% | 40% |
| Answerable-question accuracy | 62.5% | 25% |
| Retrieval hit rate at 5 | 50% | 50% |
| Mean recall at 5 | 50% | 45.83% |
| Mean reciprocal rank | 0.40 | 0.25 |
| Unanswerable handling accuracy | 100% | 100% |

BM25 performed better overall on this evaluation set. Many questions included exact investigation IDs, which worked well with keyword matching. Dense retrieval was useful for descriptive questions but had more trouble with report IDs and questions that needed evidence from multiple reports.

## Corpus definition

- Source: Transportation Safety Board of Canada aviation investigation index
- Language: English
- Status: Completed investigations only
- Occurrence dates: 2010-01-01 through 2024-12-31
- Selection: Sort by occurrence date (newest first), then investigation ID; keep the first 300 available full HTML reports
- Snapshot cutoff: A14W0046, occurrence date 2014-05-31

Active investigations and non-report pages are excluded. Each final investigation report is stored as its original HTML and as cleaned text with traceable metadata.

## Folder structure

```text
data/
  raw/                 Original final-report HTML pages
  processed/           Cleaned report text and document metadata
  chunks/              Retrieval-ready chunks
  evaluation/          The 10 verified questions and reference answers
results/
  evaluation/          Automatic metrics, human grading, and error analysis
  rag_runs/            Saved outputs from individual RAG runs
src/
  collect_corpus.py    Selects, downloads, and cleans 300 reports
  chunk_corpus.py      Splits cleaned reports into chunks
  validate_corpus.py   Checks counts, IDs, hashes, metadata, and chunk tracing
  bm25_retriever.py    Classical BM25 retrieval
  dense_retriever.py   Sentence Transformer retrieval
  build_retrieval_indexes.py  Builds both indexes
  search.py            Searches either index
config.json            Reproducible corpus and chunk settings
prepare_corpus.ps1     Runs the complete preparation pipeline
requirements.txt       Pinned Python packages
```

The downloaded HTML reports, processed corpus, chunks, and generated indexes are not committed because they are large and can be recreated with the included scripts. The verified evaluation set and final experiment results are included.

## Setup

Install Python 3.11 or newer, then run:

```powershell
python -m pip install -r requirements.txt
```

## Prepare the corpus

```powershell
.\prepare_corpus.ps1
```

The command creates:

- `data/raw/index.html`: a snapshot of the source index;
- `data/raw/reports/*.html`: 300 original final-report pages;
- `data/processed/documents.jsonl`: cleaned text and metadata;
- `data/processed/manifest.csv`: a human-readable document inventory;
- `data/processed/exclusions.json`: indexed entries skipped because no valid final-report HTML was available;
- `data/chunks/chunks.jsonl`: retrieval chunks with source tracing;
- `data/chunks/stats.json`: corpus statistics.

Re-running the command reuses valid downloaded files. Use `python src/collect_corpus.py --refresh` to download fresh copies.

## Reproducibility notes

- The source URL and SHA-256 hash are recorded for every HTML document.
- Selection and chunk settings are stored in `config.json`.
- Document IDs are the official TSB investigation numbers.
- Chunk IDs use the format `INVESTIGATION_ID_chunk_0001`.
- Raw HTML may change on the source website. Keep the downloaded snapshot used for the final experiments.

The TSB remains the source and owner of its published material. Keep source URLs and attribution with redistributed data. If redistribution is undesirable, publish the downloader and manifest instead of committing the raw HTML.

## Build retrieval indexes

The classical system uses BM25 with `k1=1.5` and `b=0.75`. The dense system uses the pretrained `BAAI/bge-small-en-v1.5` Sentence Transformer, 384-dimensional normalized embeddings, and cosine similarity. All settings are recorded in `config.json`.

```powershell
python src/build_retrieval_indexes.py
```

The first dense build downloads the pretrained model and embeds every chunk. Later runs reuse the saved index.

## Search

```powershell
python src/search.py bm25 "What aircraft was involved in occurrence A23Q0145?" --top-k 5
python src/search.py dense "What factors can cause a runway excursion?" --top-k 5
```

Add `--json` for machine-readable results. Both systems return the same fields: rank, score, chunk ID, document ID, title, source URL, and chunk text.

## Retrieval smoke test

```powershell
python src/test_retrieval.py
```

## Run retrieval-augmented generation

Ollama must be running with `llama3.2:3b` installed. Run either retrieval method with the same grounded-generation settings:

```powershell
python src/rag.py bm25 "What aircraft model and registration were involved in occurrence A23Q0145?"
python src/rag.py dense "Which aircraft struck a snow windrow during landing at Wemindji?"
```

Each run retrieves five chunks, asks Llama to answer only from that evidence, validates the inline chunk IDs, and saves the complete prompt, response, settings, timings, and retrieved passages under `results/rag_runs/`.

## Run the full evaluation

Run all 10 manually verified questions through both systems (20 total RAG runs):

```powershell
python src/evaluate.py
```

The command saves detailed JSONL, a CSV for manual answer/citation judgments, and an automatic metric summary under `results/evaluation/`. Use `python src/evaluate.py --retrieval-only` to test retrieval without calling Ollama.

## Evaluation files

- `data/evaluation/gold_questions.csv` contains the 10 questions, reference answers, question types, and ground-truth chunk IDs.
- `results/evaluation/evaluation_results.csv` contains the 20 BM25 and dense RAG runs.
- `results/evaluation/evaluation_summary.json` contains the automatic retrieval and citation-format metrics.
- `results/evaluation/human_evaluation_summary.json` contains the manually graded answer results.
- `results/evaluation/error_analysis.md` explains the main successes, failures, limitations, and possible improvements.

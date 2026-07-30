$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

python src/collect_corpus.py
python src/chunk_corpus.py
python src/validate_corpus.py

Write-Host "Corpus preparation complete."
Write-Host "Documents: data/processed/documents.jsonl"
Write-Host "Chunks:    data/chunks/chunks.jsonl"
Write-Host "Statistics:data/chunks/stats.json"

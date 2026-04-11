# Gemma — Theory-vs-Article Analysis System

> Multi-pass article-vs-theory relevance analysis runner with a local research browser.

## Overview
Gemma analyzes a fixed theory text (Morozov's "Critique of Techno-Feudal Reason") against
journalistic and academic inputs using local LLMs (Gemma via Ollama) or Claude API. Each
article goes through a 4-pass pipeline producing structured JSON artifacts. A local webapp
provides a read-only browser for inspecting all analysis results.

## Tech Stack
- Python 3.13+, hatchling build
- Analysis: Ollama (Gemma 4 31B) or Anthropic Claude
- Webapp: FastAPI + SQLite + Jinja2 + HTMX
- Dependencies managed via uv

## Quick Reference
- Analysis: `gemma-article-theory-analysis` (see run.sh / run_academic.sh)
- Ingest: `python -m webapp.ingest`
- Serve: `python -m webapp.app` or `start`
- Test: `pytest`

## Architecture Notes
- Analysis pipeline: `src/gemma/article_theory_analysis.py`
- Webapp: `webapp/` (FastAPI + Jinja2, SQLite for normalized data)
- Two output profiles: journalistic (`outputs/`) and academic (`academic_outputs/`)
- Theory preprocessing in `_theory/` under each output tree

## Documentation
- Feature inventory: `docs/FEATURES.md` (read on demand)
- Change history: `docs/CHANGELOG.md` (read on demand)

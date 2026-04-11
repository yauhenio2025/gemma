# Feature Inventory

> Auto-maintained by Claude Code. Last updated: 2026-04-11

## Analysis Pipeline

### Multi-Pass Article-Theory Analysis
- **Status**: Active
- **Description**: 4-pass analysis pipeline comparing articles against a fixed theory text
- **Entry Points**:
  - `src/gemma/article_theory_analysis.py:1` - Core analysis engine
- **Dependencies**: Ollama (local) or Anthropic API, beautifulsoup4, pypdf
- **Added**: Pre-existing

## Webapp

### SQLite Ingestion
- **Status**: Active
- **Description**: CLI command to ingest JSON analysis artifacts into normalized SQLite database
- **Entry Points**:
  - `webapp/ingest.py:1` - Ingestion CLI
  - `webapp/schema.py:1` - Database schema definitions
- **Added**: 2026-04-11

### Research Browser
- **Status**: Active
- **Description**: FastAPI webapp for browsing theory analysis artifacts
- **Entry Points**:
  - `webapp/app.py:1` - FastAPI application
  - `webapp/templates/` - Jinja2 templates
- **Dependencies**: FastAPI, Jinja2, uvicorn, aiosqlite
- **Added**: 2026-04-11

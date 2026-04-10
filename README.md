# Gemma

Standalone local pipeline for testing whether a fixed theory text is actually supported, challenged, or ignored by later articles.

The current workflow is built for:

- one theory corpus in `article_theory_workspace/nlr/`
- many comparison texts in `article_theory_workspace/others/`
- multi-pass analysis through a local Ollama model such as `gemma4-31b-bf16-256k`

The runner is intentionally skeptical. It does not treat loose overlap about AI, jobs, regulation, growth, or China as substantive engagement unless an article actually bears on the theory's mechanism or categories.

The theory preprocessing stage can be run once with Claude Opus 4.6, committed, and reused by article-only runs on another machine.

## Run

```bash
uv run gemma-article-theory-analysis --workspace article_theory_workspace
```

Or:

```bash
./article_theory_workspace/run.sh
```

Preprocess the theory only with Anthropic Claude Opus 4.6 and adaptive max-effort thinking:

```bash
uv run gemma-article-theory-analysis --provider anthropic --stage theory --overwrite
```

Run only article analysis against the committed preprocessed theory files:

```bash
uv run gemma-article-theory-analysis --stage articles
```

## Test

```bash
uv run --extra dev pytest
```

## Notes

- `article_theory_workspace/` contains the portable working folder.
- The current source corpus under `article_theory_workspace/nlr/` and `article_theory_workspace/others/` is versioned in git.
- Extracted text caches and generated article outputs stay out of git by default.
- The preprocessed theory outputs under `article_theory_workspace/outputs/_theory/` are versioned so remote machines can skip the theory stage.

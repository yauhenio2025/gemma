# Gemma

Standalone local pipeline for testing whether a fixed theory text is actually supported, challenged, or ignored by later articles.

The current workflow is built for:

- one theory corpus in `article_theory_workspace/nlr/`
- many comparison texts in `article_theory_workspace/others/`
- multi-pass analysis through a local Ollama model such as `gemma4-31b-bf16-256k`

The runner is intentionally skeptical. It does not treat loose overlap about AI, jobs, regulation, growth, or China as substantive engagement unless an article actually bears on the theory's mechanism or categories.

## Run

```bash
uv run gemma-article-theory-analysis --workspace article_theory_workspace
```

Or:

```bash
./article_theory_workspace/run.sh
```

## Test

```bash
uv run --extra dev pytest
```

## Notes

- `article_theory_workspace/` contains the portable working folder.
- The current source corpus under `article_theory_workspace/nlr/` and `article_theory_workspace/others/` is versioned in git.
- Extracted text caches and generated outputs stay out of git by default.

# Gemma

Standalone local pipeline for testing whether a fixed theory text is actually supported, challenged, or ignored by later comparison texts.

The current workflow now supports two profiles:

- `journalistic`: shorter news, policy, and op-ed style texts under `article_theory_workspace/others/`
- `academic`: long-form academic papers under `article_theory_workspace/articles/`

Both profiles use the same fixed theory corpus under `article_theory_workspace/nlr/`, but they keep separate caches and outputs so academic-paper runs do not overwrite the journalistic results.

The runner is intentionally skeptical. It does not treat loose overlap about AI, jobs, regulation, growth, or China as substantive engagement unless a comparison text actually bears on the theory's mechanism or categories. The academic profile is less news-centric: later passes rely on a rich paper map instead of rereading the full paper each time.

The theory preprocessing stage can be run once with Claude Opus 4.6, committed, and reused by article-only runs on another machine.

## Run

```bash
uv run gemma-article-theory-analysis --workspace article_theory_workspace
```

Or:

```bash
./article_theory_workspace/run.sh
```

Run the academic-paper profile:

```bash
./article_theory_workspace/run_academic.sh
```

Preprocess the theory only with Anthropic Claude Opus 4.6 and adaptive max-effort thinking:

```bash
uv run gemma-article-theory-analysis --provider anthropic --stage theory --overwrite
```

Run only article analysis against the committed preprocessed theory files:

```bash
uv run gemma-article-theory-analysis --stage articles
```

Run only academic-paper analysis against the academic theory preprocessing:

```bash
uv run gemma-article-theory-analysis --profile academic --stage articles
```

For the calibrated second run, overwrite the prior article outputs and fail loudly if any article cannot be parsed:

```bash
uv run gemma-article-theory-analysis --stage articles --overwrite --parallel 2
```

The runner now retries malformed JSON responses and writes either:

- `article_theory_workspace/outputs/failures.json` for journalistic runs
- `article_theory_workspace/academic_outputs/failures.json` for academic runs

before exiting nonzero if any comparison text still fails.

## Test

```bash
uv run --extra dev pytest
```

## Notes

- `article_theory_workspace/` contains the portable working folder.
- The current source corpus under `article_theory_workspace/nlr/`, `article_theory_workspace/others/`, and `article_theory_workspace/articles/` is versioned in git.
- Extracted text caches and generated journalistic/academic outputs stay out of git by default.
- The preprocessed theory outputs under `article_theory_workspace/outputs/_theory/` are versioned so remote machines can skip the theory stage.
- The academic theory outputs under `article_theory_workspace/academic_outputs/_theory/` can also be versioned for the same reason.

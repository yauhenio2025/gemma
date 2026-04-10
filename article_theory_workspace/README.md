# Article Theory Workspace

This folder is a portable workspace for testing whether later articles actually support, challenge, or fail to engage a fixed theory text.

Structure:

- `nlr/`: put the theory source here. One NLR piece is the expected default.
- `others/`: put all comparison articles here.
- `cache/`: extracted plain-text versions of the inputs.
- `outputs/`: one folder per article plus an overall `index.md`.

Run from the repo root:

```bash
uv run gemma-article-theory-analysis --workspace article_theory_workspace
```

Or run the wrapper in this folder:

```bash
./run.sh
```

Run the theory preprocessing stage only:

```bash
./run.sh --provider anthropic --stage theory --overwrite
```

Run only the article stage using committed theory preprocessing:

```bash
./run.sh --stage articles
```

Run the calibrated second pass over the articles:

```bash
./run.sh --stage articles --overwrite --parallel 2
```

If any article still fails after automatic JSON repair and retries, the command exits nonzero and writes `outputs/failures.json`.

Useful flags:

```bash
./run.sh --overwrite
./run.sh --model gemma4-31b-bf16-256k
./run.sh --ollama-url http://localhost:11434
./run.sh --max-reconcile-rounds 3
```

What the pipeline does:

1. Extract the theory's core claims from the NLR text.
2. Build a strict rubric designed to reject superficial relevance.
3. Map each article on its own terms.
4. Run a skeptical relevance audit against the theory map.
5. Run a counter-audit to detect grade inflation or missed points.
6. Produce a final judgment.
7. Run extra reconcile passes when the earlier passes disagree or confidence stays low.

Supported input types:

- `.txt`
- `.md`
- `.html`
- `.docx`
- `.pdf`

The workspace keeps generated cache and article outputs out of git.

The current source corpus in `nlr/` and `others/` is versioned so the seeded materials travel with the repo.

The preprocessed theory outputs in `outputs/_theory/` are also versioned so another machine can run only `--stage articles`.

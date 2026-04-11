# Article Theory Workspace

This folder is a portable workspace for testing whether later comparison texts actually support, challenge, or fail to engage a fixed theory text.

Structure:

- `nlr/`: put the theory source here. One NLR piece is the expected default.
- `others/`: put journalistic comparison articles here.
- `articles/`: put longer academic comparison papers here.
- `cache/`: extracted plain-text versions of the inputs.
- `outputs/`: journalistic analysis outputs.
- `academic_outputs/`: academic-paper analysis outputs.

Run from the repo root:

```bash
uv run gemma-article-theory-analysis --workspace article_theory_workspace
```

Or run the wrapper in this folder:

```bash
./run.sh
```

Run the academic profile wrapper:

```bash
./run_academic.sh
```

Run the corpus-level claim brief wrapper:

```bash
./run_claim_briefs.sh --provider anthropic --overwrite
```

Run the theory preprocessing stage only:

```bash
./run.sh --provider anthropic --stage theory --overwrite
```

Run only the article stage using committed theory preprocessing:

```bash
./run.sh --stage articles
```

Run only the academic-paper stage using academic theory preprocessing:

```bash
./run_academic.sh --stage articles
```

Run only the new theory-implications pass against already analyzed academic papers:

```bash
./run_academic.sh --stage implications --implication-min-verdict marginal
```

Run the calibrated second pass over the articles:

```bash
./run.sh --stage articles --overwrite --parallel 2
```

If any journalistic article still fails after automatic JSON repair and retries, the command exits nonzero and writes `outputs/failures.json`.

If any academic paper still fails, it writes `academic_outputs/failures.json`.

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
3. Map each comparison text on its own terms.
4. Run a skeptical relevance audit against the theory map.
5. Run a counter-audit to detect grade inflation or missed points.
6. Produce a final judgment.
7. Run extra reconcile passes when the earlier passes disagree or confidence stays low.
8. Optionally run a theory-implications pass on relevant/marginal items to extract confirmations, qualifications, extensions, revision pressure, and follow-up questions.
9. Optionally synthesize all `05_theory_implications.json` artifacts into one corpus-level claim brief per theory claim.

The academic profile differs in two ways:

1. It reads from `articles/` and writes to `academic_outputs/`.
2. It builds a richer first-pass paper digest so later passes can rely on the paper map instead of repeatedly sending the full paper text.

Supported input types:

- `.txt`
- `.md`
- `.html`
- `.docx`
- `.pdf`

The workspace keeps generated cache and journalistic/academic outputs out of git.

The current source corpus in `nlr/`, `others/`, and `articles/` is versioned so the seeded materials travel with the repo.

The preprocessed theory outputs in `outputs/_theory/` are also versioned so another machine can run only `--stage articles`.

Academic theory preprocessing can be versioned under `academic_outputs/_theory/` for the same reason.

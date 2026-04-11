from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import textwrap
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

SUPPORTED_EXTENSIONS = {".docx", ".htm", ".html", ".md", ".pdf", ".txt"}
STRENGTH_ORDER = {"none": 0, "weak": 1, "moderate": 2, "strong": 3}
IMPLICATION_VERDICTS = {"none", "relevant", "marginal", "contextual", "irrelevant"}
SUSPECT_NOISE_PATTERNS = (
    "oflipstick",
    "andEastermost",
    "funerals of profit rates",
    "themidterms",
    "C://",
    "いくつ",
    "работает",
    "relação",
)
DEFAULT_WORKSPACE = Path("article_theory_workspace")
DEFAULT_OLLAMA_MODEL = "gemma4-31b-bf16-256k"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-6"
JOURNALISTIC_SYSTEM_PROMPT = """You are evaluating whether a non-academic text bears on a political-economy theory essay.

Use calibrated relevance, not maximal strictness.

Distinguish:
- direct evidence that supports or challenges a specific theoretical mechanism
- indirect or illustrative evidence that makes one section of the theory more concrete
- contextual relevance that is useful background but not evidence
- genuine irrelevance

Do not inflate contextual overlap into direct proof, but do not erase useful indirect relevance. A policy story, industrial-policy story, capital-expenditure story, sanctions story, labor-displacement story, or platform-competition story can be relevant even if it cannot adjudicate rent vs. profit by itself.

Output valid JSON only."""

ACADEMIC_SYSTEM_PROMPT = """You are evaluating whether an academic text bears on a political-economy theory essay.

Use calibrated relevance, not maximal strictness.

Distinguish:
- direct theoretical engagement with the essay's categories or mechanism
- empirical case-study evidence that supports or challenges one part of the theory
- adjacent conceptual or historical literature that helps situate the theory without deciding it
- genuine irrelevance

Academic papers can matter through explicit argument, literature intervention, conceptual clarification, historical reconstruction, labor-process analysis, or firm/case evidence. Do not mistake shared vocabulary for engagement, but do not dismiss a paper just because it is not written about the essay itself.

Output valid JSON only."""

DEFAULT_SYSTEM_PROMPT = JOURNALISTIC_SYSTEM_PROMPT


@dataclass(frozen=True, slots=True)
class AnalysisProfileConfig:
    key: str
    system_prompt: str
    comparison_subdir: str
    cache_subdir: str
    outputs_subdir: str
    comparison_label: str
    document_label: str
    index_title: str
    default_max_comparison_chars: int


PROFILE_CONFIGS = {
    "journalistic": AnalysisProfileConfig(
        key="journalistic",
        system_prompt=JOURNALISTIC_SYSTEM_PROMPT,
        comparison_subdir="others",
        cache_subdir="others",
        outputs_subdir="outputs",
        comparison_label="comparison articles",
        document_label="article",
        index_title="Article Theory Analysis Index",
        default_max_comparison_chars=90_000,
    ),
    "academic": AnalysisProfileConfig(
        key="academic",
        system_prompt=ACADEMIC_SYSTEM_PROMPT,
        comparison_subdir="articles",
        cache_subdir="articles",
        outputs_subdir="academic_outputs",
        comparison_label="academic comparison papers",
        document_label="paper",
        index_title="Academic Theory Analysis Index",
        default_max_comparison_chars=160_000,
    ),
}


@dataclass(slots=True)
class Document:
    path: Path
    slug: str
    text: str
    extracted_path: Path


class JsonResponseError(ValueError):
    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


def profile_config(profile: str) -> AnalysisProfileConfig:
    try:
        return PROFILE_CONFIGS[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown analysis profile: {profile}") from exc


class OllamaClient:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.1,
        num_ctx: int | None = 131072,
        timeout_seconds: int = 3600,
        max_json_retries: int = 2,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout_seconds = timeout_seconds
        self.max_json_retries = max_json_retries

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return chat_json_with_retries(self._chat_text, messages, self.max_json_retries)

    def _chat_text(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system_prompt}, *messages],
            "stream": True,
            "format": "json",
            "options": {
                "temperature": self.temperature,
            },
        }
        if self.num_ctx is not None:
            payload["options"]["num_ctx"] = self.num_ctx
        request = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                content_parts: list[str] = []
                for line in response:
                    chunk = json.loads(line.decode("utf-8"))
                    content_parts.append(chunk.get("message", {}).get("content", ""))
                    if chunk.get("done"):
                        break
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Unable to reach Ollama at {self.base_url}. Start the server or pass --ollama-url."
            ) from exc
        return "".join(content_parts).strip()


class AnthropicClient:
    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str = "https://api.anthropic.com",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 32_000,
        effort: str = "max",
        thinking_type: str = "adaptive",
        timeout_seconds: int = 1800,
        max_retries: int = 3,
        max_json_retries: int = 2,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {api_key_env}; export it before using the Anthropic provider.")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.effort = effort
        self.thinking_type = thinking_type
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_json_retries = max_json_retries

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return chat_json_with_retries(self._chat_text, messages, self.max_json_retries)

    def _chat_text(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system_prompt,
            "messages": messages,
            "output_config": {
                "effort": self.effort,
            },
        }
        if self.thinking_type != "disabled":
            payload["thinking"] = {"type": self.thinking_type}

        body = self._post_messages(payload)
        text = "".join(
            block.get("text", "")
            for block in body.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        if not text:
            content_types = [
                block.get("type", "unknown")
                for block in body.get("content", [])
                if isinstance(block, dict)
            ]
            raise RuntimeError(f"Anthropic response contained no text block; content types: {content_types}")
        return text

    def _post_messages(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        for attempt in range(1, self.max_retries + 1):
            request = urllib.request.Request(
                url=f"{self.base_url}/v1/messages",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                    "x-api-key": self.api_key,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code in {408, 409, 429, 500, 502, 503, 529} and attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(
                    f"Anthropic request failed with HTTP {exc.code}: {body[:500]}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(
                    f"Unable to reach Anthropic at {self.base_url}. Check network access."
                ) from exc
        raise RuntimeError("Anthropic request failed after retries.")


class TheoryArticleRunner:
    def __init__(
        self,
        workspace: Path,
        client: OllamaClient,
        analysis_profile: str = "journalistic",
        comparison_subdir: str | None = None,
        cache_subdir: str | None = None,
        outputs_subdir: str | None = None,
        overwrite: bool = False,
        max_theory_chars: int = 180_000,
        max_article_chars: int | None = None,
        implication_min_verdict: str = "none",
        min_confidence: float = 0.7,
        max_reconcile_rounds: int = 2,
        max_step_retries: int = 2,
        allow_failures: bool = False,
    ) -> None:
        self.workspace = workspace
        self.client = client
        self.profile = profile_config(analysis_profile)
        self.analysis_profile = self.profile.key
        self.comparison_subdir = comparison_subdir or self.profile.comparison_subdir
        self.cache_subdir = cache_subdir or self.profile.cache_subdir
        self.outputs_subdir = outputs_subdir or self.profile.outputs_subdir
        self.overwrite = overwrite
        self.max_theory_chars = max_theory_chars
        self.max_article_chars = max_article_chars or self.profile.default_max_comparison_chars
        if implication_min_verdict not in IMPLICATION_VERDICTS:
            raise ValueError(f"Unknown implication_min_verdict: {implication_min_verdict}")
        self.implication_min_verdict = implication_min_verdict
        self.min_confidence = min_confidence
        self.max_reconcile_rounds = max_reconcile_rounds
        self.max_step_retries = max_step_retries
        self.allow_failures = allow_failures
        self.cache_dir = workspace / "cache"
        self.theory_dir = workspace / "nlr"
        self.comparison_dir = workspace / self.comparison_subdir
        self.comparison_cache_dir = self.cache_dir / self.cache_subdir
        self.outputs_dir = workspace / self.outputs_subdir

    def run(self, stage: str = "all") -> dict[str, Any]:
        ensure_workspace(
            self.workspace,
            comparison_subdir=self.comparison_subdir,
            outputs_subdir=self.outputs_subdir,
            cache_subdir=self.cache_subdir,
        )
        theory_documents = load_documents(self.theory_dir, self.cache_dir / "nlr", self.max_theory_chars)
        if not theory_documents:
            raise RuntimeError(f"No supported theory files found in {self.theory_dir}")

        theory_outputs_dir = self.outputs_dir / "_theory"
        theory_outputs_dir.mkdir(parents=True, exist_ok=True)
        theory_map_path = theory_outputs_dir / "01_theory_map.json"
        theory_rubric_path = theory_outputs_dir / "02_theory_rubric.json"
        if stage in {"all", "theory"}:
            theory_map = self._load_or_run(
                theory_map_path,
                self._build_theory_map,
                theory_documents,
            )
            theory_rubric = self._load_or_run(
                theory_rubric_path,
                self._build_theory_rubric,
                theory_map,
            )
        elif stage in {"articles", "implications"}:
            theory_map = self._load_required(theory_map_path, "preprocessed theory map")
            theory_rubric = self._load_required(theory_rubric_path, "preprocessed theory rubric")
        else:
            raise ValueError(f"Unknown stage: {stage}")

        if stage == "theory":
            index = self._write_theory_index(theory_documents, theory_map_path, theory_rubric_path)
            return {
                "workspace": str(self.workspace),
                "model": self.client.model,
                "stage": stage,
                "theory_files": [str(doc.path) for doc in theory_documents],
                "theory_index_path": str(index),
                "theory_map_path": str(theory_map_path),
                "theory_rubric_path": str(theory_rubric_path),
            }

        other_documents = load_documents(
            self.comparison_dir,
            self.comparison_cache_dir,
            self.max_article_chars,
        )
        if not other_documents:
            raise RuntimeError(f"No supported comparison files found in {self.comparison_dir}")

        max_workers = int(os.environ.get("GEMMA_PARALLEL", "3"))
        analyses: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        worker = (
            lambda article: self._refresh_implications(article, theory_map, theory_rubric)
            if stage == "implications"
            else self._analyze_article(article, theory_map, theory_rubric)
        )
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(worker, article): article for article in other_documents}
                for future in as_completed(futures):
                    article = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            analyses.append(result)
                    except Exception as exc:
                        failures.append(
                            {
                                "article_path": str(article.path),
                                "article_slug": article.slug,
                                "error": str(exc),
                            }
                        )
                        print(f"FAILED: {article.slug}: {exc}")
        else:
            for article in other_documents:
                try:
                    result = worker(article)
                    if result is not None:
                        analyses.append(result)
                except Exception as exc:
                    failures.append(
                        {
                            "article_path": str(article.path),
                            "article_slug": article.slug,
                            "error": str(exc),
                        }
                    )
                    print(f"FAILED: {article.slug}: {exc}")

        if stage == "implications":
            if failures and not self.allow_failures:
                raise RuntimeError(
                    f"{len(failures)} implication refresh(es) failed. See logs for details."
                )
            return {
                "workspace": str(self.workspace),
                "model": self.client.model,
                "stage": stage,
                "theory_files": [str(doc.path) for doc in theory_documents],
                "implication_count": len(analyses),
                "failed_article_count": len(failures),
            }

        index = self._write_index(theory_documents, analyses, failures)
        if failures and not self.allow_failures:
            raise RuntimeError(
                f"{len(failures)} article(s) failed. See {self.outputs_dir / 'failures.json'}."
            )
        return {
            "workspace": str(self.workspace),
            "model": self.client.model,
            "stage": stage,
            "theory_files": [str(doc.path) for doc in theory_documents],
            "article_count": len(analyses),
            "failed_article_count": len(failures),
            "index_path": str(index),
        }

    def _is_academic(self) -> bool:
        return self.analysis_profile == "academic"

    def _document_label(self) -> str:
        return self.profile.document_label

    def _comparison_label(self) -> str:
        return self.profile.comparison_label

    def _later_pass_context(self, article: Document, article_map: dict[str, Any]) -> str:
        if self._is_academic():
            return (
                "The full paper was already read in pass 1. For the later passes, use the structured paper map "
                "below as the primary digest of the paper's argument, method, evidence, and theoretical "
                "intervention. Do not invent details beyond that map."
            )
        return f"{self._document_label().title()} text:\n{article.text}"

    def _implications_enabled(self) -> bool:
        return self.implication_min_verdict != "none"

    def _should_generate_implications(self, verdict: str | None) -> bool:
        if not self._implications_enabled() or not verdict:
            return False
        return verdict_rank(verdict) <= verdict_rank(self.implication_min_verdict)

    def _build_theory_map(self, theory_documents: list[Document]) -> dict[str, Any]:
        theory_corpus = render_document_bundle(theory_documents, "NLR theory corpus")
        if self._is_academic():
            prompt = textwrap.dedent(
                f"""
                Read the theory corpus and build an expansive map for evaluating later academic texts against it.

                Constraints:
                - Capture the strongest 4 to 7 core claims.
                - Also capture secondary themes that an academic paper may usefully develop, refine, illustrate, or contest.
                - Prefer precise claims about mechanism, categories, evidence, method, or periodization.
                - Identify where a later paper could bear on the theory through direct conceptual argument, empirical case-study evidence, labor-process analysis, historical reconstruction, or critique of rival frameworks.
                - Distinguish direct theoretical engagement from adjacent literature that is useful but not decisive.
                - Include false positives only as warnings against overclaiming, not as a reason to erase all indirect relevance.

                Return JSON with exactly these keys:
                {{
                  "thesis": "string",
                  "core_claims": [
                    {{
                      "id": "C1",
                      "claim": "string",
                      "why_it_matters": "string",
                      "support_requirements": ["string"],
                      "challenge_requirements": ["string"],
                      "indirect_relevance_hooks": ["string"],
                      "false_positive_matches": ["string"]
                    }}
                  ],
                  "secondary_themes": [
                    {{
                      "id": "T1",
                      "theme": "string",
                      "why_articles_may_matter": "string",
                      "typical_article_signals": ["string"]
                    }}
                  ],
                  "article_relevance_hooks": [
                    {{
                      "hook": "string",
                      "counts_as": "contextual|illustrative|direct",
                      "notes": "string"
                    }}
                  ],
                  "conceptual_boundaries": [
                    {{
                      "boundary": "string",
                      "explanation": "string"
                    }}
                  ],
                  "relevance_red_flags": ["string"],
                  "open_questions": ["string"]
                }}

                {theory_corpus}
                """
            ).strip()
        else:
            prompt = textwrap.dedent(
                f"""
                Read the theory corpus and build an expansive map for evaluating later news articles against it.

                Constraints:
                - Capture the strongest 4 to 7 core claims.
                - Also capture secondary themes that a news article may usefully illustrate without proving or disproving the theory.
                - Prefer precise claims about mechanism, categories, evidence, or method.
                - Separate direct evidentiary hooks from contextual or illustrative hooks.
                - Avoid the previous over-strict mistake: do not require a news article to adjudicate rent vs. profit before it can be called contextually or marginally relevant.
                - Include false positives only as warnings against overclaiming, not as a reason to erase all indirect relevance.

                Return JSON with exactly these keys:
                {{
                  "thesis": "string",
                  "core_claims": [
                    {{
                      "id": "C1",
                      "claim": "string",
                      "why_it_matters": "string",
                      "support_requirements": ["string"],
                      "challenge_requirements": ["string"],
                      "indirect_relevance_hooks": ["string"],
                      "false_positive_matches": ["string"]
                    }}
                  ],
                  "secondary_themes": [
                    {{
                      "id": "T1",
                      "theme": "string",
                      "why_articles_may_matter": "string",
                      "typical_article_signals": ["string"]
                    }}
                  ],
                  "article_relevance_hooks": [
                    {{
                      "hook": "string",
                      "counts_as": "contextual|illustrative|direct",
                      "notes": "string"
                    }}
                  ],
                  "conceptual_boundaries": [
                    {{
                      "boundary": "string",
                      "explanation": "string"
                    }}
                  ],
                  "relevance_red_flags": ["string"],
                  "open_questions": ["string"]
                }}

                {theory_corpus}
                """
            ).strip()
        return self.client.chat_json([{"role": "user", "content": prompt}])

    def _build_theory_rubric(self, theory_map: dict[str, Any]) -> dict[str, Any]:
        if self._is_academic():
            prompt = textwrap.dedent(
                f"""
                Build a calibrated evaluation rubric for judging whether a later academic paper bears on the theory map below.

                Constraints:
                - Do not default every paper to irrelevance just because it is not written as a direct reply to the essay.
                - Distinguish contextual relevance, illustrative relevance, direct empirical relevance, and direct theoretical engagement.
                - A paper may count as relevant through conceptual argument, literature intervention, historical reconstruction, case-study evidence, labor-process analysis, or critique of monopoly/competition/rent/platform-capitalism frameworks.
                - Make both anti-grade-inflation and anti-false-negative rules explicit.
                - Shared vocabulary alone is not enough.

                Return JSON with exactly these keys:
                {{
                  "relevance_tiers": [
                    {{
                      "label": "irrelevant|contextual|marginal|relevant",
                      "definition": "string",
                      "recommended_use": "ignore|mention_in_passing|use_as_minor_update|use_as_substantive_evidence"
                    }}
                  ],
                  "relevance_questions": ["string"],
                  "what_counts_as_real_evidence": ["string"],
                  "what_counts_as_contextual_or_illustrative_relevance": ["string"],
                  "what_does_not_count": ["string"],
                  "support_strength_scale": [
                    {{
                      "label": "none|weak|moderate|strong",
                      "definition": "string"
                    }}
                  ],
                  "challenge_strength_scale": [
                    {{
                      "label": "none|weak|moderate|strong",
                      "definition": "string"
                    }}
                  ],
                  "anti_grade_inflation_rule": "string",
                  "anti_false_negative_rule": "string",
                  "default_verdict_rule": "string"
                }}

                Theory map:
                {json.dumps(theory_map, ensure_ascii=False, indent=2)}
                """
            ).strip()
        else:
            prompt = textwrap.dedent(
                f"""
                Build a calibrated evaluation rubric for judging whether a later article bears on the theory map below.

                Constraints:
                - Do not default every article to irrelevance just because it lacks direct proof.
                - Distinguish contextual relevance, illustrative relevance, and direct evidence.
                - Make both anti-grade-inflation and anti-false-negative rules explicit.
                - Articles about state industrial policy, geopolitics, data centers, AI infrastructure, platform lock-in, talent flows, sanctions, labor displacement, and public legitimacy may be contextually or marginally relevant even when they do not adjudicate rent vs. profit.

                Return JSON with exactly these keys:
                {{
                  "relevance_tiers": [
                    {{
                      "label": "irrelevant|contextual|marginal|relevant",
                      "definition": "string",
                      "recommended_use": "ignore|mention_in_passing|use_as_minor_update|use_as_substantive_evidence"
                    }}
                  ],
                  "relevance_questions": ["string"],
                  "what_counts_as_real_evidence": ["string"],
                  "what_counts_as_contextual_or_illustrative_relevance": ["string"],
                  "what_does_not_count": ["string"],
                  "support_strength_scale": [
                    {{
                      "label": "none|weak|moderate|strong",
                      "definition": "string"
                    }}
                  ],
                  "challenge_strength_scale": [
                    {{
                      "label": "none|weak|moderate|strong",
                      "definition": "string"
                    }}
                  ],
                  "anti_grade_inflation_rule": "string",
                  "anti_false_negative_rule": "string",
                  "default_verdict_rule": "string"
                }}

                Theory map:
                {json.dumps(theory_map, ensure_ascii=False, indent=2)}
                """
            ).strip()
        return self.client.chat_json([{"role": "user", "content": prompt}])

    def _analyze_article(
        self,
        article: Document,
        theory_map: dict[str, Any],
        theory_rubric: dict[str, Any],
    ) -> dict[str, Any]:
        article_dir = self.outputs_dir / article.slug
        article_dir.mkdir(parents=True, exist_ok=True)

        article_map = self._load_or_run(
            article_dir / "01_article_map.json",
            self._build_article_map,
            article,
        )
        relevance_audit = self._load_or_run(
            article_dir / "02_relevance_audit.json",
            self._build_relevance_audit,
            article,
            theory_map,
            theory_rubric,
            article_map,
        )
        counter_audit = self._load_or_run(
            article_dir / "03_counter_audit.json",
            self._build_counter_audit,
            article,
            theory_map,
            theory_rubric,
            article_map,
            relevance_audit,
        )
        final_judgment = self._load_or_run(
            article_dir / "04_final_judgment.json",
            self._build_final_judgment,
            article,
            theory_map,
            theory_rubric,
            article_map,
            relevance_audit,
            counter_audit,
        )

        final_result = final_judgment
        reconcile_rounds = 0
        while reconcile_rounds < self.max_reconcile_rounds and needs_reconcile(
            relevance_audit, counter_audit, final_result, self.min_confidence
        ):
            reconcile_rounds += 1
            reconcile_path = article_dir / f"04b_reconcile_round_{reconcile_rounds}.json"
            final_result = self._load_or_run(
                reconcile_path,
                self._build_reconcile_pass,
                article,
                theory_map,
                theory_rubric,
                article_map,
                relevance_audit,
                counter_audit,
                final_result,
                reconcile_rounds,
            )

        implication_path = article_dir / "05_theory_implications.json"
        theory_implications = self._load_or_run(
            implication_path,
            self._build_theory_implications,
            article,
            theory_map,
            theory_rubric,
            article_map,
            relevance_audit,
            counter_audit,
            final_result,
        ) if self._should_generate_implications(
            final_result.get("overall_verdict") or final_result.get("reconciled_verdict")
        ) else self._maybe_load_implications(implication_path)
        if not self._should_generate_implications(
            final_result.get("overall_verdict") or final_result.get("reconciled_verdict")
        ) and implication_path.exists():
            implication_path.unlink()
            theory_implications = None

        report_path = article_dir / "report.md"
        report_path.write_text(
            render_report(
                article=article,
                article_map=article_map,
                theory_map=theory_map,
                theory_implications=theory_implications,
                final_result=final_result,
                initial_audit=relevance_audit,
                counter_audit=counter_audit,
                document_label=self._document_label(),
            ),
            encoding="utf-8",
        )

        return {
            "article_path": str(article.path),
            "article_slug": article.slug,
            "report_path": str(report_path),
            "verdict": final_result.get("overall_verdict") or final_result.get("reconciled_verdict") or "unknown",
            "confidence": final_result.get("confidence")
            or final_result.get("reconciled_confidence")
            or 0.0,
            "recommended_use": final_result.get("recommended_use", "unknown"),
        }

    def _refresh_implications(
        self,
        article: Document,
        theory_map: dict[str, Any],
        theory_rubric: dict[str, Any],
    ) -> dict[str, Any] | None:
        article_dir = self.outputs_dir / article.slug
        if not article_dir.exists():
            return None
        article_map = self._load_required(article_dir / "01_article_map.json", f"article map for {article.slug}")
        relevance_audit = self._load_required(
            article_dir / "02_relevance_audit.json",
            f"relevance audit for {article.slug}",
        )
        counter_audit = self._load_required(
            article_dir / "03_counter_audit.json",
            f"counter audit for {article.slug}",
        )
        final_result = self._load_existing_final_result(article_dir, article.slug)
        implication_path = article_dir / "05_theory_implications.json"
        verdict = final_result.get("overall_verdict") or final_result.get("reconciled_verdict")
        if not self._should_generate_implications(verdict):
            if implication_path.exists():
                implication_path.unlink()
            report_path = article_dir / "report.md"
            report_path.write_text(
                render_report(
                    article=article,
                    article_map=article_map,
                    theory_map=theory_map,
                    theory_implications=None,
                    final_result=final_result,
                    initial_audit=relevance_audit,
                    counter_audit=counter_audit,
                    document_label=self._document_label(),
                ),
                encoding="utf-8",
            )
            return None

        theory_implications = self._load_or_run(
            implication_path,
            self._build_theory_implications,
            article,
            theory_map,
            theory_rubric,
            article_map,
            relevance_audit,
            counter_audit,
            final_result,
        )
        report_path = article_dir / "report.md"
        report_path.write_text(
            render_report(
                article=article,
                article_map=article_map,
                theory_map=theory_map,
                theory_implications=theory_implications,
                final_result=final_result,
                initial_audit=relevance_audit,
                counter_audit=counter_audit,
                document_label=self._document_label(),
            ),
            encoding="utf-8",
        )
        return {
            "article_path": str(article.path),
            "article_slug": article.slug,
            "report_path": str(report_path),
            "verdict": verdict or "unknown",
            "confidence": theory_implications.get("confidence", 0.0),
            "recommended_use": final_result.get("recommended_use", "unknown"),
        }

    def _build_article_map(self, article: Document) -> dict[str, Any]:
        if self._is_academic():
            prompt = textwrap.dedent(
                f"""
                Read the academic paper and map what it actually argues before trying to connect it to the theory.

                Constraints:
                - Stay descriptive.
                - Do not infer relevance yet.
                - Capture the paper's thesis, research question, method or approach, theoretical frameworks, empirical scope, and explicit intervention in the literature.
                - Because the paper is long, this pass should produce a rich digest that later passes can rely on without rereading the whole paper.

                Return JSON with exactly these keys:
                {{
                  "article_kind": "academic",
                  "summary": "string",
                  "research_question": "string",
                  "main_claims": ["string"],
                  "method_or_approach": ["string"],
                  "empirical_scope_or_case": ["string"],
                  "evidence_or_facts": ["string"],
                  "theoretical_frameworks": ["string"],
                  "explicit_theoretical_interventions": ["string"],
                  "section_level_moves": ["string"],
                  "rhetorical_frames": ["string"],
                  "state_and_policy_content": ["string"],
                  "capital_accumulation_content": ["string"],
                  "labor_content": ["string"],
                  "infrastructure_and_supply_chain_content": ["string"],
                  "geopolitical_competition_content": ["string"],
                  "possible_theory_hooks": ["string"],
                  "what_is_missing_for_theory_evaluation": ["string"]
                }}

                Paper file: {article.path.name}

                Paper text:
                {article.text}
                """
            ).strip()
        else:
            prompt = textwrap.dedent(
                f"""
                Read the article and map what it actually says before trying to connect it to the theory.

                Constraints:
                - Stay descriptive.
                - Do not infer theoretical relevance yet.
                - Capture concrete facts, policy content, labor content, and missing information.

                Return JSON with exactly these keys:
                {{
                  "article_kind": "news|op-ed|policy memo|academic|other",
                  "summary": "string",
                  "main_claims": ["string"],
                  "evidence_or_facts": ["string"],
                  "rhetorical_frames": ["string"],
                  "state_and_policy_content": ["string"],
                  "capital_accumulation_content": ["string"],
                  "labor_content": ["string"],
                  "infrastructure_and_supply_chain_content": ["string"],
                  "geopolitical_competition_content": ["string"],
                  "possible_theory_hooks": ["string"],
                  "what_is_missing_for_theory_evaluation": ["string"]
                }}

                Article file: {article.path.name}

                Article text:
                {article.text}
                """
            ).strip()
        return self.client.chat_json([{"role": "user", "content": prompt}])

    def _build_relevance_audit(
        self,
        article: Document,
        theory_map: dict[str, Any],
        theory_rubric: dict[str, Any],
        article_map: dict[str, Any],
    ) -> dict[str, Any]:
        if self._is_academic():
            prompt = textwrap.dedent(
                f"""
                Judge whether this academic paper bears on the theory.

                Apply the rubric in a calibrated way.
                - A paper can matter through direct conceptual argument, literature intervention, historical reconstruction, empirical case evidence, or labor-process analysis.
                - Do not mistake shared keywords for engagement.
                - If the paper is adjacent rather than directly probative, preserve that as contextual or marginal relevance instead of forcing irrelevance.
                - If the paper directly contests monopoly, competition, rent, capitalist efficiency, platform capitalism, linguistic labor, or related categories that touch the theory's claims, say so explicitly.

                Return JSON with exactly these keys:
                {{
                  "overall_initial_verdict": "irrelevant|contextual|marginal|relevant",
                  "overall_reason": "string",
                  "claim_assessments": [
                    {{
                      "claim_id": "C1",
                      "engagement_type": "none|rhetorical|contextual|illustrative|direct",
                      "support_strength": "none|weak|moderate|strong",
                      "challenge_strength": "none|weak|moderate|strong",
                      "support_points": ["string"],
                      "challenge_points": ["string"],
                      "why_limited": "string"
                    }}
                  ],
                  "contextual_relevance_points": ["string"],
                  "illustrative_relevance_points": ["string"],
                  "article_level_points_for_theory": ["string"],
                  "article_level_points_against_theory": ["string"],
                  "irrelevance_reasons": ["string"],
                  "confidence": 0.0
                }}

                Theory map:
                {json.dumps(theory_map, ensure_ascii=False, indent=2)}

                Rubric:
                {json.dumps(theory_rubric, ensure_ascii=False, indent=2)}

                Paper map:
                {json.dumps(article_map, ensure_ascii=False, indent=2)}

                {self._later_pass_context(article, article_map)}
                """
            ).strip()
        else:
            prompt = textwrap.dedent(
                f"""
                Judge whether this article bears on the theory.

                Apply the rubric in a calibrated way.
                - Do not require the article to adjudicate rent vs. profit before assigning contextual or marginal relevance.
                - If the article is useful as a state-capital, infrastructure, labor-displacement, geopolitics, supply-chain, platform-power, or public-legitimation data point, say so and assign contextual or marginal relevance.
                - If the article cannot support or challenge the theory directly, say that under limitations rather than forcing the whole verdict to irrelevant.
                - Reserve "irrelevant" for articles with no meaningful use even as background or illustration.

                Return JSON with exactly these keys:
                {{
                  "overall_initial_verdict": "irrelevant|contextual|marginal|relevant",
                  "overall_reason": "string",
                  "claim_assessments": [
                    {{
                      "claim_id": "C1",
                      "engagement_type": "none|rhetorical|contextual|illustrative|direct",
                      "support_strength": "none|weak|moderate|strong",
                      "challenge_strength": "none|weak|moderate|strong",
                      "support_points": ["string"],
                      "challenge_points": ["string"],
                      "why_limited": "string"
                    }}
                  ],
                  "contextual_relevance_points": ["string"],
                  "illustrative_relevance_points": ["string"],
                  "article_level_points_for_theory": ["string"],
                  "article_level_points_against_theory": ["string"],
                  "irrelevance_reasons": ["string"],
                  "confidence": 0.0
                }}

                Theory map:
                {json.dumps(theory_map, ensure_ascii=False, indent=2)}

                Rubric:
                {json.dumps(theory_rubric, ensure_ascii=False, indent=2)}

                Article map:
                {json.dumps(article_map, ensure_ascii=False, indent=2)}

                {self._later_pass_context(article, article_map)}
                """
            ).strip()
        return self.client.chat_json([{"role": "user", "content": prompt}])

    def _build_counter_audit(
        self,
        article: Document,
        theory_map: dict[str, Any],
        theory_rubric: dict[str, Any],
        article_map: dict[str, Any],
        relevance_audit: dict[str, Any],
    ) -> dict[str, Any]:
        if self._is_academic():
            prompt = textwrap.dedent(
                f"""
                Audit the initial relevance judgment for overclaiming, missed direct engagement, and sloppy conceptual mappings.

                Be adversarial.
                - If the initial audit inflated an adjacent paper into direct support or challenge, say so.
                - If it erased genuine theoretical or empirical relevance by demanding an impossible one-to-one match with the essay, say so.
                - Distinguish direct theoretical argument, empirical case relevance, and merely adjacent literature.
                - Preserve contextual or marginal relevance when it is real, but do not inflate it.

                Return JSON with exactly these keys:
                {{
                  "grade_inflation_detected": true,
                  "false_negative_detected": true,
                  "problems_with_initial_audit": ["string"],
                  "missing_support_points": ["string"],
                  "missing_challenge_points": ["string"],
                  "missing_contextual_or_illustrative_points": ["string"],
                  "corrected_verdict": "irrelevant|contextual|marginal|relevant",
                  "corrections_by_claim": [
                    {{
                      "claim_id": "C1",
                      "corrected_support_strength": "none|weak|moderate|strong",
                      "corrected_challenge_strength": "none|weak|moderate|strong",
                      "note": "string"
                    }}
                  ],
                  "confidence": 0.0
                }}

                Theory map:
                {json.dumps(theory_map, ensure_ascii=False, indent=2)}

                Rubric:
                {json.dumps(theory_rubric, ensure_ascii=False, indent=2)}

                Paper map:
                {json.dumps(article_map, ensure_ascii=False, indent=2)}

                Initial audit:
                {json.dumps(relevance_audit, ensure_ascii=False, indent=2)}

                {self._later_pass_context(article, article_map)}
                """
            ).strip()
        else:
            prompt = textwrap.dedent(
                f"""
                Audit the initial relevance judgment for overclaiming, sloppy mappings, and missed points.

                Be adversarial.
                - If the initial audit inflated weak evidence into support, say so.
                - If it erased a genuine contextual, illustrative, or narrow point of contact by applying an over-strict gate, say so.
                - Distinguish political-process evidence from evidence about economic mechanism.
                - Do not convert every indirect data point into proof, but preserve it as contextual or marginal relevance when useful.

                Return JSON with exactly these keys:
                {{
                  "grade_inflation_detected": true,
                  "false_negative_detected": true,
                  "problems_with_initial_audit": ["string"],
                  "missing_support_points": ["string"],
                  "missing_challenge_points": ["string"],
                  "missing_contextual_or_illustrative_points": ["string"],
                  "corrected_verdict": "irrelevant|contextual|marginal|relevant",
                  "corrections_by_claim": [
                    {{
                      "claim_id": "C1",
                      "corrected_support_strength": "none|weak|moderate|strong",
                      "corrected_challenge_strength": "none|weak|moderate|strong",
                      "note": "string"
                    }}
                  ],
                  "confidence": 0.0
                }}

                Theory map:
                {json.dumps(theory_map, ensure_ascii=False, indent=2)}

                Rubric:
                {json.dumps(theory_rubric, ensure_ascii=False, indent=2)}

                Article map:
                {json.dumps(article_map, ensure_ascii=False, indent=2)}

                Initial audit:
                {json.dumps(relevance_audit, ensure_ascii=False, indent=2)}

                {self._later_pass_context(article, article_map)}
                """
            ).strip()
        return self.client.chat_json([{"role": "user", "content": prompt}])

    def _build_final_judgment(
        self,
        article: Document,
        theory_map: dict[str, Any],
        theory_rubric: dict[str, Any],
        article_map: dict[str, Any],
        relevance_audit: dict[str, Any],
        counter_audit: dict[str, Any],
    ) -> dict[str, Any]:
        if self._is_academic():
            prompt = textwrap.dedent(
                f"""
                Produce the final judgment for this academic paper.

                Requirements:
                - Keep the verdict strict and calibrated.
                - A paper may count as relevant if it directly engages the categories or mechanisms at stake, even without new quantitative data.
                - Separate direct arguments for the theory, direct arguments against it, and contextual relevance that is useful but not probative.
                - Also state what the paper cannot adjudicate.

                Return JSON with exactly these keys:
                {{
                  "overall_verdict": "irrelevant|contextual|marginal|relevant",
                  "confidence": 0.0,
                  "relevance_mode": "none|contextual|illustrative|direct",
                  "one_paragraph_verdict": "string",
                  "contextual_relevance": ["string"],
                  "arguments_for_theory": [
                    {{
                      "strength": "weak|moderate|strong",
                      "argument": "string",
                      "claim_ids": ["C1"]
                    }}
                  ],
                  "arguments_against_theory": [
                    {{
                      "strength": "weak|moderate|strong",
                      "argument": "string",
                      "claim_ids": ["C1"]
                    }}
                  ],
                  "what_article_cannot_adjudicate": ["string"],
                  "state_capital_nexus_relevance": "string",
                  "recommended_use": "ignore|mention_in_passing|use_as_minor_update|use_as_substantive_evidence"
                }}

                Theory map:
                {json.dumps(theory_map, ensure_ascii=False, indent=2)}

                Rubric:
                {json.dumps(theory_rubric, ensure_ascii=False, indent=2)}

                Paper map:
                {json.dumps(article_map, ensure_ascii=False, indent=2)}

                Initial audit:
                {json.dumps(relevance_audit, ensure_ascii=False, indent=2)}

                Counter-audit:
                {json.dumps(counter_audit, ensure_ascii=False, indent=2)}

                {self._later_pass_context(article, article_map)}
                """
            ).strip()
        else:
            prompt = textwrap.dedent(
                f"""
                Produce the final judgment.

                Requirements:
                - Keep the verdict strict and calibrated.
                - It is acceptable for the answer to say the article is mostly irrelevant, but do not call it irrelevant if it has a real contextual or illustrative use.
                - Separate "arguments for the theory" from "arguments against the theory."
                - Also identify contextual relevance that is not yet an argument.
                - Also state what the article cannot adjudicate.

                Return JSON with exactly these keys:
                {{
                  "overall_verdict": "irrelevant|contextual|marginal|relevant",
                  "confidence": 0.0,
                  "relevance_mode": "none|contextual|illustrative|direct",
                  "one_paragraph_verdict": "string",
                  "contextual_relevance": ["string"],
                  "arguments_for_theory": [
                    {{
                      "strength": "weak|moderate|strong",
                      "argument": "string",
                      "claim_ids": ["C1"]
                    }}
                  ],
                  "arguments_against_theory": [
                    {{
                      "strength": "weak|moderate|strong",
                      "argument": "string",
                      "claim_ids": ["C1"]
                    }}
                  ],
                  "what_article_cannot_adjudicate": ["string"],
                  "state_capital_nexus_relevance": "string",
                  "recommended_use": "ignore|mention_in_passing|use_as_minor_update|use_as_substantive_evidence"
                }}

                Theory map:
                {json.dumps(theory_map, ensure_ascii=False, indent=2)}

                Rubric:
                {json.dumps(theory_rubric, ensure_ascii=False, indent=2)}

                Article map:
                {json.dumps(article_map, ensure_ascii=False, indent=2)}

                Initial audit:
                {json.dumps(relevance_audit, ensure_ascii=False, indent=2)}

                Counter-audit:
                {json.dumps(counter_audit, ensure_ascii=False, indent=2)}

                {self._later_pass_context(article, article_map)}
                """
            ).strip()
        return self.client.chat_json([{"role": "user", "content": prompt}])

    def _build_reconcile_pass(
        self,
        article: Document,
        theory_map: dict[str, Any],
        theory_rubric: dict[str, Any],
        article_map: dict[str, Any],
        relevance_audit: dict[str, Any],
        counter_audit: dict[str, Any],
        current_final: dict[str, Any],
        round_number: int,
    ) -> dict[str, Any]:
        if self._is_academic():
            prompt = textwrap.dedent(
                f"""
                Reconcile remaining disagreement or low confidence in the paper-theory assessment.

                This is reconcile round {round_number}.

                Requirements:
                - Resolve only the disputed points.
                - Tighten the verdict rather than expanding it.
                - If uncertainty remains, lower the confidence and explain why.
                - Preserve the difference between direct theoretical engagement, empirical illustration, and contextual adjacency.

                Return JSON with exactly these keys:
                {{
                  "overall_verdict": "irrelevant|contextual|marginal|relevant",
                  "reconciled_confidence": 0.0,
                  "relevance_mode": "none|contextual|illustrative|direct",
                  "one_paragraph_verdict": "string",
                  "contextual_relevance": ["string"],
                  "arguments_for_theory": [
                    {{
                      "strength": "weak|moderate|strong",
                      "argument": "string",
                      "claim_ids": ["C1"]
                    }}
                  ],
                  "arguments_against_theory": [
                    {{
                      "strength": "weak|moderate|strong",
                      "argument": "string",
                      "claim_ids": ["C1"]
                    }}
                  ],
                  "what_article_cannot_adjudicate": ["string"],
                  "state_capital_nexus_relevance": "string",
                  "recommended_use": "ignore|mention_in_passing|use_as_minor_update|use_as_substantive_evidence"
                }}

                Theory map:
                {json.dumps(theory_map, ensure_ascii=False, indent=2)}

                Rubric:
                {json.dumps(theory_rubric, ensure_ascii=False, indent=2)}

                Paper map:
                {json.dumps(article_map, ensure_ascii=False, indent=2)}

                Initial audit:
                {json.dumps(relevance_audit, ensure_ascii=False, indent=2)}

                Counter-audit:
                {json.dumps(counter_audit, ensure_ascii=False, indent=2)}

                Current final judgment:
                {json.dumps(current_final, ensure_ascii=False, indent=2)}

                {self._later_pass_context(article, article_map)}
                """
            ).strip()
        else:
            prompt = textwrap.dedent(
                f"""
                Reconcile remaining disagreement or low confidence in the article-theory assessment.

                This is reconcile round {round_number}.

                Requirements:
                - Resolve only the disputed points.
                - Tighten the verdict rather than expanding it.
                - If uncertainty remains, lower the confidence and explain why.

                Return JSON with exactly these keys:
                {{
                  "overall_verdict": "irrelevant|contextual|marginal|relevant",
                  "reconciled_confidence": 0.0,
                  "relevance_mode": "none|contextual|illustrative|direct",
                  "one_paragraph_verdict": "string",
                  "contextual_relevance": ["string"],
                  "arguments_for_theory": [
                    {{
                      "strength": "weak|moderate|strong",
                      "argument": "string",
                      "claim_ids": ["C1"]
                    }}
                  ],
                  "arguments_against_theory": [
                    {{
                      "strength": "weak|moderate|strong",
                      "argument": "string",
                      "claim_ids": ["C1"]
                    }}
                  ],
                  "what_article_cannot_adjudicate": ["string"],
                  "state_capital_nexus_relevance": "string",
                  "recommended_use": "ignore|mention_in_passing|use_as_minor_update|use_as_substantive_evidence"
                }}

                Theory map:
                {json.dumps(theory_map, ensure_ascii=False, indent=2)}

                Rubric:
                {json.dumps(theory_rubric, ensure_ascii=False, indent=2)}

                Article map:
                {json.dumps(article_map, ensure_ascii=False, indent=2)}

                Initial audit:
                {json.dumps(relevance_audit, ensure_ascii=False, indent=2)}

                Counter-audit:
                {json.dumps(counter_audit, ensure_ascii=False, indent=2)}

                Current final judgment:
                {json.dumps(current_final, ensure_ascii=False, indent=2)}

                {self._later_pass_context(article, article_map)}
                """
            ).strip()
        return self.client.chat_json([{"role": "user", "content": prompt}])

    def _build_theory_implications(
        self,
        article: Document,
        theory_map: dict[str, Any],
        theory_rubric: dict[str, Any],
        article_map: dict[str, Any],
        relevance_audit: dict[str, Any],
        counter_audit: dict[str, Any],
        final_result: dict[str, Any],
    ) -> dict[str, Any]:
        label = self._document_label()
        prompt = textwrap.dedent(
            f"""
            You are writing a theory-revision memo for a {label} that has already been judged at least
            {self.implication_min_verdict} in relevance.

            Your job is not to restate the verdict. Your job is to say what this {label} does to the
            theory map.

            Constraints:
            - Be conservative. Do not propose revisions unless the evidence or argument in the {label}
              justifies them.
            - Distinguish:
              - confirms: supports an existing claim as stated
              - qualifies: narrows or conditions an existing claim
              - extends: adds a new sub-claim, mechanism, or empirical scope
              - pressures: creates real tension or revision pressure
              - mixed: does more than one of the above
            - If the {label} mainly reinforces one existing claim without changing the map, say so.
            - Proposed revisions should be specific enough to edit the theory map, not vague commentary.
            - Use claim ids whenever possible.

            Return JSON with exactly these keys:
            {{
              "overall_implication": "confirms|qualifies|extends|pressures|mixed",
              "summary": "string",
              "claim_level_implications": [
                {{
                  "claim_id": "C1",
                  "effect": "confirms|qualifies|extends|pressures",
                  "why": "string",
                  "evidence_from_document": ["string"],
                  "proposed_revision": "string"
                }}
              ],
              "new_subclaims": ["string"],
              "new_open_questions": ["string"],
              "revision_priority": "low|medium|high",
              "recommended_follow_up": ["string"],
              "confidence": 0.0
            }}

            Theory map:
            {json.dumps(theory_map, ensure_ascii=False, indent=2)}

            Rubric:
            {json.dumps(theory_rubric, ensure_ascii=False, indent=2)}

            {label.title()} map:
            {json.dumps(article_map, ensure_ascii=False, indent=2)}

            Initial audit:
            {json.dumps(relevance_audit, ensure_ascii=False, indent=2)}

            Counter-audit:
            {json.dumps(counter_audit, ensure_ascii=False, indent=2)}

            Final judgment:
            {json.dumps(final_result, ensure_ascii=False, indent=2)}

            {self._later_pass_context(article, article_map)}
            """
        ).strip()
        return self.client.chat_json([{"role": "user", "content": prompt}])

    def _load_or_run(self, path: Path, builder: Any, *args: Any) -> dict[str, Any]:
        if path.exists() and not self.overwrite:
            payload = sanitize_payload(json.loads(path.read_text(encoding="utf-8")))
            try:
                validate_payload_for_path(path, payload, self.analysis_profile)
                return payload
            except ValueError as exc:
                print(f"STALE: {path}: {exc}; rebuilding")
        last_error: Exception | None = None
        for attempt in range(1, self.max_step_retries + 2):
            try:
                payload = builder(*args)
                payload = sanitize_payload(payload)
                validate_payload_for_path(path, payload, self.analysis_profile)
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                error_path = error_artifact_path(path)
                if error_path.exists():
                    error_path.unlink()
                return payload
            except Exception as exc:
                last_error = exc
                if attempt <= self.max_step_retries:
                    time.sleep(attempt)
                    continue
        error_path = error_artifact_path(path)
        error_path.write_text(
            f"Failed after {self.max_step_retries + 1} attempt(s).\n{type(last_error).__name__}: {last_error}\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"Failed to build {path}: {last_error}") from last_error

    def _load_required(self, path: Path, label: str) -> dict[str, Any]:
        if not path.exists():
            raise RuntimeError(
                f"Missing {label} at {path}. Run `--stage theory` first and commit the generated output."
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_existing_final_result(self, article_dir: Path, slug: str) -> dict[str, Any]:
        reconcile_paths = sorted(article_dir.glob("04b_reconcile_round_*.json"))
        if reconcile_paths:
            return self._load_required(
                reconcile_paths[-1],
                f"latest reconcile result for {slug}",
            )
        return self._load_required(article_dir / "04_final_judgment.json", f"final judgment for {slug}")

    def _maybe_load_implications(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        payload = sanitize_payload(json.loads(path.read_text(encoding="utf-8")))
        validate_payload_for_path(path, payload, self.analysis_profile)
        return payload

    def _write_theory_index(
        self,
        theory_documents: list[Document],
        theory_map_path: Path,
        theory_rubric_path: Path,
    ) -> Path:
        lines = [
            "# Preprocessed Theory Outputs",
            "",
            f"- Model: `{self.client.model}`",
            f"- Theory files: {', '.join(doc.path.name for doc in theory_documents)}",
            f"- Theory map: `{theory_map_path.relative_to(self.workspace)}`",
            f"- Theory rubric: `{theory_rubric_path.relative_to(self.workspace)}`",
            "",
            f"These files are committed so remote machines can run only the {self._comparison_label()} stage.",
            "",
        ]
        index_md = self.outputs_dir / "_theory" / "README.md"
        index_md.write_text("\n".join(lines), encoding="utf-8")
        return index_md

    def _write_index(
        self,
        theory_documents: list[Document],
        analyses: list[dict[str, Any]],
        failures: list[dict[str, str]] | None = None,
    ) -> Path:
        failures = failures or []
        failures_json = self.outputs_dir / "failures.json"
        if failures:
            failures_json.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        elif failures_json.exists():
            failures_json.unlink()

        index_json = self.outputs_dir / "index.json"
        payload = {
            "model": self.client.model,
            "analysis_profile": self.analysis_profile,
            "theory_files": [str(doc.path) for doc in theory_documents],
            "failed_articles": failures,
            "articles": sorted(
                analyses,
                key=lambda item: (
                    verdict_rank(item["verdict"]),
                    -float(item.get("confidence", 0.0)),
                    item["article_slug"],
                ),
            ),
        }
        index_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = [
            f"# {self.profile.index_title}",
            "",
            f"- Model: `{self.client.model}`",
            f"- Analysis profile: `{self.analysis_profile}`",
            f"- Theory files: {', '.join(doc.path.name for doc in theory_documents)}",
            f"- {self._document_label().title()}s analyzed: {len(analyses)}",
            f"- {self._document_label().title()}s failed: {len(failures)}",
            "",
            f"| {self._document_label().title()} | Verdict | Confidence | Recommended Use | Report |",
            "| --- | --- | --- | --- | --- |",
        ]
        for item in payload["articles"]:
            report_rel = Path(item["report_path"]).relative_to(self.workspace)
            lines.append(
                f"| {item['article_slug']} | {item['verdict']} | {item['confidence']:.2f} | "
                f"{item['recommended_use']} | `{report_rel}` |"
            )
        if failures:
            lines.extend(["", "## Failures", ""])
            for item in failures:
                lines.append(f"- `{item['article_slug']}`: {item['error']}")
        index_md = self.outputs_dir / "index.md"
        index_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return index_md


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a multi-pass theory-vs-article analysis pipeline."
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--stage", choices=("all", "theory", "articles", "implications"), default="all")
    parser.add_argument("--profile", choices=tuple(PROFILE_CONFIGS), default="journalistic")
    parser.add_argument("--provider", choices=("ollama", "anthropic"), default="ollama")
    parser.add_argument("--model")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--anthropic-url", default="https://api.anthropic.com")
    parser.add_argument("--anthropic-api-key-env", default="ANTHROPIC_API_KEY")
    parser.add_argument("--anthropic-max-tokens", type=int, default=32_000)
    parser.add_argument("--anthropic-effort", choices=("low", "medium", "high", "max"), default="max")
    parser.add_argument("--anthropic-thinking", choices=("adaptive", "disabled"), default="adaptive")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-theory-chars", type=int, default=180_000)
    parser.add_argument("--max-article-chars", type=int)
    parser.add_argument("--comparison-subdir")
    parser.add_argument("--cache-subdir")
    parser.add_argument("--outputs-subdir")
    parser.add_argument(
        "--implication-min-verdict",
        choices=tuple(sorted(IMPLICATION_VERDICTS)),
        default="none",
        help="Generate 05_theory_implications.json for documents at or above this verdict threshold.",
    )
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--max-reconcile-rounds", type=int, default=2)
    parser.add_argument("--num-ctx", type=int, default=131072)
    parser.add_argument("--parallel", type=int, default=3, help="Number of articles to process concurrently")
    parser.add_argument("--step-retries", type=int, default=2)
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()
    os.environ["GEMMA_PARALLEL"] = str(args.parallel)
    selected_profile = profile_config(args.profile)
    max_article_chars = args.max_article_chars or selected_profile.default_max_comparison_chars

    if args.provider == "anthropic":
        client = AnthropicClient(
            model=args.model or DEFAULT_ANTHROPIC_MODEL,
            api_key_env=args.anthropic_api_key_env,
            base_url=args.anthropic_url,
            system_prompt=selected_profile.system_prompt,
            max_tokens=args.anthropic_max_tokens,
            effort=args.anthropic_effort,
            thinking_type=args.anthropic_thinking,
        )
    else:
        client = OllamaClient(
            model=args.model or DEFAULT_OLLAMA_MODEL,
            base_url=args.ollama_url,
            system_prompt=selected_profile.system_prompt,
            num_ctx=args.num_ctx,
        )

    runner = TheoryArticleRunner(
        workspace=args.workspace,
        client=client,
        analysis_profile=args.profile,
        comparison_subdir=args.comparison_subdir,
        cache_subdir=args.cache_subdir,
        outputs_subdir=args.outputs_subdir,
        overwrite=args.overwrite,
        max_theory_chars=args.max_theory_chars,
        max_article_chars=max_article_chars,
        implication_min_verdict=args.implication_min_verdict,
        min_confidence=args.min_confidence,
        max_reconcile_rounds=args.max_reconcile_rounds,
        max_step_retries=args.step_retries,
        allow_failures=args.allow_failures,
    )
    result = runner.run(stage=args.stage)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def ensure_workspace(
    workspace: Path,
    comparison_subdir: str = "others",
    outputs_subdir: str = "outputs",
    cache_subdir: str = "others",
) -> None:
    for name in ("nlr", comparison_subdir, outputs_subdir, "cache", f"cache/{cache_subdir}"):
        (workspace / name).mkdir(parents=True, exist_ok=True)


def load_documents(directory: Path, cache_dir: Path, max_chars: int) -> list[Document]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    documents: list[Document] = []
    for path in sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS):
        slug = slugify(path.stem)
        extracted_path = cache_dir / f"{slug}.txt"
        text = extract_text(path)
        limited = limit_text(clean_text(text), max_chars)
        extracted_path.write_text(limited, encoding="utf-8")
        documents.append(Document(path=path, slug=slug, text=limited, extracted_path=extracted_path))
    return documents


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        return html_to_text(path)
    if suffix == ".docx":
        return docx_to_text(path)
    if suffix == ".pdf":
        return pdf_to_text(path)
    raise ValueError(f"Unsupported file type: {path}")


def html_to_text(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    return soup.get_text("\n")


def docx_to_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        with archive.open("word/document.xml") as document_xml:
            root = ET.fromstring(document_xml.read())
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    lines: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        line = "".join(texts).strip()
        if line:
            lines.append(line)
    return "\n\n".join(lines)


def pdf_to_text(path: Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        pdftotext = shutil.which("pdftotext")
        if not pdftotext:
            raise
        result = subprocess.run(
            [pdftotext, "-layout", str(path), "-"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout


def clean_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\u0000", "", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def limit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return (
        text[:head].rstrip()
        + "\n\n[... CONTENT TRUNCATED FOR PROMPT BUDGET ...]\n\n"
        + text[-tail:].lstrip()
    )


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "document"


def render_document_bundle(documents: list[Document], label: str) -> str:
    parts = [f"{label}:"]
    for document in documents:
        parts.append(f"\n--- START {document.path.name} ---\n{document.text}\n--- END {document.path.name} ---")
    return "\n".join(parts)


def parse_json_payload(content: str) -> dict[str, Any]:
    candidate = strip_json_fence(content)
    try:
        payload = json.loads(candidate)
        if not isinstance(payload, dict):
            raise JsonResponseError("Model returned JSON, but not a JSON object.", candidate)
        return payload
    except json.JSONDecodeError as first_error:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"{", candidate):
            try:
                payload, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise JsonResponseError(
            f"Model did not return valid JSON: {first_error.msg} at character {first_error.pos}.",
            candidate,
        ) from first_error


def strip_json_fence(content: str) -> str:
    candidate = content.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    return candidate.strip()


def chat_json_with_retries(
    chat_text: Any,
    messages: list[dict[str, str]],
    max_json_retries: int,
) -> dict[str, Any]:
    errors: list[str] = []
    raw = ""
    for attempt in range(max_json_retries + 1):
        prompt = messages if attempt == 0 or attempt % 2 == 0 else json_repair_messages(raw)
        raw = chat_text(prompt)
        try:
            return parse_json_payload(raw)
        except JsonResponseError as exc:
            errors.append(str(exc))
            if attempt >= max_json_retries:
                raise JsonResponseError(
                    "Model returned malformed JSON after retries: " + " | ".join(errors),
                    exc.raw,
                ) from exc
            time.sleep(attempt + 1)
    raise JsonResponseError("Model returned malformed JSON after retries.", raw)


def json_repair_messages(raw: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "Repair the following malformed model response into valid JSON only. "
                "Preserve the semantic content and keys as much as possible. "
                "Do not add markdown or commentary.\n\n"
                f"Malformed response:\n{limit_text(raw, 120_000)}"
            ),
        }
    ]


def error_artifact_path(path: Path) -> Path:
    return path.with_name(path.name + ".error.txt")


def validate_payload_for_path(
    path: Path,
    payload: dict[str, Any],
    analysis_profile: str = "journalistic",
) -> None:
    required = required_keys_for_output(path.name, analysis_profile)
    missing = sorted(key for key in required if key not in payload)
    issues = [f"missing required key `{key}`" for key in missing]
    issues.extend(find_payload_quality_issues(payload))
    if issues:
        preview = "; ".join(issues[:8])
        raise ValueError(f"Invalid payload for {path.name}: {preview}")


def sanitize_payload(payload: Any) -> Any:
    """Remove empty string list items and strings containing noise tokens.

    Returns a cleaned copy so validate_payload_for_path can still flag
    structural problems (missing keys, embedded JSON) while tolerating minor
    generation artifacts that otherwise force expensive full-call retries.
    """
    if isinstance(payload, dict):
        return {key: sanitize_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        cleaned: list[Any] = []
        for value in payload:
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    continue
                if any(pattern in value for pattern in SUSPECT_NOISE_PATTERNS):
                    continue
                cleaned.append(stripped)
            else:
                cleaned.append(sanitize_payload(value))
        return cleaned
    return payload


def required_keys_for_output(filename: str, analysis_profile: str = "journalistic") -> set[str]:
    if filename == "01_theory_map.json":
        return {
            "thesis",
            "core_claims",
            "secondary_themes",
            "article_relevance_hooks",
            "conceptual_boundaries",
            "relevance_red_flags",
            "open_questions",
        }
    if filename == "02_theory_rubric.json":
        return {
            "relevance_tiers",
            "relevance_questions",
            "what_counts_as_real_evidence",
            "what_counts_as_contextual_or_illustrative_relevance",
            "what_does_not_count",
            "support_strength_scale",
            "challenge_strength_scale",
            "anti_grade_inflation_rule",
            "anti_false_negative_rule",
            "default_verdict_rule",
        }
    if filename == "01_article_map.json":
        keys = {
            "article_kind",
            "summary",
            "main_claims",
            "evidence_or_facts",
            "rhetorical_frames",
            "state_and_policy_content",
            "capital_accumulation_content",
            "labor_content",
            "infrastructure_and_supply_chain_content",
            "geopolitical_competition_content",
            "possible_theory_hooks",
            "what_is_missing_for_theory_evaluation",
        }
        if analysis_profile == "academic":
            keys.update(
                {
                    "research_question",
                    "method_or_approach",
                    "empirical_scope_or_case",
                    "theoretical_frameworks",
                    "explicit_theoretical_interventions",
                    "section_level_moves",
                }
            )
        return keys
    if filename == "02_relevance_audit.json":
        return {
            "overall_initial_verdict",
            "overall_reason",
            "claim_assessments",
            "contextual_relevance_points",
            "illustrative_relevance_points",
            "article_level_points_for_theory",
            "article_level_points_against_theory",
            "irrelevance_reasons",
            "confidence",
        }
    if filename == "03_counter_audit.json":
        return {
            "grade_inflation_detected",
            "false_negative_detected",
            "problems_with_initial_audit",
            "missing_support_points",
            "missing_challenge_points",
            "missing_contextual_or_illustrative_points",
            "corrected_verdict",
            "corrections_by_claim",
            "confidence",
        }
    if filename == "04_final_judgment.json":
        return {
            "overall_verdict",
            "confidence",
            "relevance_mode",
            "one_paragraph_verdict",
            "contextual_relevance",
            "arguments_for_theory",
            "arguments_against_theory",
            "what_article_cannot_adjudicate",
            "state_capital_nexus_relevance",
            "recommended_use",
        }
    if filename.startswith("04b_reconcile_round_") and filename.endswith(".json"):
        return {
            "overall_verdict",
            "reconciled_confidence",
            "relevance_mode",
            "one_paragraph_verdict",
            "contextual_relevance",
            "arguments_for_theory",
            "arguments_against_theory",
            "what_article_cannot_adjudicate",
            "state_capital_nexus_relevance",
            "recommended_use",
        }
    if filename == "05_theory_implications.json":
        return {
            "overall_implication",
            "summary",
            "claim_level_implications",
            "new_subclaims",
            "new_open_questions",
            "revision_priority",
            "recommended_follow_up",
            "confidence",
        }
    return set()


def find_payload_quality_issues(payload: Any, path: str = "$") -> list[str]:
    issues: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            issues.extend(find_payload_quality_issues(value, f"{path}.{key}"))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            if isinstance(value, str) and not value.strip():
                issues.append(f"empty string list item at {path}[{index}]")
            issues.extend(find_payload_quality_issues(value, f"{path}[{index}]"))
    elif isinstance(payload, str):
        stripped = payload.strip()
        if stripped.startswith("{") and ('"' in stripped and ":" in stripped):
            issues.append(f"embedded JSON-looking string at {path}")
        if stripped.startswith("//") or "\n//" in stripped:
            issues.append(f"comment-marker noise at {path}")
        for pattern in SUSPECT_NOISE_PATTERNS:
            if pattern in payload:
                issues.append(f"suspect noise token `{pattern}` at {path}")
                break
    return issues


def needs_reconcile(
    relevance_audit: dict[str, Any],
    counter_audit: dict[str, Any],
    final_result: dict[str, Any],
    min_confidence: float,
) -> bool:
    initial_verdict = relevance_audit.get("overall_initial_verdict")
    corrected_verdict = counter_audit.get("corrected_verdict")
    final_verdict = final_result.get("overall_verdict")
    already_reconciled = "reconciled_confidence" in final_result
    confidence = float(
        final_result.get("confidence", final_result.get("reconciled_confidence", 0.0)) or 0.0
    )
    return bool(
        ((not already_reconciled) and counter_audit.get("grade_inflation_detected"))
        or ((not already_reconciled) and initial_verdict and corrected_verdict and initial_verdict != corrected_verdict)
        or (corrected_verdict and final_verdict and corrected_verdict != final_verdict)
        or confidence < min_confidence
    )


def render_report(
    article: Document,
    article_map: dict[str, Any],
    theory_map: dict[str, Any],
    theory_implications: dict[str, Any] | None,
    final_result: dict[str, Any],
    initial_audit: dict[str, Any],
    counter_audit: dict[str, Any],
    document_label: str = "article",
) -> str:
    verdict = final_result.get("overall_verdict", "unknown")
    confidence = final_result.get("confidence", final_result.get("reconciled_confidence", 0.0))
    paragraph = final_result.get("one_paragraph_verdict", "")
    relevance_mode = final_result.get("relevance_mode", "unknown")
    contextual_relevance = final_result.get("contextual_relevance", [])
    arguments_for = final_result.get("arguments_for_theory", [])
    arguments_against = final_result.get("arguments_against_theory", [])
    cannot = final_result.get("what_article_cannot_adjudicate", [])
    recommended_use = final_result.get("recommended_use", "unknown")
    state_capital = final_result.get("state_capital_nexus_relevance", "")
    document_title = document_label.title()

    lines = [
        f"# {article.path.name}",
        "",
        f"- Verdict: `{verdict}`",
        f"- Confidence: `{confidence}`",
        f"- Relevance mode: `{relevance_mode}`",
        f"- Recommended use: `{recommended_use}`",
        "",
        "## Bottom Line",
        "",
        paragraph,
        "",
        f"## {document_title} Snapshot",
        "",
        f"- Summary: {article_map.get('summary', 'Not specified.')}",
    ]
    if document_label == "paper":
        lines.extend(
            [
                f"- Research question: {article_map.get('research_question', 'Not specified.')}",
                f"- Method or approach: {', '.join(article_map.get('method_or_approach', [])) or 'Not specified.'}",
                f"- Theoretical frameworks: {', '.join(article_map.get('theoretical_frameworks', [])) or 'Not specified.'}",
                f"- Empirical scope or case: {', '.join(article_map.get('empirical_scope_or_case', [])) or 'Not specified.'}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- Main claims: {', '.join(article_map.get('main_claims', [])) or 'Not specified.'}",
                "",
            ]
        )

    lines.extend(
        [
        "## Contextual Relevance",
        "",
        ]
    )
    if contextual_relevance:
        lines.extend(f"- {item}" for item in contextual_relevance)
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Arguments For The Theory",
            "",
        ]
    )
    if arguments_for:
        for item in arguments_for:
            lines.append(
                f"- [{item.get('strength', 'unknown')}] {item.get('argument', '').strip()} "
                f"(claims: {', '.join(item.get('claim_ids', [])) or 'n/a'})"
            )
    else:
        lines.append("- None.")

    lines.extend(["", "## Arguments Against The Theory", ""])
    if arguments_against:
        for item in arguments_against:
            lines.append(
                f"- [{item.get('strength', 'unknown')}] {item.get('argument', '').strip()} "
                f"(claims: {', '.join(item.get('claim_ids', [])) or 'n/a'})"
            )
    else:
        lines.append("- None.")

    lines.extend(["", f"## What The {document_title} Cannot Adjudicate", ""])
    if cannot:
        lines.extend(f"- {item}" for item in cannot)
    else:
        lines.append("- Not specified.")

    if theory_implications:
        lines.extend(["", "## Theory Implications", ""])
        lines.append(
            f"- Overall implication: {theory_implications.get('overall_implication', 'Not specified.')}"
        )
        lines.append(
            f"- Revision priority: {theory_implications.get('revision_priority', 'Not specified.')}"
        )
        lines.append(
            f"- Confidence: {theory_implications.get('confidence', 'Not specified.')}"
        )
        lines.extend(["", theory_implications.get("summary", ""), ""])
        claim_level = theory_implications.get("claim_level_implications", [])
        if claim_level:
            lines.append("### Claim-Level Implications")
            lines.append("")
            for item in claim_level:
                lines.append(
                    f"- {item.get('claim_id', 'n/a')}: {item.get('effect', 'unknown')} — {item.get('why', '').strip()}"
                )
                evidence = ", ".join(item.get("evidence_from_document", []))
                if evidence:
                    lines.append(f"  Evidence: {evidence}")
                proposed = item.get("proposed_revision", "").strip()
                if proposed:
                    lines.append(f"  Proposed revision: {proposed}")
        new_subclaims = theory_implications.get("new_subclaims", [])
        if new_subclaims:
            lines.extend(["", "### New Subclaims", ""])
            lines.extend(f"- {item}" for item in new_subclaims)
        open_questions = theory_implications.get("new_open_questions", [])
        if open_questions:
            lines.extend(["", "### New Open Questions", ""])
            lines.extend(f"- {item}" for item in open_questions)
        follow_up = theory_implications.get("recommended_follow_up", [])
        if follow_up:
            lines.extend(["", "### Recommended Follow-Up", ""])
            lines.extend(f"- {item}" for item in follow_up)

    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- State-capital nexus relevance: {state_capital or 'Not specified.'}",
            f"- Initial audit verdict: {initial_audit.get('overall_initial_verdict', 'unknown')}",
            f"- Counter-audit verdict: {counter_audit.get('corrected_verdict', 'unknown')}",
            f"- Theory thesis: {theory_map.get('thesis', '').strip()}",
            "",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def verdict_rank(verdict: str) -> int:
    order = {"relevant": 0, "marginal": 1, "contextual": 2, "irrelevant": 3}
    return order.get(verdict, 3)


if __name__ == "__main__":
    raise SystemExit(main())

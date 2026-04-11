from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from gemma.article_theory_analysis import (
    TheoryArticleRunner,
    chat_json_with_retries,
    load_documents,
    parse_json_payload,
    validate_payload_for_path,
)


class FakeClient:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.model = "fake-model"
        self.calls: list[list[dict[str, str]]] = []

    def chat_json(self, messages: list[dict[str, str]]) -> dict:
        assert messages
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("No fake responses left for chat_json")
        return self.responses.pop(0)


def test_load_documents_reads_txt_html_and_docx(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    cache = tmp_path / "cache"
    inputs.mkdir()

    (inputs / "note.txt").write_text("plain text", encoding="utf-8")
    (inputs / "page.html").write_text("<html><body><h1>Title</h1><p>Body</p></body></html>", encoding="utf-8")

    docx_path = inputs / "memo.docx"
    with zipfile.ZipFile(docx_path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
            <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:body>
                <w:p><w:r><w:t>First paragraph</w:t></w:r></w:p>
                <w:p><w:r><w:t>Second paragraph</w:t></w:r></w:p>
              </w:body>
            </w:document>
            """,
        )

    documents = load_documents(inputs, cache, max_chars=10_000)

    assert [document.slug for document in documents] == ["memo", "note", "page"]
    assert "First paragraph" in documents[0].text
    assert "plain text" == documents[1].text
    assert "Title" in documents[2].text
    assert (cache / "memo.txt").exists()


def test_parse_json_payload_accepts_fences_and_trailing_text() -> None:
    assert parse_json_payload('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_payload('prefix {"a": 1} trailing') == {"a": 1}


def test_chat_json_with_retries_repairs_malformed_json() -> None:
    calls: list[list[dict[str, str]]] = []
    responses = iter(['{"a": ', '{"a": 1}'])

    def fake_chat(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return next(responses)

    assert chat_json_with_retries(
        fake_chat,
        [{"role": "user", "content": "Return JSON."}],
        max_json_retries=1,
    ) == {"a": 1}
    assert len(calls) == 2
    assert "Repair the following malformed model response" in calls[1][0]["content"]


def test_runner_writes_outputs_and_reconcile_round(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "nlr").mkdir(parents=True)
    (workspace / "others").mkdir(parents=True)

    (workspace / "nlr" / "morozov.txt").write_text(
        "The theory says platform capitalism is still capitalism, not techno-feudalism.",
        encoding="utf-8",
    )
    (workspace / "others" / "wsj.txt").write_text(
        "The administration says AI data centers create jobs and growth, while critics want regulation.",
        encoding="utf-8",
    )

    fake_client = FakeClient(
        [
            {
                "thesis": "Capitalism has not turned into techno-feudalism.",
                "core_claims": [
                    {
                        "id": "C1",
                        "claim": "Digital firms still operate through capitalist accumulation.",
                        "why_it_matters": "The theory rejects a civilizational break.",
                        "support_requirements": ["Evidence about production, profit, or competition."],
                        "challenge_requirements": ["Evidence that rent displaces capitalist production."],
                        "indirect_relevance_hooks": ["AI investment rhetoric."],
                        "false_positive_matches": ["Generic talk about innovation or growth."],
                    }
                ],
                "conceptual_boundaries": [
                    {
                        "boundary": "State support is not feudal sovereignty.",
                        "explanation": "Policy support for firms is still compatible with capitalism.",
                    }
                ],
                "relevance_red_flags": ["Campaign rhetoric is not a theoretical test."],
                "secondary_themes": [
                    {
                        "id": "T1",
                        "theme": "state-capital nexus",
                        "why_articles_may_matter": "Policy stories can provide context.",
                        "typical_article_signals": ["official AI strategy"],
                    }
                ],
                "article_relevance_hooks": [
                    {
                        "hook": "data center investment",
                        "counts_as": "contextual",
                        "notes": "Useful but not decisive.",
                    }
                ],
                "open_questions": [],
            },
            {
                "relevance_tiers": [
                    {
                        "label": "contextual",
                        "definition": "Useful background.",
                        "recommended_use": "mention_in_passing",
                    }
                ],
                "relevance_questions": ["Does it illuminate a theory hook?"],
                "relevance_gates": [
                    {
                        "id": "G1",
                        "question": "Does the article address the economic mechanism at issue?",
                        "failure_means": "Treat as irrelevant or marginal.",
                    }
                ],
                "what_counts_as_real_evidence": ["Evidence about accumulation, rent, or commodity form."],
                "what_counts_as_contextual_or_illustrative_relevance": ["State-capital policy context."],
                "what_does_not_count": ["Loose political talking points."],
                "support_strength_scale": [
                    {"label": "none", "definition": "No support."},
                    {"label": "weak", "definition": "Only indirect or illustrative support."},
                    {"label": "moderate", "definition": "Some direct relevance."},
                    {"label": "strong", "definition": "Directly tests the claim."},
                ],
                "challenge_strength_scale": [
                    {"label": "none", "definition": "No challenge."},
                    {"label": "weak", "definition": "Minor tension only."},
                    {"label": "moderate", "definition": "Real pressure on the claim."},
                    {"label": "strong", "definition": "Direct contradiction."},
                ],
                "anti_grade_inflation_rule": "Do not call context proof.",
                "anti_false_negative_rule": "Do not erase useful context.",
                "default_verdict_rule": "If the article provides useful context, call it contextual.",
            },
            {
                "article_kind": "news",
                "summary": "A policy story about selling AI as growth-friendly before elections.",
                "main_claims": ["Officials emphasize growth and downplay job-loss concerns."],
                "evidence_or_facts": ["The article mentions layoffs and data center jobs."],
                "rhetorical_frames": ["Growth versus regulation."],
                "state_and_policy_content": ["The White House wants light-touch AI policy."],
                "capital_accumulation_content": ["Data center investment is discussed."],
                "labor_content": ["Job losses are acknowledged but minimized."],
                "infrastructure_and_supply_chain_content": ["Data centers are discussed."],
                "geopolitical_competition_content": ["China race rhetoric is discussed."],
                "possible_theory_hooks": ["state-capital nexus", "infrastructure investment"],
                "what_is_missing_for_theory_evaluation": ["No analysis of rent, profit, or commodity form."],
            },
            {
                "overall_initial_verdict": "marginal",
                "overall_reason": "There is weak contact through investment rhetoric and state-tech alignment.",
                "claim_assessments": [
                    {
                        "claim_id": "C1",
                        "engagement_type": "indirect",
                        "support_strength": "weak",
                        "challenge_strength": "none",
                        "support_points": ["The article treats AI as a site of accumulation and investment."],
                        "challenge_points": [],
                        "why_limited": "It never addresses the theory's core mechanisms directly.",
                    }
                ],
                "contextual_relevance_points": ["State-capital policy context."],
                "illustrative_relevance_points": ["Data-center rhetoric."],
                "article_level_points_for_theory": ["It weakly fits a capitalist-growth framing."],
                "article_level_points_against_theory": [],
                "irrelevance_reasons": ["The piece is a political-process story."],
                "confidence": 0.63,
            },
            {
                "grade_inflation_detected": True,
                "false_negative_detected": False,
                "problems_with_initial_audit": ["The audit overstated political rhetoric as theoretical evidence."],
                "missing_support_points": [],
                "missing_challenge_points": [],
                "missing_contextual_or_illustrative_points": [],
                "corrected_verdict": "irrelevant",
                "corrections_by_claim": [
                    {
                        "claim_id": "C1",
                        "corrected_support_strength": "weak",
                        "corrected_challenge_strength": "none",
                        "note": "The point of contact is illustrative rather than probative.",
                    }
                ],
                "confidence": 0.8,
            },
            {
                "overall_verdict": "marginal",
                "confidence": 0.55,
                "relevance_mode": "illustrative",
                "one_paragraph_verdict": "The article is mostly a weakly related policy story.",
                "contextual_relevance": ["State-capital policy context."],
                "arguments_for_theory": [
                    {
                        "strength": "weak",
                        "argument": "It assumes AI investment produces real growth, which fits a capitalist frame.",
                        "claim_ids": ["C1"],
                    }
                ],
                "arguments_against_theory": [],
                "what_article_cannot_adjudicate": ["It does not test rent versus profit."],
                "state_capital_nexus_relevance": "It is mildly relevant as a state-capital coordination example.",
                "recommended_use": "mention_in_passing",
            },
            {
                "overall_verdict": "irrelevant",
                "reconciled_confidence": 0.84,
                "relevance_mode": "illustrative",
                "one_paragraph_verdict": "The article is largely irrelevant to the theory and only weakly illustrative.",
                "contextual_relevance": ["State-capital policy context."],
                "arguments_for_theory": [
                    {
                        "strength": "weak",
                        "argument": "The article's growth framing is at least consistent with capitalist accumulation.",
                        "claim_ids": ["C1"],
                    }
                ],
                "arguments_against_theory": [],
                "what_article_cannot_adjudicate": ["It cannot decide whether digital revenues are rents or profits."],
                "state_capital_nexus_relevance": "A minor state-capital data point.",
                "recommended_use": "ignore",
            },
        ]
    )

    runner = TheoryArticleRunner(
        workspace=workspace,
        client=fake_client,
        max_reconcile_rounds=2,
        min_confidence=0.7,
    )

    result = runner.run()

    assert result["article_count"] == 1
    report_path = workspace / "outputs" / "wsj" / "report.md"
    assert report_path.exists()
    assert "largely irrelevant" in report_path.read_text(encoding="utf-8")
    assert (workspace / "outputs" / "wsj" / "04b_reconcile_round_1.json").exists()
    assert (workspace / "outputs" / "index.md").exists()


def test_articles_stage_reuses_preprocessed_theory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "nlr").mkdir(parents=True)
    (workspace / "others").mkdir(parents=True)
    theory_output = workspace / "outputs" / "_theory"
    theory_output.mkdir(parents=True)

    (workspace / "nlr" / "morozov.txt").write_text(
        "The theory says platform capitalism is still capitalism, not techno-feudalism.",
        encoding="utf-8",
    )
    (workspace / "others" / "article.txt").write_text(
        "A short article about an AI policy fight.",
        encoding="utf-8",
    )
    (theory_output / "01_theory_map.json").write_text(
        """{
          "thesis": "Capitalism has not turned into techno-feudalism.",
          "core_claims": [
            {
              "id": "C1",
              "claim": "Digital firms still operate through capitalist accumulation.",
              "why_it_matters": "It rejects the feudal thesis.",
              "support_requirements": ["Mechanism-level evidence."],
              "challenge_requirements": ["Mechanism-level contradiction."],
              "indirect_relevance_hooks": ["AI policy context."],
              "false_positive_matches": ["Generic AI policy news."]
            }
          ],
          "secondary_themes": [],
          "article_relevance_hooks": [],
          "conceptual_boundaries": [],
          "relevance_red_flags": [],
          "open_questions": []
        }""",
        encoding="utf-8",
    )
    (theory_output / "02_theory_rubric.json").write_text(
        """{
          "relevance_tiers": [
            {"label": "contextual", "definition": "Useful context.", "recommended_use": "mention_in_passing"}
          ],
          "relevance_questions": ["Does it illuminate a theory hook?"],
          "what_counts_as_real_evidence": ["Mechanism-level evidence."],
          "what_counts_as_contextual_or_illustrative_relevance": ["Policy context."],
          "what_does_not_count": ["Topical overlap."],
          "support_strength_scale": [{"label": "none", "definition": "No support."}],
          "challenge_strength_scale": [{"label": "none", "definition": "No challenge."}],
          "anti_grade_inflation_rule": "Do not inflate context.",
          "anti_false_negative_rule": "Do not erase context.",
          "default_verdict_rule": "Default to irrelevant."
        }""",
        encoding="utf-8",
    )

    fake_client = FakeClient(
        [
            {
                "article_kind": "news",
                "summary": "A short policy story.",
                "main_claims": ["Policy conflict."],
                "evidence_or_facts": [],
                "rhetorical_frames": [],
                "state_and_policy_content": [],
                "capital_accumulation_content": [],
                "labor_content": [],
                "infrastructure_and_supply_chain_content": [],
                "geopolitical_competition_content": [],
                "possible_theory_hooks": [],
                "what_is_missing_for_theory_evaluation": ["No mechanism-level evidence."],
            },
            {
                "overall_initial_verdict": "irrelevant",
                "overall_reason": "No mechanism-level contact.",
                "claim_assessments": [
                    {
                        "claim_id": "C1",
                        "engagement_type": "none",
                        "support_strength": "none",
                        "challenge_strength": "none",
                        "support_points": [],
                        "challenge_points": [],
                        "why_limited": "Only topical overlap.",
                    }
                ],
                "contextual_relevance_points": [],
                "illustrative_relevance_points": [],
                "article_level_points_for_theory": [],
                "article_level_points_against_theory": [],
                "irrelevance_reasons": ["No mechanism-level evidence."],
                "confidence": 0.9,
            },
            {
                "grade_inflation_detected": False,
                "false_negative_detected": False,
                "problems_with_initial_audit": [],
                "missing_support_points": [],
                "missing_challenge_points": [],
                "missing_contextual_or_illustrative_points": [],
                "corrected_verdict": "irrelevant",
                "corrections_by_claim": [
                    {
                        "claim_id": "C1",
                        "corrected_support_strength": "none",
                        "corrected_challenge_strength": "none",
                        "note": "No contact.",
                    }
                ],
                "confidence": 0.9,
            },
            {
                "overall_verdict": "irrelevant",
                "confidence": 0.9,
                "relevance_mode": "none",
                "one_paragraph_verdict": "The article is irrelevant to the theory.",
                "contextual_relevance": [],
                "arguments_for_theory": [],
                "arguments_against_theory": [],
                "what_article_cannot_adjudicate": ["Everything mechanism-level."],
                "state_capital_nexus_relevance": "None.",
                "recommended_use": "ignore",
            },
        ]
    )

    runner = TheoryArticleRunner(workspace=workspace, client=fake_client)

    result = runner.run(stage="articles")

    assert result["stage"] == "articles"
    assert result["article_count"] == 1
    assert len(fake_client.calls) == 4
    assert not fake_client.responses


def test_runner_records_failures_and_raises_by_default(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "nlr").mkdir(parents=True)
    (workspace / "others").mkdir(parents=True)
    theory_output = workspace / "outputs" / "_theory"
    theory_output.mkdir(parents=True)

    (workspace / "nlr" / "morozov.txt").write_text("theory", encoding="utf-8")
    (workspace / "others" / "article.txt").write_text("article", encoding="utf-8")
    (theory_output / "01_theory_map.json").write_text(
        '{"thesis":"x","core_claims":[]}',
        encoding="utf-8",
    )
    (theory_output / "02_theory_rubric.json").write_text(
        '{"relevance_tiers":[]}',
        encoding="utf-8",
    )

    fake_client = FakeClient([])
    runner = TheoryArticleRunner(
        workspace=workspace,
        client=fake_client,
        max_step_retries=0,
        allow_failures=False,
    )

    try:
        runner.run(stage="articles")
    except RuntimeError as exc:
        assert "article(s) failed" in str(exc)
    else:
        raise AssertionError("runner should fail when an article fails")

    assert (workspace / "outputs" / "failures.json").exists()


def test_stale_outputs_are_rebuilt(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "nlr").mkdir(parents=True)
    (workspace / "others").mkdir(parents=True)

    (workspace / "nlr" / "morozov.txt").write_text("theory", encoding="utf-8")
    (workspace / "others" / "article.txt").write_text("article", encoding="utf-8")

    article_dir = workspace / "outputs" / "article"
    article_dir.mkdir(parents=True)
    (article_dir / "01_article_map.json").write_text(
        '{"article_kind":"news","summary":"old"}',
        encoding="utf-8",
    )

    fake_client = FakeClient(
        [
            {
                "thesis": "Capitalism has not turned into techno-feudalism.",
                "core_claims": [],
                "secondary_themes": [],
                "article_relevance_hooks": [],
                "conceptual_boundaries": [],
                "relevance_red_flags": [],
                "open_questions": [],
            },
            {
                "relevance_tiers": [],
                "relevance_questions": [],
                "what_counts_as_real_evidence": [],
                "what_counts_as_contextual_or_illustrative_relevance": [],
                "what_does_not_count": [],
                "support_strength_scale": [],
                "challenge_strength_scale": [],
                "anti_grade_inflation_rule": "Do not inflate context.",
                "anti_false_negative_rule": "Do not erase context.",
                "default_verdict_rule": "Default to contextual when useful.",
            },
            {
                "article_kind": "news",
                "summary": "new",
                "main_claims": [],
                "evidence_or_facts": [],
                "rhetorical_frames": [],
                "state_and_policy_content": [],
                "capital_accumulation_content": [],
                "labor_content": [],
                "infrastructure_and_supply_chain_content": [],
                "geopolitical_competition_content": [],
                "possible_theory_hooks": [],
                "what_is_missing_for_theory_evaluation": [],
            },
            {
                "overall_initial_verdict": "contextual",
                "overall_reason": "Useful background.",
                "claim_assessments": [],
                "contextual_relevance_points": [],
                "illustrative_relevance_points": [],
                "article_level_points_for_theory": [],
                "article_level_points_against_theory": [],
                "irrelevance_reasons": [],
                "confidence": 0.8,
            },
            {
                "grade_inflation_detected": False,
                "false_negative_detected": False,
                "problems_with_initial_audit": [],
                "missing_support_points": [],
                "missing_challenge_points": [],
                "missing_contextual_or_illustrative_points": [],
                "corrected_verdict": "contextual",
                "corrections_by_claim": [],
                "confidence": 0.8,
            },
            {
                "overall_verdict": "contextual",
                "confidence": 0.8,
                "relevance_mode": "contextual",
                "one_paragraph_verdict": "Useful background.",
                "contextual_relevance": [],
                "arguments_for_theory": [],
                "arguments_against_theory": [],
                "what_article_cannot_adjudicate": [],
                "state_capital_nexus_relevance": "None.",
                "recommended_use": "mention_in_passing",
            },
        ]
    )

    runner = TheoryArticleRunner(workspace=workspace, client=fake_client, max_reconcile_rounds=0)

    runner.run()

    assert '"summary": "new"' in (article_dir / "01_article_map.json").read_text(encoding="utf-8")


def test_academic_profile_uses_separate_dirs_and_schema(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "nlr").mkdir(parents=True)
    (workspace / "articles").mkdir(parents=True)
    theory_output = workspace / "academic_outputs" / "_theory"
    theory_output.mkdir(parents=True)

    (workspace / "nlr" / "morozov.txt").write_text("theory", encoding="utf-8")
    (workspace / "articles" / "amazon-paper.txt").write_text("academic paper text", encoding="utf-8")
    (theory_output / "01_theory_map.json").write_text(
        """{
          "thesis": "Capitalism has not turned into techno-feudalism.",
          "core_claims": [],
          "secondary_themes": [],
          "article_relevance_hooks": [],
          "conceptual_boundaries": [],
          "relevance_red_flags": [],
          "open_questions": []
        }""",
        encoding="utf-8",
    )
    (theory_output / "02_theory_rubric.json").write_text(
        """{
          "relevance_tiers": [],
          "relevance_questions": [],
          "what_counts_as_real_evidence": [],
          "what_counts_as_contextual_or_illustrative_relevance": [],
          "what_does_not_count": [],
          "support_strength_scale": [],
          "challenge_strength_scale": [],
          "anti_grade_inflation_rule": "Do not inflate adjacency.",
          "anti_false_negative_rule": "Do not erase real conceptual contact.",
          "default_verdict_rule": "Default to marginal when a paper contributes indirectly."
        }""",
        encoding="utf-8",
    )

    fake_client = FakeClient(
        [
            {
                "article_kind": "academic",
                "summary": "A paper about Amazon and competition.",
                "research_question": "How should Amazon's market behavior be understood?",
                "main_claims": ["Amazon reveals tensions between monopoly and competition."],
                "method_or_approach": ["political-economy analysis"],
                "empirical_scope_or_case": ["Amazon"],
                "evidence_or_facts": ["Firm behavior and market structure examples."],
                "theoretical_frameworks": ["Marxian competition theory"],
                "explicit_theoretical_interventions": ["Critiques monopoly simplifications."],
                "section_level_moves": ["Sets up debate", "Reinterprets Amazon", "Draws implications"],
                "rhetorical_frames": ["Monopoly versus competition"],
                "state_and_policy_content": [],
                "capital_accumulation_content": ["Amazon as accumulation strategy."],
                "labor_content": ["Warehouse labor and control."],
                "infrastructure_and_supply_chain_content": ["Logistics integration."],
                "geopolitical_competition_content": [],
                "possible_theory_hooks": ["competition under digital capitalism"],
                "what_is_missing_for_theory_evaluation": ["No direct discussion of techno-feudalism."],
            },
            {
                "overall_initial_verdict": "marginal",
                "overall_reason": "The paper bears indirectly through competition categories.",
                "claim_assessments": [],
                "contextual_relevance_points": ["Competition theory literature."],
                "illustrative_relevance_points": ["Amazon as a case."],
                "article_level_points_for_theory": ["The paper contests monopoly simplifications."],
                "article_level_points_against_theory": [],
                "irrelevance_reasons": [],
                "confidence": 0.82,
            },
            {
                "grade_inflation_detected": False,
                "false_negative_detected": False,
                "problems_with_initial_audit": [],
                "missing_support_points": [],
                "missing_challenge_points": [],
                "missing_contextual_or_illustrative_points": [],
                "corrected_verdict": "marginal",
                "corrections_by_claim": [],
                "confidence": 0.83,
            },
            {
                "overall_verdict": "marginal",
                "confidence": 0.84,
                "relevance_mode": "illustrative",
                "one_paragraph_verdict": "The paper is a useful minor update on competition categories.",
                "contextual_relevance": ["It sharpens debate about monopoly and competition."],
                "arguments_for_theory": [
                    {
                        "strength": "weak",
                        "argument": "It supports treating Amazon as part of capitalist competition rather than a post-capitalist rupture.",
                        "claim_ids": ["C1"],
                    }
                ],
                "arguments_against_theory": [],
                "what_article_cannot_adjudicate": ["It does not directly evaluate Morozov's overall thesis."],
                "state_capital_nexus_relevance": "Minimal.",
                "recommended_use": "use_as_minor_update",
            },
        ]
    )

    runner = TheoryArticleRunner(
        workspace=workspace,
        client=fake_client,
        analysis_profile="academic",
        max_reconcile_rounds=0,
    )

    result = runner.run(stage="articles")

    assert result["article_count"] == 1
    assert (workspace / "academic_outputs" / "amazon-paper" / "report.md").exists()
    assert (workspace / "cache" / "articles" / "amazon-paper.txt").exists()
    index_md = (workspace / "academic_outputs" / "index.md").read_text(encoding="utf-8")
    assert "Academic Theory Analysis Index" in index_md
    assert "Papers analyzed: 1" in index_md


def test_validate_payload_for_academic_article_map_requires_extra_keys(tmp_path: Path) -> None:
    path = tmp_path / "01_article_map.json"
    payload = {
        "article_kind": "academic",
        "summary": "A paper summary.",
        "main_claims": [],
        "evidence_or_facts": [],
        "rhetorical_frames": [],
        "state_and_policy_content": [],
        "capital_accumulation_content": [],
        "labor_content": [],
        "infrastructure_and_supply_chain_content": [],
        "geopolitical_competition_content": [],
        "possible_theory_hooks": [],
        "what_is_missing_for_theory_evaluation": [],
    }

    with pytest.raises(ValueError):
        validate_payload_for_path(path, payload, analysis_profile="academic")

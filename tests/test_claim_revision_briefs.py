from __future__ import annotations

import json
from pathlib import Path

from gemma.claim_revision_briefs import build_claim_briefs


class FakeClient:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.model = "fake-model"
        self.calls: list[list[dict[str, str]]] = []

    def chat_json(self, messages: list[dict[str, str]]) -> dict:
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("No fake responses left for chat_json")
        return self.responses.pop(0)


def test_build_claim_briefs_writes_index_and_claim_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "outputs" / "_theory").mkdir(parents=True)
    (workspace / "outputs" / "doc-a").mkdir(parents=True)
    (workspace / "academic_outputs" / "_theory").mkdir(parents=True)

    theory_map = {
        "thesis": "Capitalism has not turned into techno-feudalism.",
        "core_claims": [
            {
                "id": "C1",
                "claim": "Digital firms remain capitalist producers.",
                "why_it_matters": "This is the main claim.",
                "support_requirements": ["Evidence of production and competition."],
                "challenge_requirements": ["Evidence of durable rentier displacement."],
            },
            {
                "id": "C2",
                "claim": "Techno-feudalism relies on bad genealogy.",
                "why_it_matters": "This is a secondary core claim.",
                "support_requirements": [],
                "challenge_requirements": [],
            },
        ],
    }
    (workspace / "outputs" / "_theory" / "01_theory_map.json").write_text(
        json.dumps(theory_map),
        encoding="utf-8",
    )
    (workspace / "academic_outputs" / "_theory" / "01_theory_map.json").write_text(
        json.dumps(theory_map),
        encoding="utf-8",
    )
    (workspace / "outputs" / "doc-a" / "05_theory_implications.json").write_text(
        json.dumps(
            {
                "overall_implication": "confirms",
                "summary": "Confirms C1.",
                "claim_level_implications": [
                    {
                        "claim_id": "C1",
                        "effect": "confirms",
                        "why": "Supports the capitalist-producer reading.",
                        "evidence_from_document": ["One support point."],
                        "proposed_revision": "Add this as confirming evidence.",
                    }
                ],
                "new_subclaims": [],
                "new_open_questions": [],
                "revision_priority": "low",
                "recommended_follow_up": [],
                "confidence": 0.8,
            }
        ),
        encoding="utf-8",
    )
    (workspace / "outputs" / "doc-a" / "04_final_judgment.json").write_text(
        json.dumps({"overall_verdict": "relevant", "recommended_use": "use_as_substantive_evidence"}),
        encoding="utf-8",
    )
    (workspace / "outputs" / "doc-a" / "01_article_map.json").write_text(
        json.dumps({"summary": "A document about capitalist production."}),
        encoding="utf-8",
    )

    client = FakeClient(
        [
            {
                "claim_id": "C1",
                "claim_text": "Digital firms remain capitalist producers.",
                "evidence_status": "thin",
                "overall_assessment": "confirm_but_specify",
                "summary": "The corpus confirms the claim but specifies a tighter mechanism.",
                "confirmation_points": ["Confirms the basic claim."],
                "qualification_points": [],
                "extension_points": ["Adds a mechanism through circulation time."],
                "pressure_points": [],
                "proposed_claim_revision": "Specify logistics and circulation-time compression as one route to platform dominance.",
                "proposed_subclaims": ["Platform dominance can arise through circulation-time compression."],
                "open_questions": ["Does this generalize beyond Amazon-like firms?"],
                "priority": "medium",
                "documents_most_material": ["doc-a"],
                "confidence": 0.86,
            }
        ]
    )

    result = build_claim_briefs(workspace=workspace, client=client, overwrite=True)

    assert result["claim_count"] == 2
    assert (workspace / "claim_briefs" / "index.json").exists()
    c1 = json.loads((workspace / "claim_briefs" / "C1.json").read_text(encoding="utf-8"))
    assert c1["overall_assessment"] == "confirm_but_specify"
    c2 = json.loads((workspace / "claim_briefs" / "C2.json").read_text(encoding="utf-8"))
    assert c2["evidence_status"] == "none"
    assert len(client.calls) == 1

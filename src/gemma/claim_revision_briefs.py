from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gemma.article_theory_analysis import (
    AnthropicClient,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_WORKSPACE,
    OllamaClient,
    sanitize_payload,
)

PROFILES = (
    ("journalistic", "outputs"),
    ("academic", "academic_outputs"),
)

REQUIRED_KEYS = {
    "claim_id",
    "claim_text",
    "evidence_status",
    "overall_assessment",
    "summary",
    "confirmation_points",
    "qualification_points",
    "extension_points",
    "pressure_points",
    "proposed_claim_revision",
    "proposed_subclaims",
    "open_questions",
    "priority",
    "documents_most_material",
    "confidence",
}


@dataclass(slots=True)
class ClaimSource:
    profile: str
    claim_id: str
    claim_text: str
    why_it_matters: str
    support_requirements: list[str]
    challenge_requirements: list[str]


def _json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_claim_brief(payload: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_KEYS - set(payload))
    if missing:
        raise ValueError(f"Missing claim brief keys: {', '.join(missing)}")


def load_claim_sources(workspace: Path) -> dict[str, list[ClaimSource]]:
    claim_sources: dict[str, list[ClaimSource]] = defaultdict(list)
    for profile, output_subdir in PROFILES:
        theory_map = _json(workspace / output_subdir / "_theory" / "01_theory_map.json") or {}
        for claim in theory_map.get("core_claims", []):
            claim_sources[claim["id"]].append(
                ClaimSource(
                    profile=profile,
                    claim_id=claim["id"],
                    claim_text=claim.get("claim", ""),
                    why_it_matters=claim.get("why_it_matters", ""),
                    support_requirements=list(claim.get("support_requirements", []) or []),
                    challenge_requirements=list(claim.get("challenge_requirements", []) or []),
                )
            )
    return dict(sorted(claim_sources.items()))


def load_claim_evidence(workspace: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile, output_subdir in PROFILES:
        root = workspace / output_subdir
        for implication_path in sorted(root.glob("*/05_theory_implications.json")):
            if implication_path.parent.name == "_theory":
                continue
            slug = implication_path.parent.name
            implication = _json(implication_path) or {}
            final_judgment = _json(implication_path.parent / "04_final_judgment.json") or {}
            article_map = _json(implication_path.parent / "01_article_map.json") or {}
            for item in implication.get("claim_level_implications", []) or []:
                claim_id = item.get("claim_id")
                if not claim_id:
                    continue
                grouped[claim_id].append(
                    {
                        "slug": slug,
                        "profile": profile,
                        "final_verdict": final_judgment.get("overall_verdict"),
                        "recommended_use": final_judgment.get("recommended_use"),
                        "document_summary": article_map.get("summary"),
                        "overall_implication": implication.get("overall_implication"),
                        "claim_effect": item.get("effect"),
                        "why": item.get("why"),
                        "evidence_from_document": item.get("evidence_from_document", []),
                        "proposed_revision": item.get("proposed_revision"),
                    }
                )
    return dict(sorted(grouped.items()))


def synthesize_claim_brief(
    client: OllamaClient | AnthropicClient,
    claim_id: str,
    claim_sources: list[ClaimSource],
    evidence_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not evidence_rows:
        source = claim_sources[0]
        payload = {
            "claim_id": claim_id,
            "claim_text": source.claim_text,
            "evidence_status": "none",
            "overall_assessment": "unclear",
            "summary": "No document-level implication artifacts currently bear on this claim, so there is no corpus-level revision brief yet.",
            "confirmation_points": [],
            "qualification_points": [],
            "extension_points": [],
            "pressure_points": [],
            "proposed_claim_revision": "",
            "proposed_subclaims": [],
            "open_questions": ["What evidence would be needed to test this claim directly?"],
            "priority": "low",
            "documents_most_material": [],
            "confidence": 1.0,
        }
        _validate_claim_brief(payload)
        return payload

    canonical = claim_sources[0]
    claim_context = [
        {
            "profile": source.profile,
            "claim_text": source.claim_text,
            "why_it_matters": source.why_it_matters,
            "support_requirements": source.support_requirements,
            "challenge_requirements": source.challenge_requirements,
        }
        for source in claim_sources
    ]
    prompt = f"""
You are synthesizing a corpus-level theory revision brief for one claim in a political-economy theory map.

Your task:
- Read the claim metadata and the document-level implication rows below.
- Produce one clean claim brief that says whether the corpus mainly confirms, qualifies, extends, pressures, or leaves the claim unclear.
- Be conservative. Do not invent revision pressure if the evidence only confirms the claim.
- The output should help revise the theory map itself, not merely summarize documents.

Return JSON with exactly these keys:
{{
  "claim_id": "string",
  "claim_text": "string",
  "evidence_status": "none|thin|mixed|substantial",
  "overall_assessment": "stable|confirm_but_specify|extend|qualify|revise|split|unclear",
  "summary": "string",
  "confirmation_points": ["string"],
  "qualification_points": ["string"],
  "extension_points": ["string"],
  "pressure_points": ["string"],
  "proposed_claim_revision": "string",
  "proposed_subclaims": ["string"],
  "open_questions": ["string"],
  "priority": "low|medium|high",
  "documents_most_material": ["string"],
  "confidence": 0.0
}}

Claim id: {claim_id}

Claim metadata by profile:
{json.dumps(claim_context, ensure_ascii=False, indent=2)}

Document-level implication rows:
{json.dumps(evidence_rows, ensure_ascii=False, indent=2)}
""".strip()
    payload = sanitize_payload(client.chat_json([{"role": "user", "content": prompt}]))
    _validate_claim_brief(payload)
    return payload


def build_claim_briefs(
    workspace: Path,
    client: OllamaClient | AnthropicClient,
    overwrite: bool = False,
    claim_id_filter: str | None = None,
) -> dict[str, Any]:
    claim_sources = load_claim_sources(workspace)
    claim_evidence = load_claim_evidence(workspace)
    output_dir = workspace / "claim_briefs"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []

    for claim_id, sources in claim_sources.items():
        if claim_id_filter and claim_id != claim_id_filter:
            continue
        out_path = output_dir / f"{claim_id}.json"
        if out_path.exists() and not overwrite:
            payload = _json(out_path) or {}
        else:
            payload = synthesize_claim_brief(client, claim_id, sources, claim_evidence.get(claim_id, []))
            _write_json(out_path, payload)
        index_rows.append(
            {
                "claim_id": claim_id,
                "claim_text": payload.get("claim_text", sources[0].claim_text),
                "overall_assessment": payload.get("overall_assessment"),
                "priority": payload.get("priority"),
                "evidence_status": payload.get("evidence_status"),
                "confidence": payload.get("confidence"),
                "document_count": len(claim_evidence.get(claim_id, [])),
                "path": str(out_path),
            }
        )

    index_payload = {
        "claim_count": len(index_rows),
        "claims": sorted(index_rows, key=lambda row: (row["claim_id"])),
    }
    _write_json(output_dir / "index.json", index_payload)
    return index_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build corpus-level claim revision briefs from 05 implication artifacts.")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--provider", choices=("ollama", "anthropic"), default="ollama")
    parser.add_argument("--model")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--anthropic-url", default="https://api.anthropic.com")
    parser.add_argument("--anthropic-api-key-env", default="ANTHROPIC_API_KEY")
    parser.add_argument("--anthropic-max-tokens", type=int, default=16_000)
    parser.add_argument("--anthropic-effort", choices=("low", "medium", "high", "max"), default="max")
    parser.add_argument("--anthropic-thinking", choices=("adaptive", "disabled"), default="adaptive")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--claim-id")
    args = parser.parse_args()

    if args.provider == "anthropic":
        client = AnthropicClient(
            model=args.model or DEFAULT_ANTHROPIC_MODEL,
            api_key_env=args.anthropic_api_key_env,
            base_url=args.anthropic_url,
            max_tokens=args.anthropic_max_tokens,
            effort=args.anthropic_effort,
            thinking_type=args.anthropic_thinking,
        )
    else:
        client = OllamaClient(
            model=args.model or DEFAULT_OLLAMA_MODEL,
            base_url=args.ollama_url,
        )

    result = build_claim_briefs(
        workspace=args.workspace,
        client=client,
        overwrite=args.overwrite,
        claim_id_filter=args.claim_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

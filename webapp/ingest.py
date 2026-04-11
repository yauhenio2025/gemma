"""Ingest JSON analysis artifacts into the SQLite database."""

import json
import logging
import sqlite3
from pathlib import Path

from webapp.logging_config import setup_logging
from webapp.schema import DB_PATH, init_db

log = logging.getLogger("ingest")

WORKSPACE = Path(__file__).parent.parent / "article_theory_workspace"

PROFILES = [
    ("journalistic", WORKSPACE / "outputs"),
    ("academic", WORKSPACE / "academic_outputs"),
]


def _json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Failed to read %s: %s", path, e)
        return None


def _j(obj) -> str:
    """Serialize to JSON string for storage."""
    if obj is None:
        return "[]"
    return json.dumps(obj, ensure_ascii=False)


def ingest_theory(conn: sqlite3.Connection, profile: str, output_dir: Path) -> int | None:
    theory_dir = output_dir / "_theory"
    theory_map_path = theory_dir / "01_theory_map.json"
    theory_map = _json(theory_map_path)
    if not theory_map:
        log.warning("No theory map at %s", theory_map_path)
        return None

    rubric_path = theory_dir / "02_theory_rubric.json"
    rubric = _json(rubric_path)

    cur = conn.execute(
        "INSERT OR REPLACE INTO theory (id, profile, thesis, raw_path) VALUES ("
        "(SELECT id FROM theory WHERE profile = ?), ?, ?, ?)",
        (profile, profile, theory_map.get("thesis", ""), str(theory_map_path)),
    )
    theory_id = cur.lastrowid

    # Claims
    conn.execute("DELETE FROM claim WHERE theory_id = ?", (theory_id,))
    for c in theory_map.get("core_claims", []):
        conn.execute(
            "INSERT INTO claim (theory_id, claim_id, claim_text, why_it_matters, "
            "support_requirements, challenge_requirements, indirect_relevance_hooks, "
            "false_positive_matches) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                theory_id, c["id"], c["claim"],
                c.get("why_it_matters"),
                _j(c.get("support_requirements")),
                _j(c.get("challenge_requirements")),
                _j(c.get("indirect_relevance_hooks")),
                _j(c.get("false_positive_matches")),
            ),
        )

    # Themes
    conn.execute("DELETE FROM theme WHERE theory_id = ?", (theory_id,))
    for t in theory_map.get("secondary_themes", []):
        conn.execute(
            "INSERT INTO theme (theory_id, theme_id, theme_text, why_articles_may_matter, "
            "typical_signals) VALUES (?, ?, ?, ?, ?)",
            (
                theory_id, t["id"], t["theme"],
                t.get("why_articles_may_matter"),
                _j(t.get("typical_article_signals")),
            ),
        )

    # Conceptual boundaries
    conn.execute("DELETE FROM conceptual_boundary WHERE theory_id = ?", (theory_id,))
    for b in theory_map.get("conceptual_boundaries", []):
        conn.execute(
            "INSERT INTO conceptual_boundary (theory_id, boundary, explanation) "
            "VALUES (?, ?, ?)",
            (theory_id, b["boundary"], b.get("explanation")),
        )

    # Relevance hooks
    conn.execute("DELETE FROM relevance_hook WHERE theory_id = ?", (theory_id,))
    for h in theory_map.get("article_relevance_hooks", []):
        conn.execute(
            "INSERT INTO relevance_hook (theory_id, hook, counts_as, notes) "
            "VALUES (?, ?, ?, ?)",
            (theory_id, h["hook"], h.get("counts_as"), h.get("notes")),
        )

    # Rubric tiers
    if rubric:
        conn.execute("DELETE FROM relevance_tier WHERE theory_id = ?", (theory_id,))
        for tier in rubric.get("relevance_tiers", []):
            conn.execute(
                "INSERT INTO relevance_tier (theory_id, label, definition, recommended_use) "
                "VALUES (?, ?, ?, ?)",
                (theory_id, tier["label"], tier.get("definition"), tier.get("recommended_use")),
            )

    log.info("Ingested theory for profile=%s (id=%d)", profile, theory_id)
    return theory_id


def ingest_document(conn: sqlite3.Connection, profile: str, slug: str,
                    doc_dir: Path, index_entry: dict, model: str) -> int:
    # Read all steps
    article_map = _json(doc_dir / "01_article_map.json")
    relevance_audit = _json(doc_dir / "02_relevance_audit.json")
    counter_audit = _json(doc_dir / "03_counter_audit.json")
    final_judgment = _json(doc_dir / "04_final_judgment.json")

    # Find reconcile rounds
    reconcile_paths = sorted(doc_dir.glob("04b_reconcile_round_*.json"))
    reconcile_data = [(p, _json(p)) for p in reconcile_paths]
    reconcile_data = [(p, d) for p, d in reconcile_data if d is not None]

    am = article_map or {}
    ra = relevance_audit or {}
    ca = counter_audit or {}
    fj = final_judgment or {}

    last_reconcile_confidence = None
    if reconcile_data:
        last_round = reconcile_data[-1][1]
        last_reconcile_confidence = last_round.get("reconciled_confidence") or last_round.get("confidence")

    # Upsert document
    conn.execute("DELETE FROM document WHERE slug = ?", (slug,))
    cur = conn.execute(
        """INSERT INTO document (
            profile, slug, article_path, report_path, output_dir, model,
            index_verdict, index_confidence, index_recommended_use,
            article_kind, summary, main_claims, evidence_or_facts,
            possible_theory_hooks, research_question, method_or_approach,
            empirical_scope_or_case, theoretical_frameworks,
            initial_verdict, initial_reason, initial_confidence,
            contextual_relevance_points, illustrative_relevance_points,
            article_level_for_theory, article_level_against_theory,
            irrelevance_reasons,
            grade_inflation_detected, false_negative_detected,
            counter_problems, missing_support_points, missing_challenge_points,
            corrected_verdict, counter_confidence,
            final_verdict, final_confidence, relevance_mode,
            one_paragraph_verdict, contextual_relevance, cannot_adjudicate,
            state_capital_nexus, recommended_use,
            has_reconcile, reconcile_rounds, reconcile_confidence
        ) VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?,
            ?,
            ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?, ?
        )""",
        (
            profile, slug,
            index_entry.get("article_path"), index_entry.get("report_path"),
            str(doc_dir), model,
            index_entry.get("verdict"), index_entry.get("confidence"),
            index_entry.get("recommended_use"),
            # article map
            am.get("article_kind"), am.get("summary"),
            _j(am.get("main_claims")), _j(am.get("evidence_or_facts")),
            _j(am.get("possible_theory_hooks")),
            am.get("research_question"), _j(am.get("method_or_approach")),
            _j(am.get("empirical_scope_or_case")), _j(am.get("theoretical_frameworks")),
            # relevance audit
            ra.get("overall_initial_verdict"), ra.get("overall_reason"),
            ra.get("confidence"),
            _j(ra.get("contextual_relevance_points")),
            _j(ra.get("illustrative_relevance_points")),
            _j(ra.get("article_level_points_for_theory")),
            _j(ra.get("article_level_points_against_theory")),
            _j(ra.get("irrelevance_reasons")),
            # counter audit
            1 if ca.get("grade_inflation_detected") else 0,
            1 if ca.get("false_negative_detected") else 0,
            _j(ca.get("problems_with_initial_audit")),
            _j(ca.get("missing_support_points")),
            _j(ca.get("missing_challenge_points")),
            ca.get("corrected_verdict"), ca.get("confidence"),
            # final judgment
            fj.get("overall_verdict"), fj.get("confidence"),
            fj.get("relevance_mode"), fj.get("one_paragraph_verdict"),
            _j(fj.get("contextual_relevance")),
            _j(fj.get("what_article_cannot_adjudicate")),
            fj.get("state_capital_nexus_relevance"), fj.get("recommended_use"),
            # reconcile
            1 if reconcile_data else 0,
            len(reconcile_data),
            last_reconcile_confidence,
        ),
    )
    doc_id = cur.lastrowid

    # Claim assessments from relevance audit
    for ca_entry in ra.get("claim_assessments", []):
        conn.execute(
            """INSERT INTO claim_assessment (
                document_id, claim_id, stage, engagement_type,
                support_strength, challenge_strength,
                support_points, challenge_points, why_limited
            ) VALUES (?, ?, 'audit', ?, ?, ?, ?, ?, ?)""",
            (
                doc_id, ca_entry.get("claim_id"),
                ca_entry.get("engagement_type"),
                ca_entry.get("support_strength"),
                ca_entry.get("challenge_strength"),
                _j(ca_entry.get("support_points")),
                _j(ca_entry.get("challenge_points")),
                ca_entry.get("why_limited"),
            ),
        )

    # Claim corrections from counter audit
    for cc in ca.get("corrections_by_claim", []):
        conn.execute(
            """INSERT INTO claim_assessment (
                document_id, claim_id, stage,
                corrected_support_strength, corrected_challenge_strength, note
            ) VALUES (?, ?, 'counter', ?, ?, ?)""",
            (
                doc_id, cc.get("claim_id"),
                cc.get("corrected_support_strength"),
                cc.get("corrected_challenge_strength"),
                cc.get("note"),
            ),
        )

    # Arguments from final judgment
    def _insert_args(args_list, direction, source="final"):
        for arg in (args_list or []):
            cur2 = conn.execute(
                "INSERT INTO argument (document_id, direction, strength, argument, source) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc_id, direction, arg.get("strength"), arg.get("argument"), source),
            )
            arg_id = cur2.lastrowid
            for cid in (arg.get("claim_ids") or []):
                conn.execute(
                    "INSERT INTO argument_claim_link (argument_id, claim_id) VALUES (?, ?)",
                    (arg_id, cid),
                )

    _insert_args(fj.get("arguments_for_theory"), "for", "final")
    _insert_args(fj.get("arguments_against_theory"), "against", "final")

    # Reconcile rounds
    for i, (rp, rd) in enumerate(reconcile_data, 1):
        conn.execute(
            "INSERT INTO reconcile_round (document_id, round_num, raw_json, verdict, confidence, raw_path) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                doc_id, i, json.dumps(rd, ensure_ascii=False),
                rd.get("overall_verdict"),
                rd.get("reconciled_confidence") or rd.get("confidence"),
                str(rp),
            ),
        )
        _insert_args(rd.get("arguments_for_theory"), "for", f"reconcile_{i}")
        _insert_args(rd.get("arguments_against_theory"), "against", f"reconcile_{i}")

    # Store raw step JSON for tab rendering
    steps = [
        ("01_article_map", doc_dir / "01_article_map.json", article_map),
        ("02_relevance_audit", doc_dir / "02_relevance_audit.json", relevance_audit),
        ("03_counter_audit", doc_dir / "03_counter_audit.json", counter_audit),
        ("04_final_judgment", doc_dir / "04_final_judgment.json", final_judgment),
    ]
    for step_name, step_path, step_data in steps:
        if step_data:
            conn.execute(
                "INSERT INTO analysis_step (document_id, step_name, raw_json, raw_path) "
                "VALUES (?, ?, ?, ?)",
                (doc_id, step_name, json.dumps(step_data, ensure_ascii=False), str(step_path)),
            )
    for rp, rd in reconcile_data:
        conn.execute(
            "INSERT INTO analysis_step (document_id, step_name, raw_json, raw_path) "
            "VALUES (?, ?, ?, ?)",
            (doc_id, f"04b_reconcile_{rp.stem.split('_')[-1]}", json.dumps(rd, ensure_ascii=False), str(rp)),
        )

    return doc_id


def run_ingest(db_path: Path | None = None):
    setup_logging()
    path = db_path or DB_PATH
    if path.exists():
        path.unlink()
        log.info("Removed existing database at %s", path)

    conn = init_db(path)
    total = 0

    for profile, output_dir in PROFILES:
        if not output_dir.exists():
            log.warning("Output directory not found: %s", output_dir)
            continue

        ingest_theory(conn, profile, output_dir)

        index_path = output_dir / "index.json"
        index_data = _json(index_path)
        if not index_data:
            log.warning("No index.json at %s", index_path)
            continue

        model = index_data.get("model", "unknown")

        for entry in index_data.get("articles", []):
            slug = entry["article_slug"]
            doc_dir = output_dir / slug
            if not doc_dir.is_dir():
                log.warning("Missing output directory for %s", slug)
                continue

            doc_id = ingest_document(conn, profile, slug, doc_dir, entry, model)
            log.info("Ingested %s (id=%d, profile=%s)", slug, doc_id, profile)
            total += 1

    conn.commit()
    conn.close()
    log.info("Ingestion complete: %d documents into %s", total, path)
    return total


if __name__ == "__main__":
    run_ingest()

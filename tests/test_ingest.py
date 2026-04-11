"""Tests for the ingestion pipeline."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from webapp.ingest import run_ingest
from webapp.schema import init_db


@pytest.fixture
def db_path(tmp_path):
    """Run a full ingest and return the database path."""
    path = tmp_path / "test.db"
    run_ingest(db_path=path)
    return path


@pytest.fixture
def db(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def test_ingest_creates_db(db_path):
    assert db_path.exists()
    assert db_path.stat().st_size > 0


def test_ingest_document_count(db):
    count = db.execute("SELECT COUNT(*) FROM document").fetchone()[0]
    assert count >= 20  # 18 journalistic (minus 2 missing dirs) + 3 academic


def test_ingest_theory_profiles(db):
    profiles = [r[0] for r in db.execute("SELECT profile FROM theory ORDER BY profile").fetchall()]
    assert "academic" in profiles
    assert "journalistic" in profiles


def test_ingest_claims_exist(db):
    count = db.execute("SELECT COUNT(*) FROM claim").fetchone()[0]
    assert count >= 6  # C1-C6 for at least one profile


def test_ingest_themes_exist(db):
    count = db.execute("SELECT COUNT(*) FROM theme").fetchone()[0]
    assert count >= 7  # T1-T7 for at least one profile


def test_ingest_arguments_exist(db):
    count = db.execute("SELECT COUNT(*) FROM argument").fetchone()[0]
    assert count > 0


def test_ingest_claim_assessments_exist(db):
    count = db.execute("SELECT COUNT(*) FROM claim_assessment").fetchone()[0]
    assert count > 0


def test_ingest_analysis_steps_exist(db):
    count = db.execute("SELECT COUNT(*) FROM analysis_step").fetchone()[0]
    assert count > 0


def test_ingest_document_fields_populated(db):
    doc = db.execute(
        "SELECT * FROM document WHERE slug = ?",
        ("scmp-com-palantir-books-its-first-us1-billion-in-quarterly-sales-and-dodges-doge-axe",),
    ).fetchone()
    assert doc is not None
    assert doc["profile"] == "journalistic"
    assert doc["final_verdict"] == "marginal"
    assert doc["final_confidence"] == 1.0
    assert doc["recommended_use"] == "use_as_minor_update"
    assert doc["one_paragraph_verdict"] is not None
    assert len(doc["one_paragraph_verdict"]) > 50


def test_ingest_reconcile_round(db):
    doc = db.execute("SELECT * FROM document WHERE slug = 's00146-026-02990-2'").fetchone()
    assert doc is not None
    assert doc["has_reconcile"] == 1
    assert doc["reconcile_rounds"] == 1

    rounds = db.execute(
        "SELECT * FROM reconcile_round WHERE document_id = ?", (doc["id"],)
    ).fetchall()
    assert len(rounds) == 1
    assert rounds[0]["verdict"] == "contextual"


def test_ingest_argument_claim_links(db):
    links = db.execute("SELECT COUNT(*) FROM argument_claim_link").fetchone()[0]
    assert links > 0

    # Verify a known link: Palantir article supports C3
    doc = db.execute(
        "SELECT id FROM document WHERE slug LIKE '%palantir%'"
    ).fetchone()
    assert doc is not None
    args = db.execute(
        "SELECT a.id, a.direction FROM argument a WHERE a.document_id = ?",
        (doc["id"],),
    ).fetchall()
    assert len(args) > 0
    for_args = [a for a in args if a["direction"] == "for"]
    assert len(for_args) > 0
    # Check C3 is linked
    c3_links = db.execute(
        "SELECT acl.claim_id FROM argument_claim_link acl "
        "JOIN argument a ON a.id = acl.argument_id "
        "WHERE a.document_id = ?",
        (doc["id"],),
    ).fetchall()
    claim_ids = [r[0] for r in c3_links]
    assert "C3" in claim_ids


def test_ingest_academic_profile(db):
    academic = db.execute(
        "SELECT * FROM document WHERE profile = 'academic'"
    ).fetchall()
    assert len(academic) == 3
    slugs = [r["slug"] for r in academic]
    assert "maher-aquanno-2026-monopoly-or-competition-unraveling-the-amazon-paradox" in slugs


def test_ingest_theory_implications_exist(db):
    count = db.execute(
        "SELECT COUNT(*) FROM document WHERE has_theory_implications = 1"
    ).fetchone()[0]
    assert count >= 2

    amazon = db.execute(
        "SELECT * FROM document WHERE slug = ?",
        ("maher-aquanno-2026-monopoly-or-competition-unraveling-the-amazon-paradox",),
    ).fetchone()
    assert amazon is not None
    assert amazon["has_theory_implications"] == 1
    assert amazon["implication_overall"] in {"confirms", "qualifies", "extends", "pressures", "mixed"}


def test_ingest_claim_briefs_exist(db):
    count = db.execute("SELECT COUNT(*) FROM claim_brief").fetchone()[0]
    assert count >= 5

    c1 = db.execute("SELECT * FROM claim_brief WHERE claim_id = 'C1'").fetchone()
    assert c1 is not None
    assert c1["summary"] is not None

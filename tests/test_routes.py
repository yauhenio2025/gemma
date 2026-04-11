"""Tests for the webapp routes."""

import pytest
from fastapi.testclient import TestClient

from webapp.ingest import run_ingest


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Create a test client with ingested data."""
    db_path = tmp_path_factory.mktemp("data") / "test.db"
    run_ingest(db_path=db_path)

    # Patch the DB path before importing app
    import webapp.db
    original_path = webapp.db.DB_PATH
    webapp.db.DB_PATH = db_path

    from webapp.app import app
    with TestClient(app) as c:
        yield c

    webapp.db.DB_PATH = original_path


def test_home(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Research Dashboard" in resp.text
    assert "21" in resp.text or "Documents" in resp.text


def test_theory(client):
    resp = client.get("/theory")
    assert resp.status_code == 200
    assert "Thesis" in resp.text
    assert "Core Claims" in resp.text
    assert "C1" in resp.text


def test_corpus(client):
    resp = client.get("/corpus")
    assert resp.status_code == 200
    assert "Corpus" in resp.text
    assert "scmp-com" in resp.text


def test_corpus_filter_verdict(client):
    resp = client.get("/corpus?verdict=marginal")
    assert resp.status_code == 200


def test_corpus_filter_profile(client):
    resp = client.get("/corpus?profile=academic")
    assert resp.status_code == 200


def test_corpus_search(client):
    resp = client.get("/corpus?q=palantir")
    assert resp.status_code == 200
    assert "palantir" in resp.text.lower()


def test_document_detail(client):
    resp = client.get("/doc/scmp-com-palantir-books-its-first-us1-billion-in-quarterly-sales-and-dodges-doge-axe")
    assert resp.status_code == 200
    assert "Verdict Summary" in resp.text
    assert "marginal" in resp.text
    assert "C3" in resp.text


def test_document_not_found(client):
    resp = client.get("/doc/nonexistent-slug")
    assert resp.status_code == 404


def test_claims_list(client):
    resp = client.get("/claims")
    assert resp.status_code == 200
    assert "Claim Explorer" in resp.text
    assert "C1" in resp.text


def test_claim_detail(client):
    resp = client.get("/claim/C3")
    assert resp.status_code == 200
    assert "C3" in resp.text
    assert "Supporting Documents" in resp.text


def test_claim_not_found(client):
    resp = client.get("/claim/C99")
    assert resp.status_code == 404


def test_implications(client):
    resp = client.get("/implications")
    assert resp.status_code == 200
    assert "Implications" in resp.text
    assert "Evidence Accumulation" in resp.text


def test_review(client):
    resp = client.get("/review")
    assert resp.status_code == 200
    assert "Review Queue" in resp.text


def test_review_challenged_tab(client):
    resp = client.get("/review?tab=challenged")
    assert resp.status_code == 200


def test_static_css(client):
    resp = client.get("/static/style.css")
    assert resp.status_code == 200
    assert "body" in resp.text

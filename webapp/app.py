"""FastAPI research browser for theory-analysis artifacts."""

import json
import logging
from pathlib import Path

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp.db import get_db, row_to_dict, rows_to_dicts, parse_json_field
from webapp.logging_config import setup_logging

setup_logging()
log = logging.getLogger("webapp")

app = FastAPI(title="Gemma Research Browser", version="1.0.0")

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
templates.env.filters["fromjson"] = lambda v: parse_json_field(v)
templates.env.filters["tojson_pretty"] = lambda v: json.dumps(
    v if isinstance(v, (dict, list)) else parse_json_field(v), indent=2, ensure_ascii=False
)


def _db():
    return get_db()


def _render(request: Request, template: str, context: dict):
    return templates.TemplateResponse(request, template, context)


# ── Home ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    db = _db()
    stats = {
        "total": db.execute("SELECT COUNT(*) FROM document").fetchone()[0],
        "by_verdict": rows_to_dicts(db.execute(
            "SELECT final_verdict, COUNT(*) as cnt FROM document GROUP BY final_verdict ORDER BY cnt DESC"
        ).fetchall()),
        "by_profile": rows_to_dicts(db.execute(
            "SELECT profile, COUNT(*) as cnt FROM document GROUP BY profile ORDER BY cnt DESC"
        ).fetchall()),
        "by_recommended_use": rows_to_dicts(db.execute(
            "SELECT recommended_use, COUNT(*) as cnt FROM document GROUP BY recommended_use ORDER BY cnt DESC"
        ).fetchall()),
        "with_arguments_against": db.execute(
            "SELECT COUNT(DISTINCT document_id) FROM argument WHERE direction='against'"
        ).fetchone()[0],
        "with_reconcile": db.execute(
            "SELECT COUNT(*) FROM document WHERE has_reconcile = 1"
        ).fetchone()[0],
    }
    db.close()
    return _render(request, "home.html", {"stats": stats})


# ── Theory ────────────────────────────────────────────────────────────────

@app.get("/theory", response_class=HTMLResponse)
async def theory_page(request: Request):
    db = _db()
    theories = rows_to_dicts(db.execute("SELECT * FROM theory").fetchall())
    data = {}
    for t in theories:
        tid = t["id"]
        data[t["profile"]] = {
            "theory": t,
            "claims": rows_to_dicts(db.execute(
                "SELECT * FROM claim WHERE theory_id = ? ORDER BY claim_id", (tid,)
            ).fetchall()),
            "themes": rows_to_dicts(db.execute(
                "SELECT * FROM theme WHERE theory_id = ? ORDER BY theme_id", (tid,)
            ).fetchall()),
            "boundaries": rows_to_dicts(db.execute(
                "SELECT * FROM conceptual_boundary WHERE theory_id = ?", (tid,)
            ).fetchall()),
            "tiers": rows_to_dicts(db.execute(
                "SELECT * FROM relevance_tier WHERE theory_id = ? ORDER BY id", (tid,)
            ).fetchall()),
            "hooks": rows_to_dicts(db.execute(
                "SELECT * FROM relevance_hook WHERE theory_id = ?", (tid,)
            ).fetchall()),
        }
    db.close()
    return _render(request, "theory.html", {"data": data})


# ── Corpus ────────────────────────────────────────────────────────────────

@app.get("/corpus", response_class=HTMLResponse)
async def corpus_page(
    request: Request,
    profile: str | None = None,
    verdict: str | None = None,
    recommended_use: str | None = None,
    claim_id: str | None = None,
    has_against: str | None = None,
    has_reconcile: str | None = None,
    q: str | None = None,
    sort: str = "slug",
    order: str = "asc",
):
    db = _db()
    where_clauses = []
    params = []

    if profile:
        where_clauses.append("d.profile = ?")
        params.append(profile)
    if verdict:
        where_clauses.append("d.final_verdict = ?")
        params.append(verdict)
    if recommended_use:
        where_clauses.append("d.recommended_use = ?")
        params.append(recommended_use)
    if has_against == "1":
        where_clauses.append("d.id IN (SELECT DISTINCT document_id FROM argument WHERE direction='against')")
    if has_reconcile == "1":
        where_clauses.append("d.has_reconcile = 1")
    if claim_id:
        where_clauses.append(
            "d.id IN (SELECT DISTINCT a.document_id FROM argument a "
            "JOIN argument_claim_link acl ON acl.argument_id = a.id WHERE acl.claim_id = ?)"
        )
        params.append(claim_id)
    if q:
        where_clauses.append("(d.slug LIKE ? OR d.summary LIKE ? OR d.one_paragraph_verdict LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like, like])

    where = " AND ".join(where_clauses) if where_clauses else "1=1"

    allowed_sorts = {"slug", "profile", "final_verdict", "final_confidence", "recommended_use"}
    if sort not in allowed_sorts:
        sort = "slug"
    order_dir = "DESC" if order == "desc" else "ASC"

    docs = rows_to_dicts(db.execute(
        f"SELECT d.* FROM document d WHERE {where} ORDER BY d.{sort} {order_dir}",
        params,
    ).fetchall())

    # Counts for filter sidebar
    counts = {
        "by_verdict": rows_to_dicts(db.execute(
            "SELECT final_verdict, COUNT(*) as cnt FROM document GROUP BY final_verdict"
        ).fetchall()),
        "by_profile": rows_to_dicts(db.execute(
            "SELECT profile, COUNT(*) as cnt FROM document GROUP BY profile"
        ).fetchall()),
    }

    # Claim IDs for filter dropdown
    claim_ids = [r[0] for r in db.execute(
        "SELECT DISTINCT claim_id FROM claim ORDER BY claim_id"
    ).fetchall()]

    db.close()
    return _render(request, "corpus.html", {
        "docs": docs,
        "counts": counts,
        "claim_ids": claim_ids,
        "filters": {
            "profile": profile, "verdict": verdict, "recommended_use": recommended_use,
            "claim_id": claim_id, "has_against": has_against, "has_reconcile": has_reconcile,
            "q": q, "sort": sort, "order": order,
        },
    })


# ── Document Detail ───────────────────────────────────────────────────────

@app.get("/doc/{slug}", response_class=HTMLResponse)
async def document_detail(request: Request, slug: str):
    db = _db()
    doc = row_to_dict(db.execute("SELECT * FROM document WHERE slug = ?", (slug,)).fetchone())
    if not doc:
        db.close()
        return HTMLResponse("<h1>Not found</h1>", status_code=404)

    doc_id = doc["id"]
    steps = rows_to_dicts(db.execute(
        "SELECT * FROM analysis_step WHERE document_id = ? ORDER BY step_name", (doc_id,)
    ).fetchall())
    args_for = rows_to_dicts(db.execute(
        "SELECT * FROM argument WHERE document_id = ? AND direction = 'for'", (doc_id,)
    ).fetchall())
    args_against = rows_to_dicts(db.execute(
        "SELECT * FROM argument WHERE document_id = ? AND direction = 'against'", (doc_id,)
    ).fetchall())

    # Attach claim links to arguments
    for arg in args_for + args_against:
        arg["claim_ids"] = [r[0] for r in db.execute(
            "SELECT claim_id FROM argument_claim_link WHERE argument_id = ?", (arg["id"],)
        ).fetchall()]

    claim_assessments = rows_to_dicts(db.execute(
        "SELECT * FROM claim_assessment WHERE document_id = ? ORDER BY claim_id, stage", (doc_id,)
    ).fetchall())

    reconcile_rounds = rows_to_dicts(db.execute(
        "SELECT * FROM reconcile_round WHERE document_id = ? ORDER BY round_num", (doc_id,)
    ).fetchall())

    db.close()
    return _render(request, "document.html", {
        "doc": doc,
        "steps": steps,
        "args_for": args_for,
        "args_against": args_against,
        "claim_assessments": claim_assessments,
        "reconcile_rounds": reconcile_rounds,
    })


# ── Claim Explorer ────────────────────────────────────────────────────────

@app.get("/claims", response_class=HTMLResponse)
async def claims_list(request: Request):
    db = _db()
    claims = rows_to_dicts(db.execute("SELECT * FROM claim ORDER BY claim_id").fetchall())

    for c in claims:
        # Count documents linked through arguments
        c["support_count"] = db.execute(
            "SELECT COUNT(DISTINCT a.document_id) FROM argument a "
            "JOIN argument_claim_link acl ON acl.argument_id = a.id "
            "WHERE acl.claim_id = ? AND a.direction = 'for'", (c["claim_id"],)
        ).fetchone()[0]
        c["challenge_count"] = db.execute(
            "SELECT COUNT(DISTINCT a.document_id) FROM argument a "
            "JOIN argument_claim_link acl ON acl.argument_id = a.id "
            "WHERE acl.claim_id = ? AND a.direction = 'against'", (c["claim_id"],)
        ).fetchone()[0]
        # Count from claim assessments with engagement
        c["assessed_count"] = db.execute(
            "SELECT COUNT(DISTINCT document_id) FROM claim_assessment "
            "WHERE claim_id = ? AND engagement_type != 'none'", (c["claim_id"],)
        ).fetchone()[0]

    db.close()
    return _render(request, "claims.html", {"claims": claims})


@app.get("/claim/{claim_id}", response_class=HTMLResponse)
async def claim_detail(request: Request, claim_id: str):
    db = _db()
    claim = row_to_dict(db.execute("SELECT * FROM claim WHERE claim_id = ?", (claim_id,)).fetchone())
    if not claim:
        db.close()
        return HTMLResponse("<h1>Claim not found</h1>", status_code=404)

    # Documents with arguments for this claim
    support_docs = rows_to_dicts(db.execute(
        """SELECT DISTINCT d.slug, d.summary, d.final_verdict, d.profile,
            a.strength, a.argument
        FROM argument a
        JOIN argument_claim_link acl ON acl.argument_id = a.id
        JOIN document d ON d.id = a.document_id
        WHERE acl.claim_id = ? AND a.direction = 'for'
        ORDER BY CASE a.strength WHEN 'strong' THEN 0 WHEN 'moderate' THEN 1 WHEN 'weak' THEN 2 ELSE 3 END
        """, (claim_id,)
    ).fetchall())

    # Documents with arguments against
    challenge_docs = rows_to_dicts(db.execute(
        """SELECT DISTINCT d.slug, d.summary, d.final_verdict, d.profile,
            a.strength, a.argument
        FROM argument a
        JOIN argument_claim_link acl ON acl.argument_id = a.id
        JOIN document d ON d.id = a.document_id
        WHERE acl.claim_id = ? AND a.direction = 'against'
        ORDER BY CASE a.strength WHEN 'strong' THEN 0 WHEN 'moderate' THEN 1 WHEN 'weak' THEN 2 ELSE 3 END
        """, (claim_id,)
    ).fetchall())

    # Documents with claim assessments (contextual engagement)
    context_docs = rows_to_dicts(db.execute(
        """SELECT DISTINCT d.slug, d.summary, d.final_verdict, d.profile,
            ca.engagement_type, ca.support_strength, ca.challenge_strength, ca.why_limited
        FROM claim_assessment ca
        JOIN document d ON d.id = ca.document_id
        WHERE ca.claim_id = ? AND ca.stage = 'audit'
        AND ca.engagement_type = 'indirect'
        ORDER BY d.slug
        """, (claim_id,)
    ).fetchall())

    db.close()
    return _render(request, "claim_detail.html", {
        "claim": claim,
        "support_docs": support_docs,
        "challenge_docs": challenge_docs,
        "context_docs": context_docs,
    })


# ── Implications ──────────────────────────────────────────────────────────

@app.get("/implications", response_class=HTMLResponse)
async def implications_page(request: Request):
    db = _db()
    claims = rows_to_dicts(db.execute("SELECT * FROM claim ORDER BY claim_id").fetchall())

    for c in claims:
        cid = c["claim_id"]
        # Get all arguments linked to this claim, grouped by doc
        c["arguments"] = rows_to_dicts(db.execute(
            """SELECT a.direction, a.strength, a.argument,
                d.slug, d.final_verdict, d.recommended_use, d.profile
            FROM argument a
            JOIN argument_claim_link acl ON acl.argument_id = a.id
            JOIN document d ON d.id = a.document_id
            WHERE acl.claim_id = ?
            AND d.final_verdict IN ('relevant', 'marginal')
            ORDER BY a.direction,
                CASE a.strength WHEN 'strong' THEN 0 WHEN 'moderate' THEN 1 WHEN 'weak' THEN 2 ELSE 3 END
            """, (cid,)
        ).fetchall())

        # Summary counts
        c["for_count"] = sum(1 for a in c["arguments"] if a["direction"] == "for")
        c["against_count"] = sum(1 for a in c["arguments"] if a["direction"] == "against")

    db.close()
    return _render(request, "implications.html", {"claims": claims})


# ── Review Queue ──────────────────────────────────────────────────────────

@app.get("/review", response_class=HTMLResponse)
async def review_queue(request: Request, tab: str = "low_confidence"):
    db = _db()

    low_confidence = rows_to_dicts(db.execute(
        "SELECT * FROM document WHERE final_confidence < 0.8 ORDER BY final_confidence ASC"
    ).fetchall())

    reconciled = rows_to_dicts(db.execute(
        "SELECT * FROM document WHERE has_reconcile = 1 ORDER BY slug"
    ).fetchall())

    challenged = rows_to_dicts(db.execute(
        """SELECT DISTINCT d.* FROM document d
        JOIN argument a ON a.document_id = d.id
        WHERE a.direction = 'against'
        ORDER BY d.slug"""
    ).fetchall())

    db.close()
    return _render(request, "review.html", {
        "low_confidence": low_confidence,
        "reconciled": reconciled,
        "challenged": challenged,
        "tab": tab,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webapp.app:app", host="0.0.0.0", port=8000, reload=True)

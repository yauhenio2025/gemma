"""SQLite schema for the theory-analysis research browser."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "gemma_browser.db"

SCHEMA = """
-- Theory-level tables
CREATE TABLE IF NOT EXISTS theory (
    id          INTEGER PRIMARY KEY,
    profile     TEXT NOT NULL,          -- 'journalistic' or 'academic'
    thesis      TEXT NOT NULL,
    raw_path    TEXT NOT NULL            -- path to 01_theory_map.json
);

CREATE TABLE IF NOT EXISTS claim (
    id              INTEGER PRIMARY KEY,
    theory_id       INTEGER NOT NULL REFERENCES theory(id),
    claim_id        TEXT NOT NULL,       -- C1, C2, ...
    claim_text      TEXT NOT NULL,
    why_it_matters  TEXT,
    support_requirements    TEXT,         -- JSON array
    challenge_requirements  TEXT,         -- JSON array
    indirect_relevance_hooks TEXT,        -- JSON array
    false_positive_matches   TEXT         -- JSON array
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_claim ON claim(theory_id, claim_id);

CREATE TABLE IF NOT EXISTS theme (
    id              INTEGER PRIMARY KEY,
    theory_id       INTEGER NOT NULL REFERENCES theory(id),
    theme_id        TEXT NOT NULL,        -- T1, T2, ...
    theme_text      TEXT NOT NULL,
    why_articles_may_matter TEXT,
    typical_signals TEXT                  -- JSON array
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_theme ON theme(theory_id, theme_id);

CREATE TABLE IF NOT EXISTS conceptual_boundary (
    id          INTEGER PRIMARY KEY,
    theory_id   INTEGER NOT NULL REFERENCES theory(id),
    boundary    TEXT NOT NULL,
    explanation TEXT
);

CREATE TABLE IF NOT EXISTS relevance_tier (
    id              INTEGER PRIMARY KEY,
    theory_id       INTEGER NOT NULL REFERENCES theory(id),
    label           TEXT NOT NULL,       -- irrelevant/contextual/marginal/relevant
    definition      TEXT,
    recommended_use TEXT
);

CREATE TABLE IF NOT EXISTS relevance_hook (
    id          INTEGER PRIMARY KEY,
    theory_id   INTEGER NOT NULL REFERENCES theory(id),
    hook        TEXT NOT NULL,
    counts_as   TEXT,
    notes       TEXT
);

-- Document-level tables
CREATE TABLE IF NOT EXISTS document (
    id              INTEGER PRIMARY KEY,
    profile         TEXT NOT NULL,       -- 'journalistic' or 'academic'
    slug            TEXT NOT NULL UNIQUE,
    article_path    TEXT,
    report_path     TEXT,
    output_dir      TEXT,                -- path to the output directory

    -- from index.json
    model           TEXT,
    index_verdict   TEXT,
    index_confidence REAL,
    index_recommended_use TEXT,

    -- from 01_article_map.json
    article_kind    TEXT,
    summary         TEXT,
    main_claims     TEXT,                -- JSON array
    evidence_or_facts TEXT,              -- JSON array
    possible_theory_hooks TEXT,          -- JSON array

    -- academic-only fields from article_map
    research_question TEXT,
    method_or_approach TEXT,             -- JSON array
    empirical_scope_or_case TEXT,        -- JSON array
    theoretical_frameworks TEXT,         -- JSON array

    -- from 02_relevance_audit.json
    initial_verdict TEXT,
    initial_reason  TEXT,
    initial_confidence REAL,
    contextual_relevance_points TEXT,    -- JSON array
    illustrative_relevance_points TEXT,  -- JSON array
    article_level_for_theory TEXT,       -- JSON array
    article_level_against_theory TEXT,   -- JSON array
    irrelevance_reasons TEXT,            -- JSON array

    -- from 03_counter_audit.json
    grade_inflation_detected INTEGER,    -- 0/1
    false_negative_detected  INTEGER,    -- 0/1
    counter_problems TEXT,               -- JSON array
    missing_support_points TEXT,         -- JSON array
    missing_challenge_points TEXT,       -- JSON array
    corrected_verdict TEXT,
    counter_confidence REAL,

    -- from 04_final_judgment.json
    final_verdict       TEXT,
    final_confidence     REAL,
    relevance_mode       TEXT,
    one_paragraph_verdict TEXT,
    contextual_relevance TEXT,           -- JSON array
    cannot_adjudicate    TEXT,           -- JSON array
    state_capital_nexus  TEXT,
    recommended_use      TEXT,

    -- from 05_theory_implications.json
    has_theory_implications INTEGER DEFAULT 0,
    implication_overall    TEXT,
    implication_summary    TEXT,
    implication_new_subclaims TEXT,      -- JSON array
    implication_new_open_questions TEXT, -- JSON array
    implication_revision_priority TEXT,
    implication_follow_up  TEXT,         -- JSON array
    implication_confidence REAL,

    -- reconciliation metadata
    has_reconcile       INTEGER DEFAULT 0,
    reconcile_rounds    INTEGER DEFAULT 0,
    reconcile_confidence REAL
);

-- Per-claim assessments from relevance audit (02)
CREATE TABLE IF NOT EXISTS claim_assessment (
    id              INTEGER PRIMARY KEY,
    document_id     INTEGER NOT NULL REFERENCES document(id),
    claim_id        TEXT NOT NULL,        -- C1, C2, ...
    stage           TEXT NOT NULL DEFAULT 'audit',  -- 'audit' or 'counter'
    engagement_type TEXT,
    support_strength TEXT,
    challenge_strength TEXT,
    support_points  TEXT,                 -- JSON array
    challenge_points TEXT,                -- JSON array
    why_limited     TEXT,
    -- counter-audit corrections
    corrected_support_strength TEXT,
    corrected_challenge_strength TEXT,
    note            TEXT
);
CREATE INDEX IF NOT EXISTS idx_ca_doc ON claim_assessment(document_id);
CREATE INDEX IF NOT EXISTS idx_ca_claim ON claim_assessment(claim_id);

-- Arguments from final judgment (04) and reconcile rounds
CREATE TABLE IF NOT EXISTS argument (
    id          INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES document(id),
    direction   TEXT NOT NULL,           -- 'for' or 'against'
    strength    TEXT,
    argument    TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'final',  -- 'final' or 'reconcile_N'
    FOREIGN KEY (document_id) REFERENCES document(id)
);
CREATE INDEX IF NOT EXISTS idx_arg_doc ON argument(document_id);
CREATE INDEX IF NOT EXISTS idx_arg_dir ON argument(direction);

-- Theory implication claim-level rows from 05
CREATE TABLE IF NOT EXISTS theory_implication_claim (
    id          INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES document(id),
    claim_id    TEXT NOT NULL,
    effect      TEXT NOT NULL,
    why         TEXT,
    evidence_from_document TEXT,         -- JSON array
    proposed_revision TEXT
);
CREATE INDEX IF NOT EXISTS idx_tic_doc ON theory_implication_claim(document_id);
CREATE INDEX IF NOT EXISTS idx_tic_claim ON theory_implication_claim(claim_id);

-- Claim links from arguments
CREATE TABLE IF NOT EXISTS argument_claim_link (
    id          INTEGER PRIMARY KEY,
    argument_id INTEGER NOT NULL REFERENCES argument(id),
    claim_id    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_acl_arg ON argument_claim_link(argument_id);
CREATE INDEX IF NOT EXISTS idx_acl_claim ON argument_claim_link(claim_id);

-- Reconcile rounds stored as JSON blobs for display
CREATE TABLE IF NOT EXISTS reconcile_round (
    id          INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES document(id),
    round_num   INTEGER NOT NULL,
    raw_json    TEXT NOT NULL,            -- full JSON for rendering
    verdict     TEXT,
    confidence  REAL,
    raw_path    TEXT
);
CREATE INDEX IF NOT EXISTS idx_rr_doc ON reconcile_round(document_id);

-- Raw step blobs for tab rendering
CREATE TABLE IF NOT EXISTS analysis_step (
    id          INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES document(id),
    step_name   TEXT NOT NULL,            -- '01_article_map', '02_relevance_audit', etc.
    raw_json    TEXT NOT NULL,
    raw_path    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_as_doc ON analysis_step(document_id);
"""


def init_db(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn

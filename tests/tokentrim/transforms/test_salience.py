from __future__ import annotations

from tokentrim.salience import extract_query_terms, score_text_salience


def test_salience_prefers_old_unresolved_error_over_recent_fluff() -> None:
    query = extract_query_terms("fix ./tokentrim/README.md missing fixture")

    old_error = score_text_salience(
        "FileNotFoundError: missing fixture in ./tokentrim/README.md",
        query_terms=query,
        recency_rank=4,
    )
    recent_fluff = score_text_salience(
        "I am still thinking about the task and will continue soon.",
        query_terms=query,
        recency_rank=0,
    )

    assert old_error > recent_fluff


def test_salience_prefers_active_path_over_generic_discussion() -> None:
    query = extract_query_terms("update ./tokentrim/tokentrim/transforms/compaction/transform.py")

    path_text = score_text_salience(
        "Edit ./tokentrim/tokentrim/transforms/compaction/transform.py and rerun pytest.",
        query_terms=query,
        recency_rank=2,
    )
    generic_text = score_text_salience(
        "Continue debugging the issue carefully.",
        query_terms=query,
        recency_rank=0,
    )

    assert path_text > generic_text

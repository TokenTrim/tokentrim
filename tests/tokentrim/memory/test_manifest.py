from __future__ import annotations

from tokentrim.memory import FilesystemMemoryStore, MemoryWrite, format_memory_manifest, scan_memory_headers


def test_scan_memory_headers_exposes_richer_header_fields(tmp_path) -> None:
    store = FilesystemMemoryStore(root_dir=tmp_path / "memory")
    created = store.write_session_memory(
        session_id="sess_1",
        write=MemoryWrite(
            content="Use repo root /workspace/app for debugging commands.",
            kind="active_state",
            dedupe_key="repo_root",
            metadata={"title": "Repo root", "description": "Repository root path"},
            source_refs=("tool:pwd",),
        ),
    )

    headers = scan_memory_headers(
        memory_store=store,
        session_id="sess_1",
        user_id=None,
        org_id=None,
    )

    assert len(headers) == 1
    header = headers[0]
    assert header.memory_id == created.memory_id
    assert header.dedupe_key == "repo_root"
    assert header.canonical_key == "repo_root"
    assert header.source_ref_count == 1
    assert header.content_preview.startswith("Use repo root")
    assert header.freshness_bucket == "fresh"


def test_format_memory_manifest_groups_scope_and_status(tmp_path) -> None:
    store = FilesystemMemoryStore(root_dir=tmp_path / "memory")
    store.write_session_memory(
        session_id="sess_1",
        write=MemoryWrite(
            content="Repo root is /workspace/app",
            kind="task_fact",
            metadata={"title": "Repo root", "description": "Repository root path"},
        ),
    )

    manifest = format_memory_manifest(
        scan_memory_headers(
            memory_store=store,
            session_id="sess_1",
            user_id=None,
            org_id=None,
        )
    )

    assert "SESSION:" in manifest
    assert "status=active" in manifest
    assert "freshness=fresh" in manifest

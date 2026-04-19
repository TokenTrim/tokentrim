from __future__ import annotations

"""Command-line entrypoints for local-first Tokentrim workflows."""

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from tokentrim.memory import FilesystemMemoryStore
from tokentrim.consolidator import (
    AgenticConsolidatorAgent,
    ConsolidatorAgent,
    ConsolidationJobConfig,
    DeterministicConsolidatorAgent,
    ModelConsolidatorAgent,
    SessionConsolidationJob,
)
from tokentrim.tracing import FilesystemTraceStore

ConsolidationTarget = tuple[str, str, str | None]


def main(argv: list[str] | None = None) -> int:
    """Run the Tokentrim CLI and return a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "consolidate":
        return _run_consolidate(args)
    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    """Build the root CLI parser."""
    parser = argparse.ArgumentParser(prog="tokentrim")
    subparsers = parser.add_subparsers(dest="command")

    consolidate = subparsers.add_parser("consolidate", help="Run offline durable-memory consolidation.")
    consolidate.add_argument(
        "--memory-dir",
        default=".tokentrim/memory",
        help="Root directory for the filesystem memory store.",
    )
    consolidate.add_argument(
        "--trace-dir",
        default=".tokentrim/traces",
        help="Root directory for the filesystem trace store.",
    )
    consolidate.add_argument(
        "--mode",
        choices=("deterministic", "model", "agentic"),
        default="deterministic",
        help="Consolidator mode.",
    )
    consolidate.add_argument(
        "--scope",
        choices=("all", "user", "org"),
        default="all",
        help="Restrict durable writes to all, user, or org scope.",
    )
    consolidate.add_argument("--user-id", help="Target one user.")
    consolidate.add_argument("--session-id", help="Target one session. With no user-id, scans all users.")
    consolidate.add_argument("--org-id", help="Optional org id to use for targeted runs.")
    consolidate.add_argument("--model", help="Model to use for model/agentic modes.")
    apply_group = consolidate.add_mutually_exclusive_group()
    apply_group.add_argument(
        "--apply",
        dest="apply",
        action="store_true",
        help="Apply the consolidation plan to durable memory.",
    )
    apply_group.add_argument(
        "--dry-run",
        dest="apply",
        action="store_false",
        help="Print the plan without writing durable memory.",
    )
    consolidate.set_defaults(apply=False)
    return parser


def _run_consolidate(args: argparse.Namespace) -> int:
    """Run one or more offline consolidation jobs from the CLI."""
    if args.mode in {"model", "agentic"} and not args.model:
        raise SystemExit("--model is required for model and agentic consolidation modes.")

    memory_store = FilesystemMemoryStore(root_dir=args.memory_dir)
    trace_store = FilesystemTraceStore(root_dir=args.trace_dir)
    agent = _build_agent(mode=args.mode, model=args.model)
    job = SessionConsolidationJob(
        memory_store=memory_store,
        trace_store=trace_store,
        agent=agent,
        config=ConsolidationJobConfig(
            apply=bool(args.apply),
            write_scope=args.scope,
        ),
    )

    targets = list(
        _resolve_targets(
            trace_dir=Path(args.trace_dir),
            user_id=args.user_id,
            session_id=args.session_id,
            org_id=args.org_id,
        )
    )
    if not targets:
        print("No matching completed sessions found.", file=sys.stderr)
        return 1

    for target_user_id, target_session_id, target_org_id in targets:
        result = job.run(
            session_id=target_session_id,
            user_id=target_user_id,
            org_id=target_org_id,
        )
        summary = _summarize_result(
            user_id=target_user_id,
            session_id=target_session_id,
            org_id=target_org_id,
            mode=args.mode,
            scope=args.scope,
            applied=bool(args.apply),
            result=result,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _build_agent(*, mode: str, model: str | None) -> ConsolidatorAgent:
    """Create the consolidator implementation for the requested mode."""
    if mode == "deterministic":
        return DeterministicConsolidatorAgent()
    if mode == "model":
        return ModelConsolidatorAgent(model=model or "")
    if mode == "agentic":
        return AgenticConsolidatorAgent(model=model or "")
    raise ValueError(f"Unsupported mode: {mode}")


def _resolve_targets(
    *,
    trace_dir: Path,
    user_id: str | None,
    session_id: str | None,
    org_id: str | None,
) -> Iterable[ConsolidationTarget]:
    """Yield consolidation targets discovered from the completed trace index.

    The CLI intentionally derives its scan surface from persisted completed
    traces, not from memory directories. That keeps offline consolidation tied
    to sessions that actually finished and produced replayable evidence.
    """
    completed_dir = trace_dir / "completed"
    if user_id is not None and session_id is not None:
        yield (user_id, session_id, org_id)
        return
    if not completed_dir.exists():
        return
    for user_dir in sorted(path for path in completed_dir.iterdir() if path.is_dir()):
        resolved_user_id = user_dir.name
        if user_id is not None and resolved_user_id != user_id:
            continue
        for session_dir in sorted(path for path in user_dir.iterdir() if path.is_dir()):
            resolved_session_id = session_dir.name
            if session_id is not None and resolved_session_id != session_id:
                continue
            yield (resolved_user_id, resolved_session_id, org_id)


def _summarize_result(
    *,
    user_id: str,
    session_id: str,
    org_id: str | None,
    mode: str,
    scope: str,
    applied: bool,
    result,
) -> dict[str, object]:
    """Build a stable JSON summary for one consolidation target."""
    return {
        "user_id": user_id,
        "session_id": session_id,
        "org_id": org_id,
        "mode": mode,
        "scope": scope,
        "applied": applied,
        "plan": {
            "user_upserts": len(result.plan.user_upserts),
            "org_upserts": len(result.plan.org_upserts),
            "user_archives": len(result.plan.user_archives),
            "org_archives": len(result.plan.org_archives),
            "merge_operations": len(result.plan.merge_operations),
            "rationale": list(result.plan.rationale),
            "source_refs": list(result.plan.source_refs),
        },
        "apply_result": (
            None
            if result.apply_result is None
            else {
                "upserted": [record.memory_id for record in result.apply_result.upserted],
                "archived_memory_ids": list(result.apply_result.archived_memory_ids),
                "merged_source_ids": list(result.apply_result.merged_source_ids),
            }
        ),
    }

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Mapping

MemoryScope = Literal["session", "user", "org"]
MemoryStatus = Literal["active", "stale", "archived"]

_VALID_SCOPES = frozenset({"session", "user", "org"})
_VALID_STATUSES = frozenset({"active", "stale", "archived"})


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_non_empty(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _validate_salience(value: float) -> float:
    if not isinstance(value, int | float):
        raise ValueError("salience must be a number.")
    normalized = float(value)
    if normalized < 0.0 or normalized > 1.0:
        raise ValueError("salience must be between 0.0 and 1.0.")
    return normalized


def _validate_scope(value: str) -> MemoryScope:
    normalized = _validate_non_empty(value, field_name="scope")
    if normalized not in _VALID_SCOPES:
        raise ValueError(f"scope must be one of {_VALID_SCOPES!r}.")
    return normalized  # type: ignore[return-value]


def _validate_status(value: str) -> MemoryStatus:
    normalized = _validate_non_empty(value, field_name="status")
    if normalized not in _VALID_STATUSES:
        raise ValueError(f"status must be one of {_VALID_STATUSES!r}.")
    return normalized  # type: ignore[return-value]


def _normalize_source_refs(source_refs: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for ref in source_refs:
        normalized.append(_validate_non_empty(ref, field_name="source_refs entry"))
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: str
    scope: MemoryScope
    subject_id: str
    kind: str
    content: str
    salience: float = 0.5
    status: MemoryStatus = "active"
    source_refs: tuple[str, ...] = ()
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    dedupe_key: str | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_id", _validate_non_empty(self.memory_id, field_name="memory_id"))
        object.__setattr__(self, "scope", _validate_scope(self.scope))
        object.__setattr__(self, "subject_id", _validate_non_empty(self.subject_id, field_name="subject_id"))
        object.__setattr__(self, "kind", _validate_non_empty(self.kind, field_name="kind"))
        object.__setattr__(self, "content", _validate_non_empty(self.content, field_name="content"))
        object.__setattr__(self, "salience", _validate_salience(self.salience))
        object.__setattr__(self, "status", _validate_status(self.status))
        object.__setattr__(self, "source_refs", _normalize_source_refs(self.source_refs))
        object.__setattr__(self, "created_at", _validate_non_empty(self.created_at, field_name="created_at"))
        object.__setattr__(self, "updated_at", _validate_non_empty(self.updated_at, field_name="updated_at"))
        if self.dedupe_key is not None:
            object.__setattr__(
                self,
                "dedupe_key",
                _validate_non_empty(self.dedupe_key, field_name="dedupe_key"),
            )
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping when provided.")


@dataclass(frozen=True, slots=True)
class MemoryWrite:
    content: str
    kind: str
    salience: float = 0.5
    dedupe_key: str | None = None
    metadata: Mapping[str, object] | None = None
    source_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", _validate_non_empty(self.content, field_name="content"))
        object.__setattr__(self, "kind", _validate_non_empty(self.kind, field_name="kind"))
        object.__setattr__(self, "salience", _validate_salience(self.salience))
        object.__setattr__(self, "source_refs", _normalize_source_refs(self.source_refs))
        if self.dedupe_key is not None:
            object.__setattr__(
                self,
                "dedupe_key",
                _validate_non_empty(self.dedupe_key, field_name="dedupe_key"),
            )
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping when provided.")


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    session_id: str | None = None
    user_id: str | None = None
    org_id: str | None = None
    k: int = 8
    scope_weights: Mapping[MemoryScope, float] | None = None
    kind_filter: tuple[str, ...] = ()
    text_query: str | None = None

    def __post_init__(self) -> None:
        if self.session_id is not None:
            object.__setattr__(
                self,
                "session_id",
                _validate_non_empty(self.session_id, field_name="session_id"),
            )
        if self.user_id is not None:
            object.__setattr__(
                self,
                "user_id",
                _validate_non_empty(self.user_id, field_name="user_id"),
            )
        if self.org_id is not None:
            object.__setattr__(self, "org_id", _validate_non_empty(self.org_id, field_name="org_id"))
        if self.k < 1:
            raise ValueError("k must be at least 1.")
        normalized_filter = tuple(
            _validate_non_empty(kind, field_name="kind_filter entry") for kind in self.kind_filter
        )
        object.__setattr__(self, "kind_filter", normalized_filter)
        if self.text_query is not None:
            object.__setattr__(
                self,
                "text_query",
                _validate_non_empty(self.text_query, field_name="text_query"),
            )
        if self.scope_weights is not None:
            normalized_weights: dict[MemoryScope, float] = {}
            for scope, weight in self.scope_weights.items():
                normalized_scope = _validate_scope(scope)
                if not isinstance(weight, int | float):
                    raise ValueError("scope weight values must be numeric.")
                normalized_weights[normalized_scope] = float(weight)
            object.__setattr__(self, "scope_weights", normalized_weights)

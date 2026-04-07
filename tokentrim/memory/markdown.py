from __future__ import annotations

import json
from pathlib import Path

from tokentrim.memory.types import MemoryScope


def write_memory_markdown(
    path: Path,
    *,
    entry_id: str,
    scope: MemoryScope,
    created_at: str,
    keywords: tuple[str, ...],
    metadata: dict[str, object],
    content: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        f"id: {entry_id}",
        f"scope: {scope.value}",
        f"created_at: {created_at}",
        "keywords:",
    ]
    lines.extend(f"  - {keyword}" for keyword in keywords)
    lines.append(f"metadata_json: {json.dumps(metadata, sort_keys=True)}")
    lines.append("---")
    lines.append("")
    lines.append(content.rstrip())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_memory_markdown(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        raise ValueError(f"Invalid memory frontmatter in {path}")

    try:
        closing_index = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError(f"Missing closing frontmatter marker in {path}") from exc

    frontmatter = _parse_frontmatter(lines[1:closing_index], path)
    content = "\n".join(lines[closing_index + 1 :]).strip()
    frontmatter["content"] = content
    return frontmatter


def _parse_frontmatter(lines: list[str], path: Path) -> dict[str, object]:
    parsed: dict[str, object] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if ": " in line:
            key, value = line.split(": ", 1)
            if key == "metadata_json":
                parsed[key] = json.loads(value)
            else:
                parsed[key] = value
            index += 1
            continue
        if line.endswith(":"):
            key = line[:-1]
            index += 1
            values: list[str] = []
            while index < len(lines) and lines[index].startswith("  - "):
                values.append(lines[index][4:])
                index += 1
            parsed[key] = tuple(values)
            continue
        raise ValueError(f"Invalid frontmatter line in {path}: {line}")
    return parsed

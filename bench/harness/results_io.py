"""Reading a result directory's JSON files: one reader, one missing-file rule.

`run.json`, `noise.json`, `conformance.json` and the like are optional to
their readers: a label measured before a field existed, or a file a crashed
run never wrote, reads as `{}`. Row files are not optional to most callers,
so `rows_of` says None and the caller decides whether that is an error.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_json(path: Path) -> dict:
    """The JSON object at `path`, or `{}` when it is missing, unreadable, not
    JSON, or not an object."""
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def rows_of(path: Path) -> list[dict] | None:
    """A `rows.json` list, or None when the file is missing."""
    return json.loads(path.read_text()) if path.exists() else None

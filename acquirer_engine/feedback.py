"""Banker feedback: acquirers flagged irrelevant for a target are excluded on the next run."""

from __future__ import annotations

import difflib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .errors import ConfigError

log = logging.getLogger(__name__)


class FeedbackError(ConfigError):
    """An --exclude name cannot be matched to an acquirer in the data."""


def _read(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("root is not an object")
        return data
    except (json.JSONDecodeError, ValueError) as e:
        backup = path.with_suffix(".corrupt.json")
        path.replace(backup)
        log.warning("Feedback file unreadable (%s); moved to %s and starting fresh", e, backup.name)
        return {}


def resolve(name: str, known: list[str]) -> str:
    """Match a user-typed name to a known acquirer (exact, case-insensitive, unique substring, fuzzy).
    Raises FeedbackError listing candidates.
    """
    lookup = {k.casefold(): k for k in known}
    if name.casefold() in lookup:
        return lookup[name.casefold()]
    partial = [k for k in known if name.casefold() in k.casefold()]
    if len(partial) == 1:
        return partial[0]
    options = partial or difflib.get_close_matches(name, known, n=5, cutoff=0.5)
    raise FeedbackError(f"Cannot match acquirer '{name}'. Candidates: {options or 'none'}")


def excluded(path: Path, target_slug: str) -> list[dict]:
    """Exclusions recorded for one target slug (empty list if none or the file is missing)."""
    return list(_read(path).get(target_slug, []))


def add(path: Path, target_slug: str, names: list[str], known: list[str], reason: str) -> list[str]:
    """Persist exclusions for a target; returns the canonical names newly added.
    Raises FeedbackError for unknown names.
    """
    data = _read(path)
    entries = data.get(target_slug, [])
    have = {e["acquirer"] for e in entries}
    added = []
    for n in names:
        canon = resolve(n, known)
        if canon not in have:
            entries = [
                *entries,
                {
                    "acquirer": canon,
                    "reason": reason,
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
            ]
            have.add(canon)
            added.append(canon)
    path.write_text(json.dumps({**data, target_slug: entries}, indent=2), encoding="utf-8")
    return added


def clear(path: Path, target_slug: str) -> None:
    """Forget every exclusion for one target slug (no-op if there are none)."""
    data = _read(path)
    if target_slug in data:
        path.write_text(json.dumps({k: v for k, v in data.items() if k != target_slug}, indent=2), encoding="utf-8")

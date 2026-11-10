"""Scan a directory tree for API-key-shaped strings before anything is shipped.

Usage: python -m acquirer_engine.secretscan <dir>   (exit 1 and list hits if anything matches)

Patterns require realistic key lengths, so documentation placeholders such as
``ANTHROPIC_API_KEY=sk-ant-...`` do not trip the scan, while a pasted real key does.
Standard library only, so it can run on a staged copy before dependencies are installed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERNS: dict[str, re.Pattern[str]] = {
    "anthropic key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "openai-style key": re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_\-]{20,}"),
    "google api key": re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    "google AQ token": re.compile(r"(?<![A-Za-z0-9])AQ\.[A-Za-z0-9_\-]{20,}"),
    "assigned *_API_KEY": re.compile(r"[A-Z][A-Z0-9_]*_API_KEY[ \t]*=[ \t]*['\"]?[A-Za-z0-9_\-.]{12,}"),
}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".zip", ".pyc", ".docx", ".pdf"}


def scan_text(text: str) -> list[str]:
    """Return the names of every pattern found in ``text`` (empty list if clean)."""
    return [name for name, rx in PATTERNS.items() if rx.search(text)]


def scan_tree(root: Path) -> list[tuple[str, str]]:
    """Scan every file under ``root``; return ``(relative path, pattern name)`` for each hit.

    Matched text is never returned or printed, so the scan cannot leak the secret it found.
    Unreadable or binary files are skipped by suffix; other files are decoded leniently.
    """
    hits: list[tuple[str, str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        text = path.read_bytes().decode("utf-8", errors="ignore")
        hits.extend((str(path.relative_to(root)), name) for name in scan_text(text))
    return hits


def main(argv: list[str] | None = None) -> int:
    """CLI entry: exit 0 when clean, 1 when any file matches, 2 on bad usage."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or not Path(args[0]).is_dir():
        print("usage: python -m acquirer_engine.secretscan <dir>", file=sys.stderr)
        return 2
    hits = scan_tree(Path(args[0]))
    for rel, name in hits:
        print(f"SECRET? {rel}: {name}", file=sys.stderr)
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())

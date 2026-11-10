"""Command line entry: `python -m acquirer_engine [options]` (or the `acquirer-engine` console script).

Exit codes: 0 ok · 1 unexpected error or every page failed · 2 bad usage · 3 data · 4 config · 5 LLM · 6 render.
Errors print one line; pass --debug for the traceback.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from pydantic import ValidationError

from . import config as C
from . import data, eda, engine, feedback, render, settings
from . import eval as evaluation
from .errors import ConfigError, EngineError
from .llm import Cache, LLMClient, make_client
from .profile import TargetProfile, load_yaml, validate_against
from .schema import SchemaError, load_schema_map

log = logging.getLogger("acquirer_engine")
KEY_ENVS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
PROVIDERS = ["anthropic", "openai", "gemini", "local", "replay", "offline"]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="acquirer-engine", description="Rank likely acquirers and write one-page rationales."
    )
    t = p.add_argument_group("target (defaults: Healthcare Services, $200M, regional, private, strong margin)")
    t.add_argument("--target", type=Path, help="YAML target profile (overrides defaults; flags override YAML)")
    t.add_argument("--name")
    t.add_argument("--sector")
    t.add_argument("--ev", type=float, help="target EV, $M")
    t.add_argument("--geography")
    t.add_argument("--ownership")
    t.add_argument("--margin-pct", type=float, help="target EBITDA margin %% (default: dataset 75th percentile)")
    d = p.add_argument_group("data")
    d.add_argument("--data", type=Path, default=C.DEFAULT_DATA)
    d.add_argument("--schema-map", type=Path, help="YAML of canonical_name: csv_header to override header matching")
    d.add_argument(
        "--infer-acquirer-type",
        action="store_true",
        help="if the file has no acquirer_type, infer it from names (labelled inferred)",
    )
    p.add_argument("--compare", nargs=2, type=Path, metavar=("A.yaml", "B.yaml"), help="side-by-side mode")
    p.add_argument("--eda-only", action="store_true", help="run stage 0 (data profile) only; no LLM, no key")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument(
        "--min-per-type",
        type=int,
        default=None,
        help=f"buyer-mix floor: at least N strategics and N sponsors (default {C.MIN_PER_TYPE}; 0 disables)",
    )
    p.add_argument(
        "--provider",
        default=None,
        choices=PROVIDERS,
        help="default: first provider whose API key is set (anthropic, openai, gemini). "
        "replay = serve the committed sample from .cache/llm without a key",
    )
    p.add_argument("--model", help="override model id")
    p.add_argument(
        "--exclude", action="append", default=[], metavar="NAME", help="flag acquirer irrelevant (persisted)"
    )
    p.add_argument("--reason", default="flagged irrelevant by user", help="reason stored with --exclude")
    p.add_argument("--clear-feedback", action="store_true", help="forget exclusions for this target")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--refresh", action="store_true", help="ignore cache reads")
    p.add_argument("--workers", type=int, default=None, help="parallel LLM calls (default: provider-specific)")
    p.add_argument("--out-dir", type=Path, default=C.DEFAULT_OUT)
    p.add_argument("--feedback-file", type=Path, default=C.FEEDBACK_FILE)
    o = p.add_argument_group("configuration and logging")
    o.add_argument("--config", type=Path, help="YAML of tunable settings (precedence: defaults < file < env < --set)")
    o.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="override one setting, e.g. MIN_COMPS=8"
    )
    o.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    o.add_argument("--log-json", action="store_true", help="one JSON object per log line")
    o.add_argument("-v", "--verbose", action="store_true", help="same as --log-level INFO")
    o.add_argument("--debug", action="store_true", help="show tracebacks")
    return p


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        out.update(getattr(record, "fields", {}))
        return json.dumps(out, default=str)


def setup_logging(level: str, as_json: bool) -> None:
    """Configure the root logger once. This package never logs prompts or key values."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter() if as_json else logging.Formatter("%(levelname)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    quiet = max(logging.WARNING, logging.getLevelName(level))
    for noisy in ("httpx", "httpcore", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(quiet)


def load_env_file(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines). Real environment variables always win. Values are never logged."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def pick_provider(requested: str | None) -> str:
    """Explicit --provider, else the first provider whose key is set, else anthropic (fails with a clear message)."""
    if requested:
        return requested
    for name, env in KEY_ENVS.items():
        if os.environ.get(env):
            return name
    return C.DEFAULT_PROVIDER


def _profile(args: argparse.Namespace, yaml_path: Path | None, ds: data.Dataset) -> TargetProfile:
    base = load_yaml(yaml_path).model_dump() if yaml_path else {}
    flags = {
        "name": args.name,
        "sector": args.sector,
        "ev_mm": args.ev,
        "geography": args.geography,
        "ownership": args.ownership,
        "ebitda_margin_pct": args.margin_pct,
    }
    try:
        prof = TargetProfile(**{**base, **{k: v for k, v in flags.items() if v is not None}})
    except ValidationError as e:
        err = e.errors()[0]
        raise ConfigError(f"invalid target profile: {err['loc'][0] if err['loc'] else ''}: {err['msg']}") from e
    return validate_against(prof, ds)


def _print_schema(ds: data.Dataset) -> None:
    moved = {k: m for k, m in ds.schema.items() if m.method != "exact"}
    print(
        f"Schema: {len(ds.schema) - len(moved)} column(s) matched by name"
        + (f"; {len(moved)} mapped:" if moved else "")
    )
    for k, m in moved.items():
        print(f"  {m.source!r:30} -> {k:22} ({m.method}, confidence {m.confidence})")
    for f, why in ds.features.items():
        if why != "on":
            print(f"  feature {f}: {why}")


def _run_one(
    args: argparse.Namespace, ds: data.Dataset, client: LLMClient, cache: Cache, prof: TargetProfile
) -> tuple[dict, dict]:
    known = sorted(ds.df["acquirer"].unique())
    if args.clear_feedback:
        feedback.clear(args.feedback_file, prof.slug)
    if args.exclude:
        added = feedback.add(args.feedback_file, prof.slug, args.exclude, known, args.reason)
        if added:
            print(f"Excluded for {prof.slug}: {', '.join(added)}")
    excl = frozenset(e["acquirer"].casefold() for e in feedback.excluded(args.feedback_file, prof.slug))
    lo, hi = ds.df["deal_size_mm"].quantile([0.01, 0.99])
    if not lo <= prof.ev_mm <= hi:
        log.warning(
            "Target EV $%gM is outside the bulk of the data ($%.0f-%.0fM); comps will be thin.", prof.ev_mm, lo, hi
        )
    min_per_type = C.MIN_PER_TYPE if args.min_per_type is None else args.min_per_type
    report = engine.run(ds, prof, client, cache, args.top_n, excl, args.workers, min_per_type)
    report["meta"]["settings"] = settings.effective()
    report["evaluation"] = evaluation.score(report)
    paths = render.write_report(report, args.out_dir / prof.slug)
    return report, paths


def _summary(r: dict) -> int:
    m = r["meta"]
    status: dict[str, int] = {}
    for p in r["pages"]:
        status[p["status"]] = status.get(p["status"], 0) + 1
    sr = m.get("source_run")
    elapsed = (
        (f"replayed live run of {sr.get('generated_at', '?')} ({sr.get('elapsed_s', '?')}s)" if sr else "replayed")
        if m["elapsed_s"] is None
        else f"{m['elapsed_s']}s"
    )
    cost = f" · est. cost ${m['cost_usd']:.2f}" if m.get("cost_usd") else ""
    print(
        f"{r['target']['name']}: {len(r['pages'])} acquirers · {status} · {m['provider']}/{m['model']} · "
        f"{elapsed} · cache hits {m['cache_hits']}{cost}"
    )
    labels = sum((p["evidence"].get("pricing_vs_peers") or {}).get("position") == "below" for p in r["pages"])
    print(
        f"  pricing: {labels} buyer(s) priced below like-for-like peers; "
        f"{evaluation.below_peers_count(r)} page(s) say 'below peers'"
    )
    for w in r["warnings"]:
        print(f"  warning: {w}")
    misses = [p["candidate"]["acquirer"] for p in r["pages"] if p["status"] == "failed" and m["provider"] == "replay"]
    if misses:
        print(
            f"error: replay cache miss for {len(misses)} acquirer(s): {', '.join(misses)}. The cache was built for a "
            "different prompt version, target or data; run once with a live provider.",
            file=sys.stderr,
        )
        return 1
    if not r["pages"]:
        print(
            "error: 0 eligible acquirers (each needs a relevant deal and at least one closed deal); "
            "an empty buyer list is not a result",
            file=sys.stderr,
        )
        return 1
    if all(p["status"] == "failed" for p in r["pages"]):
        return 1
    return 0


def _main(args: argparse.Namespace) -> int:
    settings.apply(args.config, args.set)
    if args.top_n < 1:
        raise ConfigError("--top-n must be >= 1")
    if args.provider == "replay" and (args.refresh or args.no_cache):
        raise ConfigError("--provider replay reads only from the cache; drop --refresh/--no-cache")
    try:
        schema_map = load_schema_map(args.schema_map) if args.schema_map else None
    except (SchemaError, OSError) as e:
        raise ConfigError(str(e)) from e
    ds = data.load(args.data, schema_map, args.infer_acquirer_type)
    _print_schema(ds)
    if args.eda_only:
        prof = _profile(args, args.target, ds)
        paths = render.write_eda(
            eda.run(ds, prof), prof.model_dump(), args.data.name, ds.sha256, args.out_dir / prof.slug
        )
        print(f"EDA: {paths['md']}\n     {paths['json']}")
        return 0
    client = make_client(args.provider, args.model)
    if args.workers is None:
        args.workers = getattr(client, "default_workers", C.DEFAULT_WORKERS)
    cache = Cache(C.CACHE_DIR, enabled=not args.no_cache, refresh=args.refresh)
    if args.compare:
        a_yaml, b_yaml = args.compare
        if a_yaml.resolve() == b_yaml.resolve():
            log.warning("Both compare targets are the same file.")
        ra, pa = _run_one(args, ds, client, cache, _profile(args, a_yaml, ds))
        rb, pb = _run_one(args, ds, client, cache, _profile(args, b_yaml, ds))
        out = render.write_compare(
            ra,
            rb,
            f"{pa['html'].parent.name}/report.html",
            f"{pb['html'].parent.name}/report.html",
            args.out_dir / "compare.html",
        )
        print(f"Comparison: {out}")
        reports = [ra, rb]
    else:
        report, paths = _run_one(args, ds, client, cache, _profile(args, args.target, ds))
        reports = [report]
        print(f"Report: {paths['html']}\n        {paths['md']}\n        {paths['json']}")
    return max(_summary(r) for r in reports)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run, and map every failure to a one-line message and a distinct exit code."""
    load_env_file(C.ENV_FILE)
    args = _parser().parse_args(argv)
    args.provider = pick_provider(args.provider)
    setup_logging("INFO" if args.verbose and args.log_level == "WARNING" else args.log_level, args.log_json)
    t0 = time.perf_counter()
    try:
        return _main(args)
    except EngineError as e:
        if args.debug:
            raise
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as e:  # last resort: one line for users; the traceback goes to the debug log / --debug
        if args.debug:
            raise
        log.debug("unexpected error", exc_info=True)
        print(f"error: unexpected {type(e).__name__}: {e} (re-run with --debug for the traceback)", file=sys.stderr)
        return 1
    finally:
        log.info("run finished", extra={"fields": {"elapsed_s": round(time.perf_counter() - t0, 2)}})

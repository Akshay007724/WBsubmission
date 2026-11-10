"""Provider-agnostic LLM layer: Anthropic / OpenAI / offline, disk cache, validate-and-repair loop."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

from pydantic import ValidationError

from . import config as C
from . import offline, prompts, validate
from .errors import LLMError
from .schemas import Rationale, tool_schema

log = logging.getLogger(__name__)
TOOL_NAME = "submit_rationale"


class _Usage(threading.local):
    """Token usage of the last call on THIS thread (calls run in parallel on one shared client)."""

    tokens_in: int | None = None
    tokens_out: int | None = None


class ReplayMiss(LLMError):
    """Raised by the replay provider when a prompt has no validated cache entry."""


class LLMClient(Protocol):
    """What every provider implements: complete(system, user) -> dict matching the Rationale schema."""

    provider: str
    model: str

    def complete(self, system: str, user: str) -> dict:
        """Return one parsed JSON object for the prompt (see the implementations for failure behaviour)."""
        ...


class AnthropicClient:
    """Claude via native structured outputs (messages.parse). Needs ANTHROPIC_API_KEY."""

    provider = "anthropic"

    @property
    def default_workers(self) -> int:
        """Parallel calls for this provider (read at call time so settings overrides apply)."""
        return C.ANTHROPIC_WORKERS  # read at call time so --set ANTHROPIC_WORKERS=... applies

    def __init__(self, model: str):
        self.usage = _Usage()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY is not set. Export it, or run with --provider offline.")
        import anthropic

        self.model = model
        # SDK handles exponential backoff on 429/5xx/timeouts.
        self._c = anthropic.Anthropic(max_retries=C.LLM_MAX_RETRIES, timeout=C.LLM_TIMEOUT_S)

    def complete(self, system: str, user: str) -> dict:
        """Send one page request via messages.parse; return the parsed object.

        Raises LLMError on refusal, truncation or missing output. SDK errors (timeouts, 429s after the SDK's
        retries) propagate to generate(), which records them on the page instead of failing the run.
        """
        # Current Claude models: no sampling params, no forced tool_choice; native structured outputs
        # (messages.parse) constrain the reply to the Rationale schema. Thinking is adaptive by
        # default and counts toward max_tokens, hence the generous cap.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": C.ANTHROPIC_MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_format": Rationale,
        }
        if C.ANTHROPIC_EFFORT:
            kwargs["output_config"] = {"effort": C.ANTHROPIC_EFFORT}
        resp = self._c.messages.parse(**kwargs)
        self.usage.tokens_in, self.usage.tokens_out = resp.usage.input_tokens, resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            raise LLMError(f"model declined (category={getattr(resp.stop_details, 'category', None)})")
        if resp.stop_reason == "max_tokens":
            raise LLMError("response truncated (max_tokens)")
        if resp.parsed_output is None:
            raise LLMError(f"no structured output (stop_reason={resp.stop_reason})")
        return resp.parsed_output.model_dump()


class OpenAIClient:
    """OpenAI chat-completions API; also serves OpenAI-compatible providers via subclasses."""

    provider = "openai"
    key_env = "OPENAI_API_KEY"
    base_url: str | None = None
    supports_seed = True
    max_retries = C.LLM_MAX_RETRIES
    max_tokens = C.LLM_MAX_TOKENS
    extra_params: ClassVar[dict[str, str]] = {}
    use_json_schema = False  # True: grammar-constrained decoding against our schema (llama.cpp servers)

    def __init__(self, model: str):
        self.usage = _Usage()
        key = os.environ.get(self.key_env)
        if not key:
            raise LLMError(f"{self.key_env} is not set. Export it (or put it in .env), or run with --provider offline.")
        import openai

        self.model = model
        self._c = openai.OpenAI(
            api_key=key, base_url=self.base_url, max_retries=self.max_retries, timeout=C.LLM_TIMEOUT_S
        )

    def complete(self, system: str, user: str) -> dict:
        """Send one page request via chat.completions (JSON mode or a schema grammar); return the parsed object.

        Raises LLMError on refusal, truncation or invalid JSON; SDK errors propagate to generate().
        """
        schema = json.dumps(tool_schema())
        extra: dict[str, Any] = {**({"seed": C.OPENAI_SEED} if self.supports_seed else {}), **self.extra_params}
        resp = self._c.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=self.max_tokens,
            **extra,
            response_format=(
                {"type": "json_schema", "json_schema": {"name": "rationale", "schema": tool_schema()}}
                if self.use_json_schema
                else {"type": "json_object"}
            ),
            messages=[
                {
                    "role": "system",
                    "content": f"{system}\nReturn ONLY a JSON object matching this JSON schema:\n{schema}",
                },
                {"role": "user", "content": user},
            ],
        )
        if getattr(resp, "usage", None):
            self.usage.tokens_in, self.usage.tokens_out = resp.usage.prompt_tokens, resp.usage.completion_tokens
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise LLMError("response truncated (length)")
        if getattr(choice.message, "refusal", None):
            raise LLMError(f"model refused: {choice.message.refusal}")
        try:
            return json.loads(_strip_fences(choice.message.content or ""))
        except json.JSONDecodeError as e:
            raise LLMError(f"invalid JSON from model: {e}") from e


def _strip_fences(text: str) -> str:
    """Models sometimes wrap JSON in ```json fences even in JSON mode; keep the outermost object."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


class GeminiClient(OpenAIClient):
    """Google Gemini through its OpenAI-compatible endpoint (verified: JSON mode works, `seed` is rejected)."""

    provider = "gemini"
    key_env = "GEMINI_API_KEY"
    base_url = C.GEMINI_BASE_URL
    supports_seed = False
    # Free-tier keys are rate limited per minute: retry longer (SDK backoff honours Retry-After).
    max_retries = C.GEMINI_MAX_RETRIES
    # Gemini 3.x "thinks" by default and bills it to max_tokens; keep it low and leave headroom.
    max_tokens = C.GEMINI_MAX_TOKENS
    extra_params: ClassVar[dict[str, str]] = {"reasoning_effort": "low"}
    default_workers = C.GEMINI_WORKERS
    _daily_quota_hit = False  # shared across threads: stop calling once the per-day cap is reached

    def complete(self, system: str, user: str) -> dict:
        """As OpenAIClient.complete, but stops calling once Gemini's per-day quota is exhausted (LLMError)."""
        import openai

        if GeminiClient._daily_quota_hit:
            raise LLMError("Gemini daily quota exhausted (skipped)")
        try:
            return super().complete(system, user)
        except openai.RateLimitError as e:
            if "PerDay" in e.response.text:
                GeminiClient._daily_quota_hit = True
                raise LLMError(
                    "Gemini free-tier daily quota exhausted (per model, per day). "
                    "Wait for reset, use another --model, or enable billing."
                ) from e
            raise


class LocalClient(OpenAIClient):
    """Local OpenAI-compatible server: Docker Model Runner (default) or Ollama. No API key needed.

    Set LOCAL_LLM_BASE_URL to switch, e.g. http://localhost:11434/v1 for Ollama.
    """

    provider = "local"
    max_retries = 1
    # Small local models echo a schema back instead of filling it; a grammar makes that impossible.
    use_json_schema = True
    max_tokens = C.LOCAL_MAX_TOKENS
    default_workers = C.LOCAL_WORKERS

    def __init__(self, model: str):
        import openai

        self.usage = _Usage()
        self.model = model
        self.base_url = os.environ.get("LOCAL_LLM_BASE_URL", C.LOCAL_BASE_URL)
        self._c = openai.OpenAI(
            api_key="local", base_url=self.base_url, max_retries=self.max_retries, timeout=C.LOCAL_TIMEOUT_S
        )
        try:
            available = [m.id for m in self._c.models.list()]
        except openai.APIConnectionError as e:
            raise LLMError(
                f"No local model server at {self.base_url}. Start Docker Model Runner with host TCP "
                "enabled (docker desktop enable model-runner --tcp 12434) or Ollama, or set "
                "LOCAL_LLM_BASE_URL."
            ) from e
        # Servers report names differently ("docker.io/ai/llama3.2:latest" vs "ai/llama3.2:latest").
        match = [a for a in available if a == model or a.endswith("/" + model) or a.split(":")[0].endswith(model)]
        if not match:
            raise LLMError(
                f"Model '{model}' not found at {self.base_url}. Available: {available or 'none (still pulling?)'}"
            )
        self.model = match[0]


class ReplayClient:
    """Serve validated answers from .cache/llm only: no API key, no network.

    Cache keys include provider and model, so replay looks up entries written by
    ``cache_provider`` (default: the Anthropic run that produced sample_output/).
    """

    provider = "replay"

    def __init__(self, model: str | None = None, cache_provider: str = "anthropic"):
        self.cache_provider = cache_provider
        self.model = model or C.DEFAULT_MODELS[cache_provider]

    def complete(self, system: str, user: str) -> dict:
        """Always raises ReplayMiss: reaching the client means the cache had no validated answer."""
        raise ReplayMiss(
            f"no cached answer for this prompt under {self.cache_provider}/{self.model} "
            f"(prompt version {C.PROMPT_VERSION}); run once with a live provider, or use --provider offline"
        )


class OfflineClient:
    """Deterministic template writer. For tests and key-less review; clearly labelled in output."""

    provider = "offline"
    model = C.DEFAULT_MODELS["offline"]

    def __init__(self, model: str | None = None):
        self._packs: dict[str, tuple[dict, dict]] = {}

    def register(self, pack: dict, market: dict) -> None:
        """Give the offline writer the pack and market for one acquirer."""
        self._packs[pack["acquirer"]] = (pack, market)

    def complete(self, system: str, user: str) -> dict:
        """Return the deterministic template page for the acquirer named in the prompt (never fails)."""
        first_line = user.split("acquirer: ", 1)[1].split("\n", 1)[0]
        name = first_line.rsplit(" (", 1)[0]  # names may contain parentheses, e.g. "Aspen Dental (ADMI)"
        pack, market = self._packs[name]
        return offline.write(pack, market)


def make_client(provider: str, model: str | None) -> LLMClient:
    """Build the client for a provider; raises LLMError for an unknown provider or a missing key."""
    if provider not in C.DEFAULT_MODELS:
        raise LLMError(f"Unknown provider '{provider}'. Use anthropic, openai, gemini, local, replay or offline.")
    model = model or C.DEFAULT_MODELS[provider]
    if provider == "anthropic":
        return AnthropicClient(model)
    if provider == "openai":
        return OpenAIClient(model)
    if provider == "gemini":
        return GeminiClient(model)
    if provider == "local":
        return LocalClient(model)
    if provider == "offline":
        return OfflineClient()
    if provider == "replay":
        return ReplayClient(model if model != C.DEFAULT_MODELS.get("replay") else None)
    raise LLMError(f"Unknown provider '{provider}'. Use anthropic, openai, gemini, local, replay or offline.")


@dataclass
class Result:
    """Outcome of generating one page: status verified | repaired | unverified | failed, plus timing and tokens."""

    acquirer: str
    rationale: Rationale | None
    status: str  # verified | repaired | unverified | failed
    issues: list[str] = field(default_factory=list)
    cached: bool = False
    attempts: int = 0
    cache_key: str = ""
    elapsed_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


class Cache:
    """Disk cache of validated answers keyed by provider, model, prompt version, prompts and schema."""

    def __init__(self, root: Path, enabled: bool = True, refresh: bool = False):
        self.root, self.enabled, self.refresh = root, enabled, refresh

    def key(self, client: LLMClient, system: str, user: str) -> str:
        """Stable sha256 of everything that determines the answer."""
        provider = getattr(client, "cache_provider", client.provider)  # replay reads another provider's entries
        blob = json.dumps([provider, client.model, C.PROMPT_VERSION, system, user, tool_schema()], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        """Cached entry, or None on miss, refresh, disabled cache or an unreadable file (logged)."""
        p = self.root / f"{key}.json"
        if not self.enabled or self.refresh or not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("Ignoring unreadable cache entry %s", p.name)
            return None

    def put(self, key: str, value: dict) -> None:
        """Atomic write (temp file + rename). A cache that cannot be written is logged, not fatal."""
        if not self.enabled:
            return
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / f"{key}.{os.getpid()}.{threading.get_ident()}.tmp"
            tmp.write_text(json.dumps(value, indent=1), encoding="utf-8")
            tmp.replace(self.root / f"{key}.json")
        except OSError as e:
            log.warning("cache write failed (%s); continuing without caching this page", e)


def generate(
    client: LLMClient,
    cache: Cache,
    system: str,
    user: str,
    pack: dict,
    market: dict,
    target: dict,
    max_repairs: int = 1,
) -> Result:
    """One page: cache lookup, then call -> schema -> checker, with up to ``max_repairs`` repair attempts.

    Never raises for provider problems: they become a ``failed``/``unverified`` Result so one bad page cannot
    lose the other nine. Only validated answers are cached.
    """
    name = pack["acquirer"]
    t0 = time.perf_counter()
    key = cache.key(client, system, user)
    hit = cache.get(key)
    # Live runs reuse only checked answers; replay reproduces exactly what was shipped, unverified pages included.
    if hit and (hit.get("status") in ("verified", "repaired") or client.provider == "replay"):
        log.info("page from cache", extra={"fields": {"acquirer": name, "cache": "hit"}})
        return Result(
            name, Rationale.model_validate(hit["rationale"]), hit["status"], hit.get("issues", []), True, 0, key
        )

    usage = getattr(client, "usage", None)
    tin = tout = 0
    prompt, last, issues = user, None, []
    raw: dict | None = None

    def done(status: str, rationale: Rationale | None, issues: list[str], attempts: int) -> Result:
        res = Result(
            name, rationale, status, issues, False, attempts, key, round(time.perf_counter() - t0, 2), tin, tout
        )
        log.info(
            "page done",
            extra={
                "fields": {
                    "acquirer": name,
                    "status": status,
                    "attempts": attempts,
                    "elapsed_s": res.elapsed_s,
                    "tokens_in": tin,
                    "tokens_out": tout,
                }
            },
        )
        return res

    for attempt in range(1, max_repairs + 2):
        try:
            raw = client.complete(system, prompt)
        except ReplayMiss as e:  # not retryable: the cache will not change between attempts
            return done("failed", None, [str(e)], attempt)
        except Exception as e:  # network, truncation, refusal: each provider SDK raises its own types
            log.warning("%s attempt %d failed: %s: %s", name, attempt, type(e).__name__, e)
            issues = [f"call failed: {type(e).__name__}: {e}"]
            continue
        finally:
            if usage is not None and usage.tokens_in is not None:
                tin, tout = tin + usage.tokens_in, tout + (usage.tokens_out or 0)
                usage.tokens_in = usage.tokens_out = None
        try:
            last = Rationale.model_validate(raw)
        except ValidationError as e:
            issues = [f"schema: {err['loc']}: {err['msg']}" for err in e.errors()][:8]
            raw = None
        else:
            issues = validate.check(last, pack, market, target)
        if not issues and last:
            status = "verified" if attempt == 1 else "repaired"
            cache.put(key, {"rationale": last.model_dump(), "status": status})
            return done(status, last, [], attempt)
        log.info("repair needed", extra={"fields": {"acquirer": name, "attempt": attempt, "issues": len(issues)}})
        prev = json.dumps(raw, default=str)[:6000] if raw else "(unparseable)"
        prompt = f"{user}\n\n<previous_answer>\n{prev}\n</previous_answer>\n\n{prompts.repair_message(issues)}"
    if last:  # stored for replay fidelity; live runs ignore unverified entries and try again
        cache.put(key, {"rationale": last.model_dump(), "status": "unverified", "issues": issues})
    return done("unverified" if last else "failed", last, issues, max_repairs + 1)

# Architecture

## Design principle

**Code ranks and computes; the LLM writes from a fact pack; the checker verifies.**

- Ranking, comps, size roles and conviction priors are deterministic, so a banker can ask "why is X #3?" and get a numeric answer.
- The LLM turns a dense, labelled fact pack into prose that compares each buyer with the others.
- Every number, cited deal and qualitative claim the model writes is checked against the facts it was shown. Failures go back to the model once, with the specific errors. Anything still wrong is shown with a badge, never hidden.

## Data flow

```
 any M&A CSV ──► schema.py ──► units.py ──► data.load ──► Dataset (df, vocab, features, notes, schema)
                 headers→canonical  $/K/M/B, %, dates      degrade, never invent        │
                                                                                         ▼
 target (YAML/flags) ──► profile.validate_against(ds)                         eda.run  (stage 0)
                                                                              findings + charts
                                                                                         │
                                ┌────────────────────────────────────────────────────────┘
                                ▼
 scoring.score_deals ──► aggregate ──► select (mix floor, qualification) ──► top N + displaced
                                │
                                ▼
 evidence.market_comps (anchor, like-for-like pricing peers) ──► evidence.build × N
       descriptor · size role · precedents · pricing_vs_peers · risks · conviction prior
                                │
                                ▼
 evidence.add_closest_competitors ──► prompts.build_user × N
                                │
                                ▼
 llm.generate × N (parallel, cache) ──► validate.check ──► repair once ──► engine.cross_page_feedback
                                                                          (near-duplicate theses,
                                                                           repeated risks) ──► repair once
                                │
                                ▼
 engine: rank/conviction notes, warnings, timing, cost ──► eval.score ──► render (HTML / MD / JSON)
```

## Module map

| Module | Responsibility |
|---|---|
| `cli.py` | Arguments, logging (`--log-level`, `--log-json`), error → exit code mapping (3 data · 4 config · 5 LLM · 6 render · 1 unexpected / empty result) |
| `settings.py` | Validated overrides of tunables: defaults < `--config` YAML < `ACQ_*` env < `--set`. A fingerprint goes in the run stamp |
| `config.py` | Every default weight and threshold, each with a one-line rationale |
| `errors.py` | `EngineError` → `DataError`, `ConfigError`, `LLMError`, `RenderError` |
| `schema.py` | Canonical fields (core / important / optional), alias table, fuzzy matching, `--schema-map` |
| `units.py` | Money (suffixes, header hints, magnitude), percent vs fraction, dates → year + quarter |
| `data.py` | Load, map, convert, dedupe; feature flags (on / computed / inferred / off: why); vocabulary and sector adjacency (healthcare preset or derived from co-investment) |
| `eda.py`, `charts.py` | Stage 0: adaptive data-quality findings, each with the engine's response; four SVG charts |
| `profile.py` | `TargetProfile`; validation against the file's vocabulary |
| `scoring.py` | Per-deal relevance, acquirer score with shrinkage, mix floor with qualification and a displaced/replacement log |
| `evidence.py` | Fact packs: descriptor, size role, comps, like-for-like pricing, risks, conviction points, closest competitor |
| `prompts.py` | System rules (all checkable) and tagged-JSON user prompt; `PROMPT_VERSION` |
| `schemas.py` | `Rationale`: the structured output contract |
| `llm.py` | Providers (Anthropic, OpenAI, Gemini, local, replay, offline), disk cache with atomic writes, `generate()` with repair, timing and token usage |
| `validate.py` | The checker: numbers, IDs, attribution before and after IDs, counted integers, `$`, specialist claims, market-wide figures, competitor named, banned phrases, word limits; cross-page similarity and repeated risks |
| `engine.py` | Orchestration, cross-page repair pass, rank-vs-conviction notes, coverage warnings, cost estimate |
| `eval.py` | Deterministic scoring of a report (no LLM judge) |
| `render.py` | Autoescaped HTML, Markdown, JSON; EDA outputs |
| `secretscan.py` | Pre-ship scan for key-shaped strings |

## Key decisions

- **Two valuation benchmarks.**
  - *How a buyer pays* compares its comps with same-type peers under identical filters, at any margin.
  - *What this target should command* is the high-margin anchor.

  In this data, margin and multiple correlate (0.56), so mixing the two made most buyers look cheap.
- **Rank vs conviction.** Score is overall comparable-deal activity; conviction is direct evidence for this deal. Order stays by score, and each divergence is explained on the page.
- **Market-wide facts appear once**, in a Market context section. The checker rejects restated market figures and risks repeated on three or more other pages.
- **Graceful degradation.** A missing column switches off only what needs it, and the report says what and why. Missing values stay missing.
- **Reproducibility.**
  - Only validated answers are cached, under keys covering provider, model, prompt version, prompts and schema.
  - `--provider replay` is byte-identical by design.
  - Ranking is deterministic, with a fixed tie-break.

## Extending

| To add… | Do this |
|---|---|
| A provider | Implement `complete(system, user) -> dict` and register it in `make_client` |
| A data vendor's headers | Add aliases in `schema.FIELDS`, or ship a `--schema-map` YAML |
| A sector preset | Add pairs to `config._ADJ_PAIRS`; files outside the preset already use derived adjacency |
| A checker rule | Add a function in `validate.py`, call it from `check()`, state the rule in `prompts.SYSTEM`, bump `PROMPT_VERSION`, and add a passing and a failing test |

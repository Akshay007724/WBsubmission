# M&A Practitioner Review

*Written from the seat of a senior M&A banker with 15+ years in healthcare buy-side and sell-side work, reviewing the engine before its output goes in front of an MD or client.*

**The standard:** every name on the buyer list must survive the question "why are they here, and why above the next one?" Every number must trace to a deal. Nothing on the page may be something a client could catch as wrong.

## 1. Issues found and fixed in code

> **Superseded (CSV-only rule):** rows 8 (regulatory context) and 9 (defunct buyers via `external_context`) relied on knowledge from outside the CSV, so they were removed. Every fact now comes from the deals file. Checking entity status (later acquisitions, bankruptcies) is a manual pre-send step for the banker; see section 4.

| # | What a banker would catch | Why it matters on deal day | Fix | Where |
|---|---|---|---|---|
| 1 | **Minority investments counted as acquisitions.** A growth-equity minority stake was weighted 71% of a control deal. | We are selling the company. A minority investor isn't a buyer of control, and its multiple carries no control premium. | Weight cut to 0.25 (→0.5 after softening). Minority, SPAC and Merger-of-Equals deals excluded from valuation comps. Each precedent carries `usable_as_comp`. | `config.DEAL_TYPE_W`, `COMP_EXCLUDE_DEAL_TYPES`, `evidence.control_comps` |
| 2 | **SPAC and MoE multiples in the comp set.** | SPAC marks were inflated in 2020–21, and MoE deals are relative-value with no premium. Both distort the "what will it trade at" answer. | Excluded from all comp sets; the report lists what was excluded. | `evidence.market_comps` |
| 3 | **One blended peer set for every buyer.** | Sponsors underwrite to an LBO return; strategics underwrite to synergies. Benchmarking a strategic against sponsor pricing (or the reverse) misleads the valuation discussion. | Peer medians split by buyer type. Each acquirer's premium/discount is measured against its **own type**. A last-3-years subset is shown for the market cycle. | `peer_by_buyer_type`, `peer_recent_3y`, `valuation_vs_market` |
| 4 | **"Geographic Expansion" shown as every buyer's top tag.** | It appears on 90% of deals. An MD reads "top rationale: geographic expansion" ten times and stops trusting the page. | Tags are reported by **lift vs. the market base rate**; only tags at least 1.2× more common than usual are shown. | `evidence.tag_lift` |
| 5 | **No sense of whether $200M is a platform or an add-on for the buyer.** | This is the first question for any sponsor. A $200M regional services business is a platform for a mid-market fund but an add-on (through a portfolio company) for a mega-fund. For strategics: is it a stretch? | `size_position` gives the target's percentile in the buyer's own deal history and a role: platform-sized, add-on or stretch. A stretch costs a conviction point. The prompt requires the platform-vs-add-on call. | `evidence.size_position`, `prompts` rule 9 |
| 6 | **Buyers silently dropped by the strategic/sponsor floor.** Welsh Carson (the healthcare-services specialist) scored #9 and disappeared. | The first question from an MD will be "where's WCAS?" A list that quietly drops a natural buyer looks like an error. | New **Also considered** table: overall rank, score, and "displaced by mix floor" when it outscored a mix pick. | `engine.also_considered`, report |
| 7 | **Antitrust read only as "big strategic in the same sector."** | Healthcare-provider antitrust is **local-market**: the FTC challenges regional hospital and physician consolidation far more than national scale. | If the target region is specified, strategics with deals in that region get `antitrust_signal`. Unspecified region → labelled "not specified". | `evidence._risk_facts` |
| 8 | **No regulatory context.** | Above the HSR threshold every buyer files. State healthcare transaction-notice laws now reach PE deals (CA, OR, WA, NY, MA among others). CPOM affects sponsors buying physician businesses. Not-for-profit buyers need board approval and bond financing. These are the timing risks a client asks about. | A number-free `REGULATORY_CONTEXT` block, explicitly labelled outside the dataset. The LLM may use it only where this buyer's data makes it relevant. It is also printed in the report appendix. | `config.REGULATORY_CONTEXT`, `prompts` rule 10 |
| 9 | **Defunct or absorbed buyers can be pitched.** | Real names, partly synthetic deals: e.g. Steward Health Care (2024 bankruptcy), Kindred at Home (now CenterWell under Humana), Cerner (Oracle), Change Healthcare (UnitedHealth). Pitching a buyer that no longer exists ends the meeting. | The prompt **requires** the model to state any known acquisition, merger, renaming or bankruptcy in `external_context`, plus a risk flag. Bankers remove such names with `--exclude`, which is persisted. | `prompts` rule 11, `feedback.py` |
| 10 | **Acquirer IQR from 1–2 deals.** | A "range" from one or two data points is false precision, and a VP would be embarrassed defending it. | IQR is only computed with n ≥ 3. The conviction IQR signal needs at least 2 of the buyer's own comps. | `evidence.multiple_stats`, `suggested_conviction` |
| 11 | **Implied valuation not cross-checked.** | The brief gives EV but no EBITDA or revenue, so the engine shouldn't only back into EBITDA. | Implied EBITDA (EV ÷ peer median) and implied revenue (÷ assumed margin) are both shown. For the default target they are $12.8M EBITDA and $59.5M revenue → 3.36x EV/Revenue, consistent with the 3.27x peer median. | `market_comps` |
| 12 | **Thin-data target crashed the report.** | An arbitrary target with no comparable deals (e.g. $25B Dental) raised a template error. A tool that crashes on an unusual target won't be used. | Templates degrade to "n/a"; a regression test covers it. | templates, `tests/test_cli.py` |

All fixes are covered by tests, mainly `tests/test_ma_rules.py`, `tests/test_size_role.py`, `tests/test_pricing_benchmark.py` and `tests/test_rank_conviction.py`. The full suite is 151 tests, all passing.

## 2. Judgment calls a senior banker should sign off on

These are defensible defaults, but they are choices, not facts. Each is a single constant in `config.py`.

| Decision | Default | Banker view |
|---|---|---|
| Sector adjacency (HC Services ↔ Physician Groups 0.8, Home Health 0.7, Behavioral 0.7, Dental 0.6, RCM 0.5…) | `SECTOR_ADJ_PAIRS` | Reasonable for "provider services". If the target is actually physician-practice-heavy, Physician Groups should go to 0.9 and Dental up. Ask the deal team what the business really is. |
| Size tolerance σ = ln 2.5 | $80M and $500M precedents score ≈0.61 of a perfect size match | Matches how banks think about a $200M comp band ($100–400M). |
| Recency half-life 4 years | 2020 deals score 0.5 | Fits a typical PE fund cycle (5-year investment period). |
| Strategic/sponsor floor of ≥3 each | `MIN_PER_TYPE` | Standard practice for a broad buyer list. For a sponsor-only or strategic-only process, set it to 0 and say so in the README. |
| "Strong margin" = dataset p75 (21.5%) | `STRONG_MARGIN_PCTL` | Replace with the real number as soon as the client shares financials. |
| Withdrawn deals = intent signal (0.4) + execution penalty | `OUTCOME_W`, `W_EXEC_PENALTY` | A withdrawn deal often means the buyer walked on price or diligence. That is useful to know for the process, not disqualifying. |
| Not-for-profit list | `NONPROFIT_SYSTEMS` | Hand-maintained; verify before use. |

## 3. What the engine still cannot know (human check before it goes out)

1. **Corporate parents.** CenterWell Home Health and Kindred at Home, CVS/Aetna, UnitedHealth and Change Healthcare are treated as separate buyers. A banker would roll them up to the parent before calling.
2. **Current portfolio conflicts.** A sponsor may already own a direct competitor. That makes it either the best add-on buyer or conflicted, and the dataset has no portfolio data.
3. **Fund capacity.** Fund size, vintage and dry powder decide whether a sponsor is actually investing. Here it is proxied only by deal pace over the last 3 years.
4. **Data provenance.** The deals are partly synthetic: EV ≠ multiple × EBITDA in 403/500 rows, and labels are noisy. **Do not quote these multiples to a client as market data.** They are fine for demonstrating the method.
5. **Sector reclassification.** `sub_sector` repeats `sector`, so the engine can't tell, for example, ASC from urgent care within Healthcare Services.
6. **Synergy field.** `synergy_pct_of_deal` is *higher* for sponsors (8.8%) than strategics (6.5%), which is economically backwards. It is deliberately **not used** for ability-to-pay.

## 4. Pre-send checklist (for the VP)

- [ ] Every page shows a `verified` badge. Read the issues on any `unverified` page.
- [ ] "Also considered": is anyone displaced who should be on the main list for this client?
- [ ] Conviction spread is not all one level. Each High has at least 2 cited deals.
- [ ] `external_context`: check any status claim (acquired, renamed, bankrupt) and `--exclude` defunct names.
- [ ] Valuation: peer n ≥ 5, and the buyer is benchmarked against its own type.
- [ ] Regulatory flags match the target's actual state once it is known; re-run with `--geography`.
- [ ] Replace the assumed margin (and region) with client numbers, then re-run.

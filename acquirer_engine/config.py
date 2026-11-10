"""All tunable constants in one place. Every weight here is a judgment call; see README 'Scoring'."""

from __future__ import annotations

import math
from pathlib import Path

_SOURCE_TREE = Path(__file__).resolve().parent.parent
# Running from a checkout: paths are relative to the repo. Installed with pip: relative to the working
# directory (site-packages holds no data, cache or .env).
ROOT = _SOURCE_TREE if (_SOURCE_TREE / "pyproject.toml").is_file() else Path.cwd()
DEFAULT_DATA = ROOT / "ma_transactions_500.csv"
DEFAULT_OUT = ROOT / "out"
CACHE_DIR = ROOT / ".cache" / "llm"
FEEDBACK_FILE = ROOT / "feedback.json"

PROMPT_VERSION = "v1.6"  # v1.1 pricing (B); v1.2 size role (C); v1.3 descriptor (E); v1.4 market context (F)
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4o",
    # gemini-2.5-* is closed to new keys; gemini-3.8-flash timed out in testing (Nov 2026).
    "gemini": "gemini-3.6-flash",
    "local": "ai/llama3.2:latest",
    "offline": "offline-template",
    "replay": "claude-sonnet-5-5",  # replays the cache written by this anthropic model
}
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
ENV_FILE = ROOT / ".env"
GEMINI_WORKERS = 2
# Local models (Docker Model Runner / Ollama). One request at a time: a laptop GPU serialises anyway.
LOCAL_BASE_URL = "http://localhost:12434/engines/v1"
LOCAL_WORKERS = 1
LOCAL_TIMEOUT_S = 600
LOCAL_MAX_TOKENS = 3000
GEMINI_MAX_RETRIES = 6
GEMINI_MAX_TOKENS = 8000
LLM_TIMEOUT_S = 60
LLM_MAX_RETRIES = 2
LLM_MAX_TOKENS = 2500
DEFAULT_WORKERS = 5
OPENAI_SEED = 7
ANTHROPIC_MAX_TOKENS = 16000
# Sonnet 5.5 defaults to high; medium keeps a 10-page run inside the brief's 60s target.
ANTHROPIC_EFFORT: str | None = "medium"
ANTHROPIC_WORKERS = 10  # one call per acquirer, all at once
# $ per million tokens (input, output) for the run-stamp cost estimate; unknown models show no estimate.
PRICES_PER_MTOK = {"claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50), "claude-opus-5-5": (4.0, 20.0)}
THIN_SECTOR_BAND_DEALS = 20  # fewer target-sector deals than this near the target's size -> warn up front

REQUIRED_COLUMNS = [
    "transaction_id",
    "target_company",
    "acquirer",
    "sector",
    "deal_year",
    "deal_quarter",
    "deal_type",
    "geography",
    "financing_type",
    "deal_size_mm",
    "target_revenue_mm",
    "target_ebitda_mm",
    "ebitda_margin_pct",
    "revenue_growth_pct",
    "ev_ebitda_multiple",
    "ev_revenue_multiple",
    "outcome",
    "strategic_rationale_tags",
    "num_bidders",
    "days_to_close",
    "acquirer_type",
    "target_ownership_pre",
]
NUMERIC_COLUMNS = [
    "deal_year",
    "deal_size_mm",
    "target_revenue_mm",
    "target_ebitda_mm",
    "ebitda_margin_pct",
    "revenue_growth_pct",
    "ev_ebitda_multiple",
    "ev_revenue_multiple",
    "synergy_pct_of_deal",
    "num_bidders",
    "days_to_close",
]

SECTORS = [
    "Healthcare Services",
    "Physician Groups",
    "Home Health/Hospice",
    "Behavioral Health",
    "Dental",
    "Revenue Cycle",
    "Health IT",
    "Health Insurance",
    "Medical Devices",
    "Pharma/Biotech",
]
REGIONS = [
    "Northeast",
    "Southeast",
    "Midwest",
    "Mountain West",
    "Mid-Atlantic",
    "Great Plains",
    "West Coast",
    "Southwest",
]
BROAD_GEOS = {"National", "Multi-Regional"}

# Sector adjacency (symmetric). Rationale: provider-services businesses share reimbursement
# exposure, labor models and roll-up playbooks; tech / payer / product sectors share less.
SECTOR_ADJ_DEFAULT = 0.2
_ADJ_PAIRS = {
    ("Healthcare Services", "Physician Groups"): 0.8,
    ("Healthcare Services", "Home Health/Hospice"): 0.7,
    ("Healthcare Services", "Behavioral Health"): 0.7,
    ("Healthcare Services", "Dental"): 0.6,
    ("Healthcare Services", "Revenue Cycle"): 0.5,
    ("Healthcare Services", "Health Insurance"): 0.4,
    ("Healthcare Services", "Health IT"): 0.35,
    ("Healthcare Services", "Medical Devices"): 0.15,
    ("Healthcare Services", "Pharma/Biotech"): 0.1,
    ("Physician Groups", "Dental"): 0.6,
    ("Physician Groups", "Behavioral Health"): 0.6,
    ("Physician Groups", "Home Health/Hospice"): 0.5,
    ("Physician Groups", "Revenue Cycle"): 0.5,
    ("Physician Groups", "Health Insurance"): 0.5,
    ("Behavioral Health", "Home Health/Hospice"): 0.5,
    ("Behavioral Health", "Dental"): 0.4,
    ("Home Health/Hospice", "Dental"): 0.3,
    ("Home Health/Hospice", "Health Insurance"): 0.4,
    ("Revenue Cycle", "Health IT"): 0.8,
    ("Health IT", "Health Insurance"): 0.3,
    ("Health IT", "Medical Devices"): 0.3,
    ("Medical Devices", "Pharma/Biotech"): 0.5,
    ("Revenue Cycle", "Health Insurance"): 0.3,
}
SECTOR_ADJ_PAIRS = {**_ADJ_PAIRS, **{(b, a): v for (a, b), v in _ADJ_PAIRS.items()}}

# Derived adjacency (non-preset sectors) is mapped onto the preset's scale: cosine 0 -> 0.1, cosine 1 -> 0.9.
ADJ_DERIVED_MIN, ADJ_DERIVED_MAX = 0.1, 0.9
# Neutral weights. A missing column contributes 1.0 (no information, no penalty). A value present in the
# file but absent from the preset maps gets the midpoint of that preset's scale.
UNKNOWN_OWNERSHIP_W, UNKNOWN_DEAL_TYPE_W, UNKNOWN_OUTCOME_W = 0.75, 0.7, 0.5

SIZE_SIGMA = math.log(2.5)  # $80M / $500M deal vs $200M target -> ~0.61
RECENCY_HALF_LIFE_Y = 4.0
STALE_PENDING_YEARS = 2
OUTCOME_W = {"Closed": 1.0, "Pending": 0.6, "Withdrawn": 0.4, "Terminated": 0.3, "Rumored": 0.3}
STALE_PENDING_W = 0.3
OWNERSHIP_W = {"Private": 1.0, "PE-Backed": 0.85, "Non-Profit": 0.6, "Public": 0.5}
MARGIN_SIGMA_PP = 6.0
GEO_SAME, GEO_BROAD, GEO_OTHER = 1.0, 0.85, 0.6
DEAL_TYPE_W = {
    "Platform Investment": 1.0,
    "Bolt-on Acquisition": 1.0,
    "Strategic Acquisition": 1.0,
    "Leveraged Buyout": 0.9,
    "Recapitalization": 0.8,
    "Management Buyout": 0.7,
    "Carve-out": 0.6,
    # Not control acquisitions: a minority investor is not a buyer of the company.
    "Minority Investment": 0.25,
    "SPAC Merger": 0.4,
    "Merger of Equals": 0.4,
}
# Deal types whose multiples are not control-transaction comps (no control premium / SPAC marks).
COMP_EXCLUDE_DEAL_TYPES = {"Minority Investment", "SPAC Merger", "Merger of Equals"}
# Tags carried by most deals carry no signal; we report tags by lift vs. the dataset base rate.
TAG_MIN_LIFT = 1.2
MIN_ACQ_COMPS_FOR_SIGNAL = 2
# A buyer's median within ±1 turn of its like-for-like peer median is reported as "in line with".
PRICING_IN_LINE_TURNS = 1.0
# Size positioning (where the target's EV sits in the buyer's own deal-size history):
ADD_ON_PCTL = 35  # below the buyer's 35th percentile: smaller than ~2/3 of what it buys -> add-on / tuck-in
ADD_ON_MEDIAN_FRACTION = 0.5  # or below half its median deal: too small to be a platform for this buyer
STRETCH_P90_MULTIPLE = 2.0  # above 2x its 90th percentile (or above its max): a stretch, not routine
PLATFORM_EVIDENCE_MIN = 2  # >= 2 deals labelled Platform Investment: compare with those sizes instead
PLATFORM_MATCH_MULTIPLE = 2.0  # ...but only if the target is <= 2x their median; far above them, percentiles decide
# (labels are noisy, so one labelled deal is not evidence)

DEAL_TYPE_EXPONENT = 0.5  # labels are noisy -> soften their influence

# Acquirer aggregation
SHRINKAGE_K = 3.0
W_MASS, W_FIT, W_BEST, W_EXEC_PENALTY = 0.45, 0.30, 0.25, 0.15
MIN_DEAL_RELEVANCE = 0.15
MIN_PER_TYPE = 3
ALSO_CONSIDERED_N = 10
STRONG_MARGIN_PCTL = 0.75

# Evidence / comps
ADJACENT_SECTOR_MIN = 0.5
COMP_SIZE_BAND = 2.5
COMP_WIDE_BAND = 4.0
MIN_COMPS = 5
TOP_PRECEDENTS = 5
RELEVANT_DEAL_W = 0.3
RECENT_YEARS = 3
ANTITRUST_LARGE_DEAL_MM = 1000
ANTITRUST_MIN_SECTOR_DEALS = 3
HIGH_BIDDERS = 4

# Suggested conviction (evidence points). LLM may move it one notch with a stated reason.
CONV_BAND_STRONG, CONV_BAND_SOME = 5, 3
CONV_FAIL_SHARE = 0.25
CONV_HIGH, CONV_MEDIUM = 5.0, 3.0


BANNED_PHRASES = [
    "leading",
    "well-positioned",
    "well positioned",
    "track record of strategic acquisitions",
    "robust",
    "best-in-class",
    "world-class",
    "uniquely positioned",
    "proven track record",
    "strong fit",
    "strong track record",
    "demonstrates its ability",
    "make it a strong",
]
DISTINCT_JACCARD_MAX = 0.35
REPEATED_RISK_MIN_PAGES = 3  # a risk near-identical on 3+ other pages is market-wide, not buyer-specific
WORD_LIMITS = {
    "acquirer_overview": 110,
    "strategic_fit_thesis": 160,
    "valuation_context": 110,
    "conviction_rationale": 60,
}

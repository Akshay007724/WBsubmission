"""Structured output contract for one acquirer one-pager."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RiskCategory = Literal[
    "antitrust", "integration", "financing_capacity", "competitive_process", "valuation", "execution", "other"
]


class Precedent(BaseModel):
    """A cited prior deal and why it is relevant."""

    transaction_id: str
    why_relevant: str


class RiskFlag(BaseModel):
    """One risk, categorised, tied to a specific data point."""

    category: RiskCategory
    detail: str


class Conviction(BaseModel):
    """High / Medium / Low with a data-tied rationale."""

    level: Literal["High", "Medium", "Low"]
    rationale: str


class Rationale(BaseModel):
    """The one-pager the LLM must return (validated by pydantic, then by validate.check)."""

    headline: str = Field(description="One line, <= 20 words: the single sharpest reason this buyer transacts")
    acquirer_overview: str
    strategic_fit_thesis: str
    precedent_activity: list[Precedent] = Field(min_length=1, max_length=5)
    valuation_context: str
    risk_flags: list[RiskFlag] = Field(min_length=2, max_length=4)
    conviction: Conviction


def tool_schema() -> dict:
    """JSON schema of Rationale, used for structured output and in cache keys."""
    schema = Rationale.model_json_schema()
    return schema

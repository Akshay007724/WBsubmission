"""Four small EDA charts as self-contained SVG strings (no plotting dependency).

Colours use CSS variables with fallbacks, so the same SVG reads in light and dark pages.
"""

from __future__ import annotations

import math
from html import escape

import pandas as pd

W, H, PAD = 420, 220, 34
INK, MUTE, BAR = "var(--ink,#16202b)", "var(--mute,#5b6875)", "var(--accent,#0b4f7c)"


def _frame(title: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-label="{escape(title)}" style="font:11px sans-serif;max-width:100%">'
        f'<title>{escape(title)}</title><text x="{PAD}" y="14" fill="{INK}" font-weight="600">{escape(title)}</text>'
        f'<line x1="{PAD}" y1="{H - PAD}" x2="{W - 8}" y2="{H - PAD}" stroke="{MUTE}"/>{body}</svg>'
    )


def bars(title: str, labels: list[str], values: list[float], rotate: bool = False) -> str:
    """Vertical bar chart with value labels."""
    if not values:
        return _frame(title, f'<text x="{PAD}" y="{H / 2}" fill="{MUTE}">no data</text>')
    top = max(values) or 1
    n = len(values)
    bw = (W - PAD - 8) / n
    out = []
    for i, (lab, v) in enumerate(zip(labels, values, strict=False)):
        h = (H - 2 * PAD - 14) * v / top
        x = PAD + i * bw + bw * 0.15
        y = H - PAD - h
        out.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw * 0.7:.1f}" height="{h:.1f}" fill="{BAR}"><title>'
            f"{escape(str(lab))}: {v:g}</title></rect>"
        )
        out.append(
            f'<text x="{x + bw * 0.35:.1f}" y="{y - 3:.1f}" text-anchor="middle" '
            f'fill="{INK}" font-size="9">{v:g}</text>'
        )
        tx, ty = x + bw * 0.35, H - PAD + 12
        anchor, tf = ("end", f' transform="rotate(-35 {tx:.1f} {ty:.1f})"') if rotate else ("middle", "")
        out.append(
            f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}" fill="{MUTE}" font-size="9"{tf}>'
            f"{escape(str(lab))[:16]}</text>"
        )
    return _frame(title, "".join(out))


def size_histogram(sizes: pd.Series) -> str:
    """Deal sizes on log-spaced bins ($M), the scale the size kernel works on."""
    s = sizes.dropna()
    s = s[s > 0]
    if s.empty:
        return bars("Deal size distribution ($M, log bins)", [], [])
    edges = [10**e for e in range(math.floor(math.log10(s.min())), math.ceil(math.log10(s.max())) + 1)]
    if len(edges) < 2:
        edges = [s.min(), s.max() + 1]
    counts = pd.cut(s, bins=edges, include_lowest=True).value_counts(sort=False)
    labels = [f"{int(iv.left):,}-{int(iv.right):,}" for iv in counts.index]
    return bars("Deal size distribution ($M, log bins)", labels, [float(c) for c in counts.values])


def scatter(title: str, x: pd.Series, y: pd.Series, xlab: str, ylab: str) -> str:
    """Scatter with axis ranges from the data; points beyond the 99th percentile are clipped to the edge."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if d.empty:
        return _frame(title, f'<text x="{PAD}" y="{H / 2}" fill="{MUTE}">no data</text>')
    x0, x1 = d["x"].quantile(0.01), d["x"].quantile(0.99)
    y0, y1 = d["y"].quantile(0.01), d["y"].quantile(0.99)
    sx = lambda v: PAD + (min(max(v, x0), x1) - x0) / ((x1 - x0) or 1) * (W - PAD - 12)  # noqa: E731
    sy = lambda v: H - PAD - (min(max(v, y0), y1) - y0) / ((y1 - y0) or 1) * (H - 2 * PAD - 10)  # noqa: E731
    pts = "".join(
        f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="2" fill="{BAR}" fill-opacity="0.45"/>'
        for a, b in zip(d["x"], d["y"], strict=False)
    )
    axes = (
        f'<line x1="{PAD}" y1="{PAD - 10}" x2="{PAD}" y2="{H - PAD}" stroke="{MUTE}"/>'
        f'<text x="{W - 10}" y="{H - 8}" text-anchor="end" fill="{MUTE}">{escape(xlab)} ({x0:.0f}-{x1:.0f})</text>'
        f'<text x="{PAD + 4}" y="{PAD - 2}" fill="{MUTE}">{escape(ylab)} ({y0:.0f}-{y1:.0f})</text>'
    )
    return _frame(title, axes + pts)


def eda_charts(df: pd.DataFrame) -> dict[str, str]:
    """The four stage-0 charts: size distribution, deals per year, sector mix, multiple vs margin."""
    years = df["deal_year"].dropna().astype(int).value_counts().sort_index()
    sectors = df["sector"].value_counts().head(12)
    return {
        "size_distribution": size_histogram(df["deal_size_mm"]),
        "deals_per_year": bars("Deals per year", [str(y) for y in years.index], [float(v) for v in years.values]),
        "sector_mix": bars("Sector mix (top 12)", list(sectors.index), [float(v) for v in sectors.values], rotate=True),
        "multiple_vs_margin": scatter(
            "EV/EBITDA vs EBITDA margin", df["ebitda_margin_pct"], df["ev_ebitda_multiple"], "margin %", "EV/EBITDA x"
        ),
    }

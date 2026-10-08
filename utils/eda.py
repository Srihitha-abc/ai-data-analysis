"""Exploratory data analysis: column detection, KPIs and Plotly charts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Colorblind-checked categorical palette, assigned in this fixed order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PRIMARY = PALETTE[0]
OTHER_GRAY = "#b4b2a9"  # neutral color for the folded "Other" slice
# Diverging scale for correlations: red (negative) -> gray (0) -> blue (positive).
DIVERGING = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]
MAX_PIE_SLICES = 6  # beyond this, smaller categories fold into "Other"


def style_figure(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Apply the app's shared chart theme."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)) if title else None,
        template="plotly_white",
        colorway=PALETTE,
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#0b0b0b"),
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(font_size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, linecolor="#d6d5d0")
    fig.update_yaxes(gridcolor="#ecebe7", zeroline=False)
    return fig


# ---------- Column detection ----------

def detect_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Group columns into date, numeric and categorical."""
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    numeric = [c for c in df.select_dtypes(include="number").columns if not _looks_like_id(df, c)]
    categorical = [
        c for c in df.columns
        if c not in dates and c not in numeric and 1 < df[c].nunique() <= 50
    ]
    return {"date": dates, "numeric": numeric, "categorical": categorical}


def _looks_like_id(df: pd.DataFrame, col: str) -> bool:
    return "id" in col.lower().split("_") and df[col].is_unique


# ---------- KPIs ----------

def compute_kpis(df: pd.DataFrame, metrics: list[str], date_col: str | None) -> list[dict]:
    """Total / average per metric, plus last-period vs previous-period change."""
    period, last, prev = None, None, None
    if date_col:
        span_days = (df[date_col].max() - df[date_col].min()).days
        period = "M" if span_days > 90 else "W" if span_days > 14 else "D"
        periods = df[date_col].dt.to_period(period)
        unique = sorted(periods.dropna().unique())
        if len(unique) >= 2:
            last, prev = unique[-1], unique[-2]

    kpis = []
    for col in metrics:
        kpi = {"label": col, "total": df[col].sum(), "average": df[col].mean(), "change": None}
        if last is not None:
            last_val = df.loc[periods == last, col].sum()
            prev_val = df.loc[periods == prev, col].sum()
            if prev_val:
                kpi["change"] = (last_val - prev_val) / abs(prev_val) * 100
                kpi["change_label"] = f"{last} vs {prev}"
        kpis.append(kpi)
    return kpis


def format_number(value: float) -> str:
    """Compact formatting: 1.2K, 3.4M, 5.6B."""
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"{value / threshold:,.1f}{suffix}"
    return f"{value:,.2f}" if value % 1 else f"{value:,.0f}"


# ---------- Charts ----------

def line_over_time(df: pd.DataFrame, date_col: str, metric: str, color_by: str | None = None) -> go.Figure:
    span_days = (df[date_col].max() - df[date_col].min()).days
    freq = "MS" if span_days > 180 else "W" if span_days > 30 else "D"
    keys = [pd.Grouper(key=date_col, freq=freq)] + ([color_by] if color_by else [])
    grouped = df.groupby(keys)[metric].sum().reset_index()
    if color_by:
        top = df.groupby(color_by)[metric].sum().nlargest(len(PALETTE)).index
        grouped = grouped[grouped[color_by].isin(top)]
    fig = px.line(grouped, x=date_col, y=metric, color=color_by, markers=True)
    fig.update_traces(line=dict(width=2), marker=dict(size=8))
    fig.update_layout(hovermode="x unified")
    label = {"MS": "monthly", "W": "weekly", "D": "daily"}[freq]
    return style_figure(fig, f"{metric} over time ({label})")


def donut_chart(df: pd.DataFrame, category: str, metric: str | None = None) -> go.Figure:
    values = df.groupby(category)[metric].sum() if metric else df[category].value_counts()
    values = values.sort_values(ascending=False)
    colors = PALETTE[: len(values)]
    if len(values) > MAX_PIE_SLICES:
        values = pd.concat([values.iloc[: MAX_PIE_SLICES - 1], pd.Series({"Other": values.iloc[MAX_PIE_SLICES - 1:].sum()})])
        colors = PALETTE[: MAX_PIE_SLICES - 1] + [OTHER_GRAY]
    fig = go.Figure(go.Pie(
        labels=values.index.astype(str), values=values.values, hole=0.55, sort=False,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="percent", hovertemplate="%{label}: %{value:,.0f} (%{percent})<extra></extra>",
    ))
    style_figure(fig, f"{metric or 'Row count'} by {category}")
    # Legend below the donut so it never collides with the title.
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5))
    return fig


def missing_values_chart(df: pd.DataFrame) -> go.Figure:
    """Missing values per column in the raw data (shown on the Data tab)."""
    missing = df.isna().sum()
    total = int(missing.sum())
    fig = go.Figure(go.Bar(
        x=missing.index.astype(str), y=missing.values, marker_color=PRIMARY,
        hovertemplate="%{x}: %{y:,} missing<extra></extra>",
    ))
    fig.update_layout(bargap=0.35, yaxis_title="Missing values", height=280)
    fig.update_yaxes(rangemode="tozero")
    return style_figure(fig, "Missing values per column (raw data)" + ("" if total else ": none found"))


def bar_chart(df: pd.DataFrame, category: str, metric: str, agg: str = "sum", top_n: int = 15) -> go.Figure:
    grouped = df.groupby(category)[metric].agg(agg).sort_values(ascending=False).head(top_n).reset_index()
    fig = px.bar(grouped, x=category, y=metric, text_auto=".3s")
    fig.update_traces(marker_color=PRIMARY, textposition="outside", cliponaxis=False)
    fig.update_layout(bargap=0.35)
    return style_figure(fig, f"{agg.title()} of {metric} by {category}")


def histogram(df: pd.DataFrame, column: str, bins: int = 30) -> go.Figure:
    # Bin in numpy and send ~30 bars, not every raw value, so big files stay fast.
    values = df[column].dropna().to_numpy(dtype=float)
    counts, edges = np.histogram(values, bins=bins) if len(values) else ([], [0])
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(
        x=centers, y=counts, width=np.diff(edges), marker_color=PRIMARY,
        marker_line=dict(color="white", width=1),
        customdata=np.stack([edges[:-1], edges[1:]], axis=-1) if len(counts) else None,
        hovertemplate="%{customdata[0]:,.4~g} to %{customdata[1]:,.4~g}<br>Count: %{y:,}<extra></extra>",
    ))
    fig.update_layout(bargap=0, xaxis_title=column, yaxis_title="Count")
    return style_figure(fig, f"Distribution of {column}")


def correlation_heatmap(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    corr = df[columns].corr(numeric_only=True).round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale=DIVERGING, zmin=-1, zmax=1, aspect="auto")
    fig.update_xaxes(side="bottom")
    return style_figure(fig, "Correlation between numeric columns")

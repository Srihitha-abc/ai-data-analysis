"""Statistics for the AI report and Markdown -> PDF export."""

from __future__ import annotations

import re

import pandas as pd

from utils.eda import detect_columns


def summary_statistics(df: pd.DataFrame) -> str:
    """Pre-compute real numbers so the LLM reports facts instead of guessing."""
    cols = detect_columns(df)
    parts = [f"Shape: {df.shape[0]} rows x {df.shape[1]} columns"]
    if cols["numeric"]:
        parts.append("Numeric summary:\n" + df[cols["numeric"]].describe().round(2).to_string())
    for cat in cols["categorical"][:4]:
        for metric in cols["numeric"][:2]:
            top = df.groupby(cat)[metric].sum().sort_values(ascending=False).round(2)
            parts.append(f"Total {metric} by {cat}:\n{top.head(10).to_string()}")
    if cols["date"] and cols["numeric"]:
        date_col, metric = cols["date"][0], cols["numeric"][-1]
        monthly = df.groupby(df[date_col].dt.to_period("M"))[metric].sum().round(2)
        parts.append(f"Monthly {metric}:\n{monthly.to_string()}")
    if len(cols["numeric"]) >= 2:
        corr = df[cols["numeric"]].corr().round(2)
        parts.append("Correlations:\n" + corr.to_string())
    for col in cols["numeric"]:  # outliers via the IQR rule
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
        if len(outliers):
            parts.append(f"{col}: {len(outliers)} outlier(s) by IQR rule (max {df[col].max():,.2f})")
    return "\n\n".join(parts)


def _latin1(text: str) -> str:
    """Built-in PDF fonts only support Latin-1; swap common Unicode characters."""
    replacements = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-",
                    "—": "-", "•": "-", "…": "...", "₹": "Rs.", "€": "EUR", "‑": "-", "‐": "-", "−": "-", "≈": "~",
                    " ": " ", " ": " ", "≤": "<=", "≥": ">=", "→": "->"}
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


def markdown_to_pdf(markdown: str) -> bytes:
    """Render simple Markdown (headings, bullets, bold, paragraphs) to PDF bytes."""
    from fpdf import FPDF  # imported lazily: only needed when a PDF is downloaded

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    sizes = {1: 18, 2: 14, 3: 12}

    for raw in markdown.splitlines():
        line = _latin1(raw.rstrip()).replace("`", "")
        if not line.strip():
            pdf.ln(3)
            continue
        if re.match(r"^\s*\|?\s*:?-{3,}", line):  # table separator row |---|---|
            continue
        if line.lstrip().startswith("|"):  # table row: render cells separated by spaces
            line = "   ".join(c.strip() for c in line.strip().strip("|").split("|"))
        heading = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading:
            level = min(len(heading.group(1)), 3)
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", sizes[level])
            pdf.set_text_color(42, 120, 214) if level == 1 else pdf.set_text_color(11, 11, 11)
            pdf.multi_cell(0, 8, heading.group(2).replace("**", ""), align="L", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(11, 11, 11)
            continue
        bullet = re.match(r"^(\s*)(?:[-*+]|\d+\.)\s+(.*)", line)
        pdf.set_font("Helvetica", "", 11)
        if bullet:
            depth = min(len(bullet.group(1).expandtabs(4)) // 2, 3)  # nested bullets are indented
            pdf.set_x(pdf.l_margin + 4 + depth * 6)
            pdf.multi_cell(0, 6, "-  " + bullet.group(2), markdown=True, align="L", new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.multi_cell(0, 6, line, markdown=True, align="L", new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())

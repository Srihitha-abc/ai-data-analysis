"""AI Data Analysis: turn raw data into insights in seconds.

Run with:  python -m streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html
import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from utils import eda
from utils.cleaning import clean_data, load_file, profile
from utils.llm import GROQ_MODEL, LLMError, explain_result, generate_code, generate_report, get_api_key
from utils.report import markdown_to_pdf, summary_statistics
from utils.sandbox import result_to_text, run_code

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_sales.csv"
EXAMPLE_QUESTIONS = [
    "Which product category had the highest growth last month?",
    "What are the top 5 products by total revenue?",
    "Show monthly revenue by region as a line chart",
    "What is the average revenue per unit for each category?",
]
CHART_BUILDERS = {
    "line": eda.line_over_time,
    "donut": eda.donut_chart,
    "bar": eda.bar_chart,
    "histogram": eda.histogram,
    "heatmap": eda.correlation_heatmap,
    "missing": eda.missing_values_chart,
}

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.18);
        border-left: 4px solid #2a78d6;
        border-radius: 12px;
        padding: 14px 18px;
    }
    div[data-testid="stMetricLabel"] p { font-weight: 600; }
    .hero h1 { margin-bottom: 0; }
    .hero p { opacity: 0.75; margin-top: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Cached computations ----------
# Arguments starting with "_" are not hashed; `dataset_key` identifies the data instead,
# so a 50 MB DataFrame isn't re-hashed on every rerun.

@st.cache_data(show_spinner=False, max_entries=8)
def load_and_clean(file_bytes: bytes, file_name: str, do_clean: bool):
    buffer = io.BytesIO(file_bytes)
    buffer.name = file_name
    raw = load_file(buffer)
    if do_clean:
        cleaned, log = clean_data(raw)
    else:
        cleaned, log = raw, ["Automatic cleaning is turned off."]
    return raw, cleaned, log


@st.cache_data(show_spinner=False, max_entries=16)
def get_profile(dataset_key: str, which: str, _df: pd.DataFrame) -> pd.DataFrame:
    return profile(_df)


@st.cache_data(show_spinner=False, max_entries=8)
def get_columns(dataset_key: str, _df: pd.DataFrame) -> dict:
    return eda.detect_columns(_df)


@st.cache_data(show_spinner=False, max_entries=256)
def get_chart(kind: str, dataset_key: str, _df: pd.DataFrame, args: tuple):
    return CHART_BUILDERS[kind](_df, *args)


@st.cache_data(show_spinner=False, max_entries=4)
def get_pdf(report_md: str) -> bytes:
    return markdown_to_pdf(report_md)


def md_safe(text: str) -> str:
    """Escape $ so Streamlit doesn't render "$1,200 and $300" as a LaTeX formula."""
    return text.replace("$", r"\$")


def reset_analysis_state():
    st.session_state.chat = []
    st.session_state.report = None


def make_tabs(labels: list[str]):
    """Lazy tabs: only the open tab's code runs (Streamlit >= 1.5x)."""
    try:
        return st.tabs(labels, key="main_tabs", on_change="rerun")
    except TypeError:  # older Streamlit: every tab renders
        return st.tabs(labels)


def is_open(tab) -> bool:
    return getattr(tab, "open", None) is not False


for key, default in (("chat", []), ("report", None), ("dataset_key", None)):
    st.session_state.setdefault(key, default)


# ---------- Sidebar ----------

with st.sidebar:
    st.title("📊 AI Data Analyst")
    st.caption("Upload data → clean → explore → ask questions in plain English.")

    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    use_sample = st.toggle("Use the sample sales dataset", value=uploaded is None,
                           disabled=uploaded is not None)
    do_clean = st.toggle("Automatic data cleaning", value=True)

    st.divider()
    if get_api_key():
        st.success("Groq API key loaded", icon="🔑")
    else:
        st.error("GROQ_API_KEY not found. AI features are disabled. Add it to a `.env` file.", icon="🔑")
    st.caption(f"Model: `{GROQ_MODEL}`")

    if st.button("Clear chat & report", width="stretch"):
        reset_analysis_state()

if uploaded is not None:
    file_bytes, file_name = uploaded.getvalue(), uploaded.name
elif use_sample and SAMPLE_PATH.exists():
    file_bytes, file_name = SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name
else:
    st.markdown('<div class="hero"><h1>AI Data Analyst</h1>'
                "<p>Turn raw data into insights in seconds.</p></div>", unsafe_allow_html=True)
    st.info("👈 Upload a CSV/Excel file or turn on the sample dataset to get started.")
    st.stop()

try:
    with st.spinner("Loading and cleaning data..."):
        raw_df, df, clean_log = load_and_clean(file_bytes, file_name, do_clean)
except Exception as e:
    st.error(f"Couldn't read **{file_name}**: {e}")
    st.stop()

if df.empty:
    st.error("The dataset is empty after cleaning.")
    st.stop()

# A new dataset clears the previous chat and report.
dataset_key = f"{hashlib.md5(file_bytes).hexdigest()}-{do_clean}"
if st.session_state.dataset_key != dataset_key:
    st.session_state.dataset_key = dataset_key
    reset_analysis_state()

cols = get_columns(dataset_key, df)

st.markdown(f'<div class="hero"><h1>AI Data Analyst</h1><p>Analyzing <b>{html.escape(file_name)}</b> · '
            f"{len(df):,} rows · {len(df.columns)} columns</p></div>", unsafe_allow_html=True)

tab_data, tab_dash, tab_chat, tab_report = make_tabs(
    ["📋 Data & Cleaning", "📊 Dashboard", "💬 Ask your data", "📝 AI Report"]
)


# ---------- Tab 1: Data & cleaning ----------

@st.fragment
def render_data_tab():
    c1, c2, c3, c4 = st.columns(4)
    removed = len(raw_df) - len(df)
    c1.metric("Rows", f"{len(df):,}", delta=f"-{removed:,} after cleaning" if removed else None)
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing values (raw)", f"{int(raw_df.isna().sum().sum()):,}")
    c4.metric("Duplicate rows (raw)", f"{int(raw_df.duplicated().sum()):,}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("🧹 Cleaning summary")
        for item in clean_log:
            st.markdown(f"- {item}")
    with right:
        # This chart also makes the browser load Plotly early, so the Dashboard opens faster.
        st.plotly_chart(get_chart("missing", f"{dataset_key}-raw", raw_df, ()), width="stretch")

    st.subheader("Preview")
    view = st.radio("Show", ["Cleaned data", "Raw data"], horizontal=True, label_visibility="collapsed")
    shown = df if view == "Cleaned data" else raw_df
    st.dataframe(shown.head(100), width="stretch", height=320)

    st.subheader("Column profile")
    st.dataframe(get_profile(dataset_key, view, shown), width="stretch", hide_index=True)

    st.download_button("⬇️ Download cleaned CSV", df.to_csv(index=False).encode(),
                       file_name=f"cleaned_{Path(file_name).stem}.csv", mime="text/csv", on_click="ignore")


# ---------- Tab 2: Auto EDA dashboard ----------

@st.fragment
def render_dashboard():
    """A fragment: changing a dropdown reruns only this function, not the whole app."""
    if not cols["numeric"]:
        st.warning("No numeric columns found, so there is nothing to chart.")
        return

    none_label = lambda v: "(none)" if v is None else str(v)
    f1, f2, f3 = st.columns(3)
    metric = f1.selectbox("Metric", cols["numeric"], index=len(cols["numeric"]) - 1)
    category = f2.selectbox("Category", cols["categorical"] or [None], format_func=none_label)
    date_col = f3.selectbox("Date column", cols["date"] or [None], format_func=none_label)

    # KPI cards
    kpi_metrics = st.multiselect("KPI cards", cols["numeric"], default=cols["numeric"][-3:][::-1])
    kpis = eda.compute_kpis(df, kpi_metrics, date_col)
    kpi_cols = st.columns(len(kpis) + 1)
    kpi_cols[0].metric("Records", f"{len(df):,}")
    for col, kpi in zip(kpi_cols[1:], kpis):
        delta = f"{kpi['change']:+.1f}%" if kpi["change"] is not None else None
        col.metric(f"Total {kpi['label']}", eda.format_number(kpi["total"]), delta=delta,
                   help=f"Average: {eda.format_number(kpi['average'])}"
                        + (f" · Change: {kpi['change_label']}" if delta else ""))

    def chart(kind: str, *args):
        st.plotly_chart(get_chart(kind, dataset_key, df, args), width="stretch")

    st.write("")
    left, right = st.columns([2, 1])
    with left:
        if date_col:
            split = st.checkbox(f"Split by {category}", value=False) if category else False
            chart("line", date_col, metric, category if split else None)
        else:
            st.info("No date column detected, so the time-series chart is hidden.")
    with right:
        if category:
            chart("donut", category, metric)

    left, right = st.columns(2)
    with left:
        if category:
            agg = st.radio("Aggregation", ["sum", "mean", "count"], horizontal=True)
            chart("bar", category, metric, agg)
        else:
            st.info("No category column detected, so the bar and donut charts are hidden.")
    with right:
        hist_col = st.selectbox("Histogram column", cols["numeric"], index=cols["numeric"].index(metric))
        chart("histogram", hist_col)

    if len(cols["numeric"]) >= 2:
        chart("heatmap", cols["numeric"])


# ---------- Tab 3: Ask your data ----------

def answer_question(question: str) -> dict:
    """Question -> Groq writes pandas code -> run it safely -> explain the result."""
    turn = {"question": question, "code": None, "answer": None, "result": None, "fig": None, "error": None}
    try:
        code = generate_code(question, df, st.session_state.chat)
        execution = run_code(code, df)
        if execution.error:  # one self-correction attempt: send the error back to the model
            code = generate_code(question, df, st.session_state.chat,
                                 previous_error=execution.error, previous_code=code)
            execution = run_code(code, df)
        turn["code"] = code
        if execution.error:
            turn["error"] = execution.error
            return turn
        turn["result"] = execution.result
        if execution.fig is not None:
            turn["fig"] = eda.style_figure(execution.fig, execution.fig.layout.title.text)
        if execution.result is None and execution.fig is None:
            turn["answer"] = ("I couldn't compute an answer to that. I can only analyze the loaded data "
                              "(I can't access files, the internet or your system).")
        elif execution.result is None:  # chart only: nothing to put into words
            turn["answer"] = "Here's the chart you asked for."
        elif isinstance(execution.result, str):  # already a sentence (e.g. a polite refusal)
            turn["answer"] = execution.result
        else:
            turn["answer"] = explain_result(question, result_to_text(execution.result),
                                            has_chart=execution.fig is not None)
    except LLMError as e:
        turn["error"] = str(e)
    return turn


def render_turn(turn: dict):
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if turn["error"]:
            st.error(turn["error"])
        if turn["answer"]:
            st.markdown(md_safe(turn["answer"]))
        result = turn["result"]
        if isinstance(result, (pd.DataFrame, pd.Series)):
            st.dataframe(result, width="stretch")
        elif result is not None and not turn["answer"]:
            st.code(result_to_text(result))
        if turn["fig"] is not None:
            st.plotly_chart(turn["fig"], width="stretch")
        if turn["code"]:
            with st.expander("View generated code"):
                st.code(turn["code"], language="python")


@st.fragment
def render_chat():
    st.caption("Ask questions in plain English. Groq writes pandas code, the app runs it in a "
               "restricted sandbox, and the answer appears below.")

    pending = None
    if not st.session_state.chat:
        st.markdown("**Try one of these:**")
        q_cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            if q_cols[i % 2].button(q, key=f"example_{i}", width="stretch"):
                pending = q

    for turn in st.session_state.chat:
        render_turn(turn)

    typed = st.chat_input("Ask a question about your data...")
    question = typed or pending
    if question:
        with st.spinner("Thinking..."):
            turn = answer_question(question)
        st.session_state.chat.append(turn)
        st.rerun()  # cheap: only the open tab re-runs


# ---------- Tab 4: AI summary report ----------

@st.fragment
def render_report():
    st.caption("Groq writes a report from statistics computed by pandas, so the numbers are real.")
    if st.button("✨ Generate AI report", type="primary"):
        with st.spinner("Computing statistics and writing the report..."):
            try:
                st.session_state.report = generate_report(df, summary_statistics(df))
            except LLMError as e:
                st.error(str(e))

    report_md = st.session_state.report
    if report_md:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button("⬇️ Markdown", report_md, file_name=f"insights_{stamp}.md",
                           mime="text/markdown", on_click="ignore")
        try:
            d2.download_button("⬇️ PDF", get_pdf(report_md), file_name=f"insights_{stamp}.pdf",
                               mime="application/pdf", on_click="ignore")
        except Exception as e:
            d2.warning(f"PDF export failed: {e}")
        with st.container(border=True):
            st.markdown(md_safe(report_md))


# Only the open tab runs, so the first load and tab switches stay fast.
for tab, render in ((tab_data, render_data_tab), (tab_dash, render_dashboard),
                    (tab_chat, render_chat), (tab_report, render_report)):
    if is_open(tab):
        with tab:
            render()

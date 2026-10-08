"""Groq LLM integration. Every API request goes through `call_groq()`."""

from __future__ import annotations

import os
import re

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------- Config: change the model here ----------
# Groq retires models over time (Llama 3.3 70B is no longer served), so you can also
# override the model with GROQ_MODEL in .env. If a model is retired, the fallbacks are tried.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
TEMPERATURE = 0.1   # low = more deterministic code
MAX_TOKENS = 3000   # reasoning models count their hidden reasoning toward this limit
SAMPLE_ROWS = 5


class LLMError(Exception):
    """An error with a user-friendly message, safe to show in the UI."""


def get_api_key() -> str | None:
    """Read GROQ_API_KEY from .env / environment, falling back to Streamlit secrets."""
    key = os.getenv("GROQ_API_KEY")
    if key and key.strip():
        return key.strip()
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY")
    except Exception:  # no secrets file configured
        return None


def call_groq(messages: list[dict], temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS) -> str:
    """Send chat messages to Groq and return the reply text. Raises LLMError on failure."""
    import groq

    api_key = get_api_key()
    if not api_key:
        raise LLMError(
            "GROQ_API_KEY is missing. Create a `.env` file with `GROQ_API_KEY=your_key` "
            "(get a free key at https://console.groq.com/keys) and restart the app."
        )

    client = groq.Groq(api_key=api_key, max_retries=2, timeout=60)
    for model in [GROQ_MODEL] + [m for m in FALLBACK_MODELS if m != GROQ_MODEL]:
        # gpt-oss models "think" before answering; low effort keeps answers fast.
        extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, **extra,
            )
            break
        except groq.NotFoundError:
            continue  # model retired: try the next one
        except groq.AuthenticationError:
            raise LLMError("Groq rejected the API key. Check GROQ_API_KEY in your `.env` file.")
        except groq.RateLimitError:
            raise LLMError("Groq's rate limit was reached (the free tier allows a limited number of "
                           "requests per minute). Wait a few seconds and try again.")
        except groq.APIConnectionError:
            raise LLMError("Couldn't reach Groq. Check your internet connection and try again.")
        except groq.APIStatusError as e:
            raise LLMError(f"Groq returned an error ({e.status_code}). Please try again shortly.")
    else:
        raise LLMError(f"None of the configured models ({GROQ_MODEL}, {', '.join(FALLBACK_MODELS)}) are "
                       "available. Set GROQ_MODEL in .env to a model from https://console.groq.com/docs/models.")

    content = response.choices[0].message.content or ""
    if not content.strip():
        raise LLMError("The model returned an empty answer. Please try again or rephrase the question.")
    return content


# ---------- Prompt building ----------

def describe_dataframe(df: pd.DataFrame) -> str:
    """Compact schema + sample rows to give the model context without sending the whole dataset."""
    lines = [f"Rows: {len(df)}, Columns: {len(df.columns)}", "", "Columns:"]
    for col in df.columns:
        s = df[col]
        info = f"- {col!r} ({s.dtype})"
        if pd.api.types.is_numeric_dtype(s):
            info += f": min={s.min():.4g}, max={s.max():.4g}, mean={s.mean():.4g}"
        elif pd.api.types.is_datetime64_any_dtype(s):
            if s.notna().any():
                info += f": from {s.min().date()} to {s.max().date()}"
        else:
            top = s.astype(str).value_counts().head(8).index.tolist()
            info += f": {s.nunique()} unique, e.g. {top}"
        lines.append(info)
    lines += ["", f"First {SAMPLE_ROWS} rows:", df.head(SAMPLE_ROWS).to_string(index=False)]
    return "\n".join(lines)


CODE_SYSTEM_PROMPT = """You are an expert Python data analyst. You write pandas code to answer questions about a DataFrame named `df`.

Rules:
- `df`, `pd` (pandas), `np` (numpy), `px` (plotly.express) and `go` (plotly.graph_objects) are already available. Do NOT import anything.
- Do NOT read or write files, use the network, or call os/sys/subprocess/open/eval/exec.
- Store the final answer in a variable named `result` (a number, string, Series or DataFrame).
  When you make a chart, `result` must still hold the data behind it (e.g. the aggregated DataFrame).
- If a chart would help, create a Plotly figure with a title in a variable named `fig`; otherwise set `fig = None`. Never call fig.show().
- Do not use while loops.
- Use exact column names from the schema. Datetime columns are already parsed.
- Prefer a small Series/DataFrame with the supporting numbers over a bare label (e.g. growth % per category, not just the winner's name).
- "Last month" means the most recent month present in the data, not today's date.
- If the request is not about analyzing `df` (e.g. files, system, API keys), set `result = "I can only analyze the loaded dataset."` and `fig = None`.
- Return ONLY the Python code inside one ```python block, no explanation."""


def extract_code(text: str) -> str:
    """Pull the code out of a ```python ... ``` block (or return the text as-is)."""
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


def generate_code(question: str, df: pd.DataFrame, history: list[dict] | None = None,
                  previous_error: str | None = None, previous_code: str | None = None) -> str:
    """Ask Groq for pandas code that answers `question`."""
    messages = [{"role": "system", "content": CODE_SYSTEM_PROMPT},
                {"role": "user", "content": f"DataFrame schema:\n{describe_dataframe(df)}"}]
    # Include the last few questions so follow-ups like "now by region" make sense.
    successful = [t for t in (history or []) if t.get("code") and not t.get("error")]
    for turn in successful[-3:]:
        messages.append({"role": "user", "content": f"Question: {turn['question']}"})
        messages.append({"role": "assistant", "content": f"```python\n{turn.get('code', '')}\n```"})
    messages.append({"role": "user", "content": f"Question: {question}"})
    if previous_error:
        messages.append({"role": "assistant", "content": f"```python\n{previous_code}\n```"})
        messages.append({"role": "user", "content": f"That code failed with:\n{previous_error}\nFix it and return the full corrected code."})
    return extract_code(call_groq(messages))


def explain_result(question: str, result_text: str, has_chart: bool = False) -> str:
    """Turn the computed result into a short plain-English answer."""
    done = "The app has already shown the user the requested chart and table. " if has_chart else ""
    messages = [
        {"role": "system", "content": (
            "You write a 1-3 sentence plain-English summary of a data analysis result for the user, "
            "including the key numbers. The user's request has already been fulfilled by the app; your "
            "only job is to summarize what the result shows. Use only the numbers given and never invent "
            "figures. Format numbers readably. Do not write code.")},
        {"role": "user", "content": f"User's request: {question}\n{done}\nResult:\n{result_text}\n\nSummary:"},
    ]
    return call_groq(messages, temperature=0.3, max_tokens=1000)


def generate_report(df: pd.DataFrame, stats_text: str) -> str:
    """Ask Groq for a Markdown insights report."""
    messages = [
        {"role": "system", "content": (
            "You are a senior data analyst writing a concise report for a business audience. "
            "Use Markdown with these sections: # Data Insights Report, ## Overview, ## Key Insights, "
            "## Trends, ## Anomalies & Data Quality, ## Recommendations. Use bullet points and cite "
            "specific numbers from the statistics provided. Never invent numbers.")},
        {"role": "user", "content": f"Dataset description:\n{describe_dataframe(df)}\n\nComputed statistics:\n{stats_text}"},
    ]
    return call_groq(messages, temperature=0.4, max_tokens=2000)

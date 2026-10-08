# 📊 AI Data Analyst

**Turn raw data into insights in seconds.** Upload a CSV or Excel file and the app cleans it, builds an interactive dashboard, answers plain-English questions by writing and running its own pandas code, and writes a downloadable insights report. It is built with Python, Pandas, Streamlit, Plotly and the Groq LLM API.

---

## ✨ Features

- **Upload data**: CSV / Excel upload with a preview, shape, column types, and missing-value counts.
- **Automatic data cleaning**: trims column names, drops empty rows and columns, converts text to dates and numbers (including `$1,200` style values), removes duplicates, and fills missing values. Every change is listed in a cleaning summary.
- **Auto EDA dashboard**: KPI cards (total, average, period-over-period % change), a time-series line chart, a donut breakdown, bar chart, histogram, and correlation heatmap. Dropdowns let you choose the metric, category, and date column.
- **💬 Ask your data**: chat in natural language (for example, *"Which product category had the highest growth last month?"*). The LLM writes pandas code, the app runs it in a restricted sandbox, and you get a text answer, a result table, and a chart when relevant. The generated code is viewable for every answer.
- **📝 AI summary report**: a report of insights, trends, anomalies and recommendations, based on statistics computed by pandas. It downloads as **Markdown** or **PDF**.
- **Graceful errors**: a missing API key, rate limits, network errors, retired models and bad generated code all produce clear, friendly messages.
- **Fast**: tabs load lazily (only the open tab runs), the dashboard is a Streamlit fragment (changing a dropdown re-runs only the dashboard), charts are cached, and histograms are pre-binned so large files stay quick.

## 🖼️ Screenshots

> Add your own screenshots to a `screenshots/` folder after running the app.

| Data & cleaning | Dashboard |
|---|---|
| ![Data](screenshots/data.png) | ![Dashboard](screenshots/dashboard.png) |

| Ask your data | AI report |
|---|---|
| ![Chat](screenshots/chat.png) | ![Report](screenshots/report.png) |

## 🗂️ Project structure

```
├── app.py                  # Streamlit UI (sidebar, 4 tabs)
├── utils/
│   ├── cleaning.py         # file loading, profiling, automatic cleaning
│   ├── eda.py              # column detection, KPIs, Plotly charts + theme
│   ├── llm.py              # Groq client: call_groq() + prompt builders
│   ├── sandbox.py          # restricted execution of generated code
│   └── report.py           # statistics for the report + Markdown → PDF
├── data/
│   ├── sample_sales.csv    # ~500-row demo dataset
│   └── generate_sample.py  # script that regenerates the demo dataset
├── .streamlit/config.toml  # color theme
├── requirements.txt
└── .env.example
```

## 🚀 Setup

**1. Clone and install** (Python 3.10+):

```bash
git clone https://github.com/<your-username>/ai-data-analyst.git
cd ai-data-analyst
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**2. Get a free Groq API key**

1. Go to [console.groq.com](https://console.groq.com) and sign up (free, no card needed).
2. Open **API Keys** → **Create API Key** and copy the key (it starts with `gsk_`).

**3. Add the key.** Copy `.env.example` to `.env` and paste your key:

```
GROQ_API_KEY=gsk_your_key_here
```

`.env` is in `.gitignore`, so the key is never committed.

**4. Run**

```bash
python -m streamlit run app.py
```

(`python -m streamlit` works even when the `streamlit` command isn't on your PATH.)

The app opens at http://localhost:8501 with the sample sales dataset loaded.

**Model:** the default is `openai/gpt-oss-120b` (Groq retired Llama 3.3 70B). To change it, edit `GROQ_MODEL` at the top of `utils/llm.py` or add `GROQ_MODEL=...` to `.env`. If the model is ever retired, the app automatically falls back to the models in `FALLBACK_MODELS`. Current models are listed at [console.groq.com/docs/models](https://console.groq.com/docs/models).

## ☁️ Deploy free on Streamlit Community Cloud

1. Push the project to a **public GitHub repo**. Make sure `.env` is *not* committed.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **Create app**.
3. Pick your repo and branch, and set **Main file path** to `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy**. The app reads the key from `st.secrets` when no `.env` file exists, so no code changes are needed.

## 🧠 How "Ask your data" works

```
Question ──► Groq (schema + 5 sample rows) ──► pandas code
                                                   │
                     AST safety check ◄────────────┘
                            │ (blocked? → error shown)
                            ▼
              exec() with restricted builtins on a copy of df
                            │ (error? → sent back to Groq once to self-correct)
                            ▼
                result + optional Plotly fig
                            │
              Groq turns the result into a plain-English answer
```

1. **Context, not the whole dataset.** The model receives only the schema (column names, types, ranges, and top values) and 5 sample rows. This keeps prompts small and fast, and the full data never leaves your machine.
2. **Code generation.** A system prompt tells the model to write pandas code that stores its answer in `result` and an optional chart in `fig`. Follow-up questions include the last 3 Q&A turns as context.
3. **Safe execution.** Before running, the code's syntax tree is inspected. Imports (other than pandas/numpy/plotly, which are already provided), `open`/`eval`/`exec`, dunder attributes (`__class__`, ...), file methods (`to_csv`, `read_*`), `while` loops and huge powers like `10**10**10` are rejected. The code then runs with a whitelist of builtins, on a *copy* of the DataFrame, with a 15-second time limit enforced by a trace function that really stops runaway code.
4. **Self-correction.** If the code raises an error, the error message goes back to the model once so it can fix its own code.
5. **Grounded answer.** The *computed* result (not the model's guess) is sent back to Groq to phrase a 1–3 sentence answer. Every number comes from pandas.

> **Limitation:** an in-process Python sandbox reduces risk but is not perfectly secure. For a public multi-user deployment, run generated code in an isolated container or subprocess.

## 🎓 Skills demonstrated

- **Data Cleaning**: type inference, duplicate removal, missing-value imputation, change logging
- **Exploratory Data Analysis (EDA)**: automatic column detection, KPIs, period-over-period growth, outlier detection (IQR)
- **Data Visualization**: interactive Plotly dashboards with a consistent, colorblind-friendly theme
- **LLM + Data Integration**: prompt engineering, text-to-pandas code generation, sandboxed execution, self-correcting retries, grounded summarization
- **Software engineering**: modular structure, secret management, graceful error handling, cloud deployment

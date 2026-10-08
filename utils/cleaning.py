"""Loading and automatic cleaning of uploaded datasets."""

from __future__ import annotations

import re

import pandas as pd

# A text column is converted only if at least this share of its values parse.
CONVERSION_THRESHOLD = 0.9
DATE_NAME_HINTS = ("date", "time", "day", "month", "year", "timestamp", "created", "updated")


def load_file(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin-1")
        # European-style CSVs use ';' as the separator.
        if df.shape[1] == 1 and ";" in str(df.columns[0]):
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=";", encoding_errors="replace")
        return df
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Please upload a .csv, .xlsx or .xls file.")


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, missing values, unique values, example."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "missing": df.isna().sum().values,
            "missing %": (df.isna().mean() * 100).round(1).values,
            "unique": df.nunique().values,
            "example": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
        }
    ).astype({"example": str})


def _try_numeric(series: pd.Series) -> pd.Series | None:
    """Parse strings like '$1,200.50' or '15%' as numbers, or return None."""
    cleaned = series.astype(str).str.strip().str.replace(r"[$€£₹,%\s]", "", regex=True)
    cleaned = cleaned.replace({"": None, "nan": None, "None": None})
    converted = pd.to_numeric(cleaned, errors="coerce")
    non_null = series.notna().sum()
    if non_null and converted.notna().sum() / non_null >= CONVERSION_THRESHOLD:
        return converted
    return None


def _parse_dates(series: pd.Series) -> pd.Series:
    """Fast vectorised parse first; fall back to slower per-value 'mixed' parsing."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "could not infer format" warnings
        fast = pd.to_datetime(series, errors="coerce")
        if fast.notna().sum() >= CONVERSION_THRESHOLD * series.notna().sum():
            return fast
        return pd.to_datetime(series, errors="coerce", format="mixed")


def _try_datetime(series: pd.Series, column_name: str) -> pd.Series | None:
    """Parse a text column as dates, or return None."""
    values = series.dropna().astype(str)
    if values.empty:
        return None
    # Avoid treating plain numbers (e.g. IDs) as dates unless the name says it's a date.
    looks_like_date = any(h in column_name.lower() for h in DATE_NAME_HINTS)
    if not looks_like_date and not values.head(50).str.contains(r"[-/:]|[A-Za-z]{3}").all():
        return None
    # Test a sample first so text columns (e.g. product names) are rejected quickly.
    sample = values.sample(min(len(values), 200), random_state=0)
    if _parse_dates(sample).notna().mean() < CONVERSION_THRESHOLD:
        return None
    converted = _parse_dates(series)
    if converted.notna().sum() / series.notna().sum() >= CONVERSION_THRESHOLD:
        return converted
    return None


def _text_columns(df: pd.DataFrame) -> list[str]:
    """Object or string columns (pandas 3 uses a dedicated `str` dtype for text)."""
    return [c for c in df.columns
            if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean a DataFrame and return (cleaned_df, human-readable change log)."""
    df = df.copy()
    log: list[str] = []

    # 1. Tidy column names.
    original_cols = list(df.columns)
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    renamed = sum(str(a) != b for a, b in zip(original_cols, df.columns))
    if renamed:
        log.append(f"Trimmed whitespace in {renamed} column name(s).")

    # 2. Drop fully empty rows / columns.
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        log.append(f"Dropped {len(empty_cols)} completely empty column(s): {', '.join(empty_cols)}.")
    before = len(df)
    df = df.dropna(how="all")
    if before - len(df):
        log.append(f"Dropped {before - len(df)} completely empty row(s).")

    # 3. Strip whitespace in text cells.
    for col in _text_columns(df):
        df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
        df[col] = df[col].replace("", pd.NA)

    # 4. Fix data types: dates first, then numbers.
    for col in _text_columns(df):
        as_dates = _try_datetime(df[col], col)
        if as_dates is not None:
            df[col] = as_dates
            log.append(f"Converted **{col}** to datetime.")
            continue
        as_numbers = _try_numeric(df[col])
        if as_numbers is not None:
            df[col] = as_numbers
            log.append(f"Converted **{col}** to numeric.")

    # 5. Remove duplicate rows.
    dupes = int(df.duplicated().sum())
    if dupes:
        df = df.drop_duplicates().reset_index(drop=True)
        log.append(f"Removed {dupes} duplicate row(s).")

    # 6. Handle missing values.
    for col in df.columns:
        missing = int(df[col].isna().sum())
        if not missing:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df = df[df[col].notna()]
            log.append(f"Dropped {missing} row(s) with a missing **{col}** (dates can't be guessed).")
        elif pd.api.types.is_numeric_dtype(df[col]):
            median = df[col].median()
            df[col] = df[col].fillna(median)
            if pd.api.types.is_float_dtype(df[col]) and (df[col] % 1 == 0).all():
                median = round(median)
                df[col] = df[col].round().astype("int64")  # whole numbers stay integers
            log.append(f"Filled {missing} missing value(s) in **{col}** with the median ({median:,.2f}).")
        else:
            mode = df[col].mode()
            fill = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(fill)
            log.append(f"Filled {missing} missing value(s) in **{col}** with the most common value ('{fill}').")

    df = df.reset_index(drop=True)
    if not log:
        log.append("No issues found. The data was already clean.")
    return df, log

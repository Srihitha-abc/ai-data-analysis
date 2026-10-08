"""Generate data/sample_sales.csv: ~500 rows of realistic sales data.

A few duplicates and missing values are injected on purpose so the
cleaning step has something to fix during a demo.
Run: python data/generate_sample.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

PRODUCTS = {
    "Electronics": {"Laptop": 850, "Smartphone": 620, "Headphones": 120, "Monitor": 240},
    "Furniture": {"Office Chair": 180, "Standing Desk": 420, "Bookshelf": 95},
    "Office Supplies": {"Notebook Pack": 12, "Pen Set": 8, "Printer Paper": 25},
    "Clothing": {"Jacket": 75, "Sneakers": 90, "T-Shirt": 18},
}
REGIONS = ["North", "South", "East", "West"]
REGION_WEIGHTS = [0.3, 0.2, 0.28, 0.22]
N_ROWS = 490

dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, N_ROWS), unit="D")
rows = []
for date in sorted(dates):
    category = rng.choice(list(PRODUCTS), p=[0.35, 0.2, 0.25, 0.2])
    product = rng.choice(list(PRODUCTS[category]))
    price = PRODUCTS[category][product]
    # Upward trend through the year, plus a Q4 holiday boost.
    seasonal = 1 + 0.04 * date.month + (0.5 if date.month in (11, 12) else 0)
    base_units = 30 if price < 30 else 8 if price < 150 else 3
    units = max(1, int(rng.poisson(base_units * seasonal)))
    discount = rng.uniform(0.85, 1.0)
    rows.append({
        "date": date.strftime("%Y-%m-%d"),
        "product": product,
        "category": category,
        "region": rng.choice(REGIONS, p=REGION_WEIGHTS),
        "units": units,
        "revenue": round(units * price * discount, 2),
    })

df = pd.DataFrame(rows)

# Inject a few data-quality issues for the cleaning demo.
df.loc[rng.choice(len(df), 6, replace=False), "region"] = np.nan
df.loc[rng.choice(len(df), 5, replace=False), "units"] = np.nan
df["units"] = df["units"].astype("Int64")
df = pd.concat([df, df.sample(10, random_state=1)]).sort_values("date").reset_index(drop=True)

out = Path(__file__).parent / "sample_sales.csv"
df.to_csv(out, index=False)
print(f"Wrote {len(df)} rows to {out}")

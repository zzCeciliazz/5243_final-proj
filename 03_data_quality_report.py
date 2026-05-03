from __future__ import annotations

from pathlib import Path

import pandas as pd


# =========================
# Configuration
# =========================
LISTINGS_SNAPSHOT_DATE = "2025-04-01"
CALENDAR_SNAPSHOT_DATE = "2025-04-01"
REVIEWS_SNAPSHOT_DATE = "2026-04-01"
RESEARCH_START = "2025-04-01"
RESEARCH_END = "2026-03-31"

PROJECT_ROOT = Path(__file__).resolve().parents[0]
PROCESSED_TABLE_PATH = PROJECT_ROOT / "data" / "processed" / "nyc_airbnb_hidden_gem_model_table.csv"
UNDERVALUED_TABLE_PATH = PROJECT_ROOT / "data" / "processed" / "nyc_airbnb_undervalued_model_table.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "data_quality_report.md"


KEY_MISSING_FIELDS = [
    "price",
    "effective_price",
    "review_scores_rating",
    "review_scores_value",
    "review_scores_location",
    "distance_to_nearest_subway_km",
    "calendar_available_rate",
    "crime_count_1000m",
    "violent_crime_count_1000m",
    "crime_intensity_log_1000m",
    "log_effective_price",
    "predicted_log_price",
    "predicted_price",
    "price_residual_log",
    "price_gap",
    "undervaluation_ratio",
    "undervalued_candidate",
    "undervalued_cluster",
    "host_response_rate",
    "host_acceptance_rate",
    "bedrooms",
    "beds",
    "bathrooms",
]

SUMMARY_STATS_FIELDS = [
    "price",
    "effective_price",
    "review_scores_rating",
    "distance_to_nearest_subway_km",
    "calendar_available_rate",
    "crime_count_1000m",
    "crime_intensity_log_1000m",
    "log_effective_price",
    "predicted_price",
    "price_gap",
    "undervaluation_ratio",
]


def count_non_missing(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df[column].notna().sum())


def safe_value_counts(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return "_Column not found._"
    counts = df[column].value_counts(dropna=False).sort_index()
    if counts.empty:
        return "_No data._"
    rows = [f"- `{idx}`: {int(val)}" for idx, val in counts.items()]
    return "\n".join(rows)


def build_missing_summary(df: pd.DataFrame) -> str:
    lines = ["| field | missing_count | missing_pct |", "|---|---:|---:|"]
    n = len(df)
    for col in KEY_MISSING_FIELDS:
        if col not in df.columns:
            lines.append(f"| {col} | N/A | N/A |")
            continue
        missing_count = int(df[col].isna().sum())
        missing_pct = (missing_count / n * 100.0) if n else 0.0
        lines.append(f"| {col} | {missing_count} | {missing_pct:.2f}% |")
    return "\n".join(lines)


def build_summary_stats(df: pd.DataFrame) -> str:
    existing = [c for c in SUMMARY_STATS_FIELDS if c in df.columns]
    if not existing:
        return "_No summary fields found._"
    stats = df[existing].describe(percentiles=[0.25, 0.5, 0.75]).T
    stats = stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    return stats.to_markdown()


def main() -> None:
    table_path = UNDERVALUED_TABLE_PATH if UNDERVALUED_TABLE_PATH.exists() else PROCESSED_TABLE_PATH
    if not table_path.exists():
        raise FileNotFoundError(
            f"Processed model table is missing: {table_path}. "
            "Please run src/02_prepare_data.py first."
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Loading processed data: {table_path}")
    df = pd.read_csv(table_path)
    print(f"[INFO] Building report for {len(df)} rows and {df.shape[1]} columns...")

    listings_with_calendar = count_non_missing(df, "calendar_days")
    listings_with_reviews = int((df.get("reviews_in_window", pd.Series([0] * len(df))) > 0).sum())
    listings_with_crime = int((df.get("crime_count_1000m", pd.Series([0] * len(df))) > 0).sum())

    report = f"""# NYC Airbnb Data Quality Report

## Raw Files Used
- `data/raw/listings_{LISTINGS_SNAPSHOT_DATE}.csv.gz`
- `data/raw/calendar_{CALENDAR_SNAPSHOT_DATE}.csv.gz`
- `data/raw/reviews_{REVIEWS_SNAPSHOT_DATE}.csv.gz`
- `data/raw/mta_subway_stations.csv`
- `data/raw/nypd_complaints_{RESEARCH_START}_{RESEARCH_END}.csv`

## Final Table Shape
- Rows: **{len(df)}**
- Columns: **{df.shape[1]}**

## Research Date Windows
- Calendar filter window: **{RESEARCH_START}** to **{RESEARCH_END}**
- Reviews filter window: **{RESEARCH_START}** to **{RESEARCH_END}**

## Coverage Checks
- Listings with calendar data: **{listings_with_calendar}**
- Listings with review data in the research window: **{listings_with_reviews}**
- Listings with at least one complaint within 1000m: **{listings_with_crime}**

## Missing Value Summary (Key Fields)
{build_missing_summary(df)}

## Diagnostic Legacy Class Distribution
### `hidden_gem_label`
{safe_value_counts(df, "hidden_gem_label")}

### `overpriced_trap_label`
{safe_value_counts(df, "overpriced_trap_label")}

### `consumer_value_class`
{safe_value_counts(df, "consumer_value_class")}

## Undervalued Discovery Columns
### `undervalued_candidate`
{safe_value_counts(df, "undervalued_candidate")}

### `undervalued_cluster`
{safe_value_counts(df, "undervalued_cluster")}

## Summary Statistics
{build_summary_stats(df)}

## Major Caveats
- Calendar data reflects *future availability and displayed prices at scrape time*, not completed bookings.
- Reviews are used as a proxy for listing activity and consumer satisfaction, not exact stay transactions.
- Airbnb listing prices can change after the scrape date and may differ from final paid prices.
- Subway station data is static and does not capture temporary service disruptions or schedule quality.
- NYPD complaint counts represent reported incidents, not exact true crime rates. They may reflect reporting behavior, police activity, and local population or tourist density.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"[SUCCESS] Saved data quality report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

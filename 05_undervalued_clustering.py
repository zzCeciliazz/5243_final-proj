from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


PROJECT_ROOT = Path(__file__).resolve().parents[0]
INPUT_TABLE = PROJECT_ROOT / "data" / "processed" / "nyc_airbnb_undervalued_model_table.csv"
INPUT_CANDIDATES = PROJECT_ROOT / "outputs" / "undervalued_candidates.csv"
OUTPUT_CANDIDATES = PROJECT_ROOT / "outputs" / "undervalued_candidates.csv"
OUTPUT_CLUSTER_SUMMARY = PROJECT_ROOT / "outputs" / "undervalued_cluster_summary.csv"
OUTPUT_TABLE_UPDATED = PROJECT_ROOT / "data" / "processed" / "nyc_airbnb_undervalued_model_table.csv"
OUTPUT_DATA_DICT = PROJECT_ROOT / "data" / "processed" / "data_dictionary_prepared.csv"

EARTH_RADIUS_KM = 6371.0088
EPS_KM = 0.8
MIN_SAMPLES = 5


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        rows.append(
            {
                "column_name": col,
                "dtype": str(df[col].dtype),
                "missing_count": missing_count,
                "missing_pct": round((missing_count / n * 100.0), 4) if n else np.nan,
                "sample_values": " | ".join(df[col].dropna().astype(str).head(3).tolist()),
            }
        )
    return pd.DataFrame(rows)


def most_common_neighbourhoods(series: pd.Series, top_n: int = 3) -> str:
    vals = series.dropna().astype(str)
    if vals.empty:
        return "Unknown"
    return " | ".join(vals.value_counts().head(top_n).index.tolist())


def main() -> None:
    if not INPUT_TABLE.exists():
        raise FileNotFoundError(f"Missing input table: {INPUT_TABLE}. Run src/04_price_prediction.py first.")

    if INPUT_CANDIDATES.exists():
        candidates = pd.read_csv(INPUT_CANDIDATES, low_memory=False)
    else:
        base = pd.read_csv(INPUT_TABLE, low_memory=False)
        candidates = base[base.get("undervalued_candidate", 0) == 1].copy()

    if candidates.empty:
        print("[WARNING] No undervalued candidates found. Saving empty clustering outputs.")
        candidates["undervalued_cluster"] = np.nan
        candidates.to_csv(OUTPUT_CANDIDATES, index=False)
        pd.DataFrame(
            columns=[
                "cluster_id",
                "number_of_listings",
                "median_effective_price",
                "median_predicted_price",
                "median_undervaluation_ratio",
                "median_rating",
                "most_common_neighbourhoods",
                "median_subway_distance",
                "median_crime_intensity",
            ]
        ).to_csv(OUTPUT_CLUSTER_SUMMARY, index=False)
        return

    candidates = candidates.dropna(subset=["latitude", "longitude"]).copy()
    coords_rad = np.radians(candidates[["latitude", "longitude"]].to_numpy())
    eps_rad = EPS_KM / EARTH_RADIUS_KM

    clusterer = DBSCAN(eps=eps_rad, min_samples=MIN_SAMPLES, metric="haversine")
    labels = clusterer.fit_predict(coords_rad)
    candidates["undervalued_cluster"] = labels

    cluster_rows = []
    for cluster_id, grp in candidates[candidates["undervalued_cluster"] != -1].groupby("undervalued_cluster"):
        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "number_of_listings": int(len(grp)),
                "median_effective_price": float(grp["effective_price"].median()),
                "median_predicted_price": float(grp["predicted_price"].median()),
                "median_undervaluation_ratio": float(grp["undervaluation_ratio"].median()),
                "median_rating": float(grp["review_scores_rating"].median()),
                "most_common_neighbourhoods": most_common_neighbourhoods(grp.get("neighbourhood_cleansed", pd.Series(dtype=str))),
                "median_subway_distance": float(grp.get("distance_to_nearest_subway_km", pd.Series(dtype=float)).median()),
                "median_crime_intensity": float(grp.get("crime_intensity_log_1000m", pd.Series(dtype=float)).median()),
            }
        )

    summary = pd.DataFrame(cluster_rows).sort_values("number_of_listings", ascending=False)
    candidates.to_csv(OUTPUT_CANDIDATES, index=False)
    summary.to_csv(OUTPUT_CLUSTER_SUMMARY, index=False)

    full = pd.read_csv(INPUT_TABLE, low_memory=False)
    full = full.merge(
        candidates[["id", "undervalued_cluster"]],
        on="id",
        how="left",
    )
    full.to_csv(OUTPUT_TABLE_UPDATED, index=False)
    build_data_dictionary(full).to_csv(OUTPUT_DATA_DICT, index=False)

    print(f"[SUCCESS] Saved clustered candidates: {OUTPUT_CANDIDATES}")
    print(f"[SUCCESS] Saved cluster summary: {OUTPUT_CLUSTER_SUMMARY}")
    print(f"[SUCCESS] Updated undervalued table with clusters: {OUTPUT_TABLE_UPDATED}")
    print(f"[SUCCESS] Updated data dictionary: {OUTPUT_DATA_DICT}")


if __name__ == "__main__":
    main()

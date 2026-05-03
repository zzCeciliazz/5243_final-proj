from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[0]
INPUT_TABLE = PROJECT_ROOT / "data" / "processed" / "nyc_airbnb_hidden_gem_model_table.csv"
OUTPUT_TABLE = PROJECT_ROOT / "data" / "processed" / "nyc_airbnb_undervalued_model_table.csv"
OUTPUT_MODEL_COMPARE = PROJECT_ROOT / "outputs" / "model_comparison.csv"
OUTPUT_CANDIDATES = PROJECT_ROOT / "outputs" / "undervalued_candidates.csv"
OUTPUT_DATA_DICT = PROJECT_ROOT / "data" / "processed" / "data_dictionary_prepared.csv"

TARGET = "log_effective_price"
RESIDUAL_PERCENTILE = 90  # configurable: 90 or 95 are common
TEST_SIZE = 0.2
RANDOM_STATE = 42
FAST_MODE = True
CV_FOLDS = 5

LEAKAGE_EXCLUDE_COLUMNS = {
    "price",
    "effective_price",
    "log_effective_price",
    "calendar_median_price",
    "calendar_avg_price",
    "calendar_min_price",
    "calendar_max_price",
    "estimated_revenue_l365d",
}

PREFERRED_PREDICTORS = [
    "room_type",
    "property_type",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "amenity_count",
    "minimum_nights",
    "maximum_nights",
    "neighbourhood_cleansed",
    "neighbourhood_group_cleansed",
    "latitude",
    "longitude",
    "distance_to_nearest_subway_km",
    "subway_stations_within_500m",
    "subway_stations_within_1000m",
    "crime_intensity_log_1000m",
    "violent_crime_count_1000m",
    "property_crime_count_1000m",
    "host_is_superhost",
    "host_response_rate",
    "host_acceptance_rate",
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "review_scores_value",
    "number_of_reviews",
    "reviews_in_window",
    "calendar_available_rate",
    "calendar_price_volatility",
    "weekend_price_premium",
]


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


def pick_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    exclude_cols = {
        TARGET,
        "effective_price",
        "hidden_gem_label",
        "overpriced_trap_label",
        "consumer_value_class",
        "predicted_log_price",
        "predicted_price",
        "price_residual_log",
        "price_gap",
        "undervaluation_ratio",
        "undervalued_candidate",
        "undervalued_cluster",
    }
    exclude_cols = exclude_cols.union(LEAKAGE_EXCLUDE_COLUMNS)
    allow_cols = [c for c in PREFERRED_PREDICTORS if c in df.columns and c not in exclude_cols]
    numeric_cols = [c for c in allow_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in allow_cols if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def main() -> None:
    if not INPUT_TABLE.exists():
        raise FileNotFoundError(f"Missing input table: {INPUT_TABLE}. Run src/02_prepare_data.py first.")

    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MODEL_COMPARE.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_TABLE, low_memory=False)
    if TARGET not in df.columns:
        raise ValueError(
            f"Missing required target column `{TARGET}`. "
            "Please ensure src/02_prepare_data.py has been rerun with log_effective_price."
        )
    df = df[df["effective_price"].notna() & (df["effective_price"] > 0)].copy()

    numeric_cols, categorical_cols = pick_feature_columns(df)
    features = numeric_cols + categorical_cols
    if not features:
        raise ValueError("No feature columns available for model training.")

    X = df[features]
    y = df[TARGET].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    if FAST_MODE:
        model_specs = [
            ("Ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
            (
                "RandomForest",
                RandomForestRegressor(
                    n_estimators=80,
                    max_depth=18,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
            ("GradientBoosting", GradientBoostingRegressor(random_state=RANDOM_STATE)),
        ]
    else:
        model_specs = [
            ("Ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
            ("RandomForest", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
            ("GradientBoosting", GradientBoostingRegressor(random_state=RANDOM_STATE)),
        ]

    comparison_rows = []
    trained_models = {}
    for name, estimator in model_specs:
        print(f"[INFO] Training model: {name}")
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics = evaluate_regression(y_test, pred)
        comparison_rows.append({"model": name, **metrics})
        trained_models[name] = pipe

    comparison_df = pd.DataFrame(comparison_rows).sort_values("RMSE")
    comparison_df.to_csv(OUTPUT_MODEL_COMPARE, index=False)
    best_model_name = comparison_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    print(f"[INFO] Best model selected by RMSE: {best_model_name}")

    # Out-of-fold predictions avoid in-sample fitted prediction leakage.
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(f"[INFO] Generating out-of-fold predictions with CV={CV_FOLDS}...")
    oof_pred_log = cross_val_predict(best_model, X, y, cv=cv, method="predict")
    df["predicted_log_price"] = oof_pred_log
    df["predicted_price"] = np.expm1(df["predicted_log_price"])
    df["price_residual_log"] = df["predicted_log_price"] - df["log_effective_price"]
    df["price_gap"] = df["predicted_price"] - df["effective_price"]
    df["undervaluation_ratio"] = np.where(
        df["effective_price"] > 0,
        df["predicted_price"] / df["effective_price"],
        np.nan,
    )

    residual_threshold = np.nanpercentile(df["price_residual_log"], RESIDUAL_PERCENTILE)
    has_reviews = (df["number_of_reviews"].fillna(0) >= 5) | (df["reviews_in_window"].fillna(0) >= 2)
    df["undervalued_candidate"] = (
        (df["price_residual_log"] >= residual_threshold)
        & (df["review_scores_rating"].fillna(-1) >= 4.8)
        & has_reviews
        & (df["effective_price"] > 0)
    ).astype(int)

    df.to_csv(OUTPUT_TABLE, index=False)
    df[df["undervalued_candidate"] == 1].to_csv(OUTPUT_CANDIDATES, index=False)
    build_data_dictionary(df).to_csv(OUTPUT_DATA_DICT, index=False)

    print(f"[SUCCESS] Saved model comparison: {OUTPUT_MODEL_COMPARE}")
    print(f"[SUCCESS] Saved undervalued model table: {OUTPUT_TABLE}")
    print(f"[SUCCESS] Saved undervalued candidates: {OUTPUT_CANDIDATES}")
    print(f"[SUCCESS] Updated data dictionary: {OUTPUT_DATA_DICT}")
    print(f"[INFO] Residual percentile threshold used: p{RESIDUAL_PERCENTILE} = {residual_threshold:.6f}")
    print(f"[INFO] FAST_MODE={FAST_MODE}")
    print(f"[INFO] CV_FOLDS={CV_FOLDS}")
    print(f"[INFO] Predictor count used: {len(features)}")


if __name__ == "__main__":
    main()

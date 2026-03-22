from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS = [
    "unit_cost",
    "lead_time_days",
    "capacity",
    "reliability_score",
    "historical_mean_delay",
    "historical_delay_std",
    "distance_km",
    "transportation_mode",
    "inventory_level",
    "demand_forecast",
    "order_size",
    "seasonality_index",
    "weather_risk",
    "geopolitical_risk",
    "bottleneck_risk",
    "fixed_penalty",
    "supplier_id",
    "component",
]


@dataclass
class ModelArtifacts:
    pipeline: Pipeline
    metrics: Dict[str, float]


def _build_pipeline() -> Pipeline:
    categorical = ["transportation_mode", "supplier_id", "component"]
    numeric = [col for col in FEATURE_COLUMNS if col not in categorical]

    # Encode categorical fields and pass numeric features unchanged.
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ]
    )

    regressor = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def train_delay_model(df: pd.DataFrame, target_col: str = "delay_days") -> ModelArtifacts:
    X = df[FEATURE_COLUMNS]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    # Clip to non-negative delays because negative delay days are not meaningful here.
    predictions = np.clip(pipeline.predict(X_test), 0, None)
    residuals = y_test.to_numpy(dtype=float) - predictions
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        # Residual spread is reused to sample uncertainty scenarios later.
        "residual_std": float(np.std(residuals)),
    }

    return ModelArtifacts(pipeline=pipeline, metrics=metrics)


def predict_delays(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    delays = model.predict(df[FEATURE_COLUMNS])
    return np.clip(delays, 0, None)

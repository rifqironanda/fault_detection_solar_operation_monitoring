"""Unsupervised anomaly detection for inverter operations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


FEATURE_COLUMNS = [
    "irradiance_w_m2",
    "ambient_temp_c",
    "power_kw",
    "voltage_v",
    "current_a",
    "expected_power_kw",
    "performance_ratio",
    "power_consistency_ratio",
]


def detect_anomalies(
    data: pd.DataFrame,
    contamination: float,
    random_seed: int,
) -> pd.DataFrame:
    """Detect unusual daytime operating conditions per inverter.

    Night/low-irradiance observations are excluded from model training
    and are not labelled as ML anomalies.
    """
    result = data.copy()
    result["is_anomaly"] = False
    result["anomaly_score"] = np.nan

    for _, group in result.groupby("inverter_id", sort=False):
        if "is_daylight" in group.columns:
            operating = group[group["is_daylight"]].copy()
        else:
            operating = group[
                group["irradiance_w_m2"] >= 200.0
            ].copy()

        if len(operating) < 10:
            continue

        features = (
            operating[FEATURE_COLUMNS]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        model = IsolationForest(
            contamination=contamination,
            random_state=random_seed,
            n_estimators=100,
        )

        prediction = model.fit_predict(features)
        anomaly_score = -model.score_samples(features)

        result.loc[
            operating.index,
            "is_anomaly",
        ] = prediction == -1

        result.loc[
            operating.index,
            "anomaly_score",
        ] = anomaly_score

    return result

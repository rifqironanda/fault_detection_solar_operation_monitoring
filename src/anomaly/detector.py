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
    """Detect unusual operating conditions independently per inverter."""

    result = data.copy()

    # Pastikan output memiliki kolom anomaly.
    result["is_anomaly"] = False
    result["anomaly_score"] = np.nan

    # ---------------------------------------------------------
    # Train one Isolation Forest for each inverter
    # ---------------------------------------------------------
    for inverter_id, group in result.groupby(
        "inverter_id",
        sort=False,
    ):

        # Ambil feature yang digunakan model.
        features = (
            group[FEATURE_COLUMNS]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )

        # Jangan melatih model jika data terlalu sedikit.
        if len(features) < 10:
            continue

        model = IsolationForest(
            contamination=contamination,
            random_state=random_seed,
            n_estimators=200,
        )

        prediction = model.fit_predict(features)

        anomaly_score = -model.score_samples(features)

        # Gunakan index asli group sehingga hasil kembali
        # ke inverter dan timestamp yang tepat.
        result.loc[
            group.index,
            "is_anomaly",
        ] = prediction == -1

        result.loc[
            group.index,
            "anomaly_score",
        ] = anomaly_score

    return result
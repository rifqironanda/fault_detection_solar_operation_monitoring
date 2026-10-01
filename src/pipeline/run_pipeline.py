"""End-to-end solar monitoring pipeline."""

from __future__ import annotations

import pandas as pd

from src.alerts.engine import generate_alerts
from src.anomaly.detector import detect_anomalies
from src.config.settings import DEFAULT_DATA_PATH, MonitoringConfig
from src.data.loader import load_pv_data, prepare_pv_data
from src.physics.baseline import add_performance_features


def calculate_summary_metrics(
    data: pd.DataFrame,
    alerts: pd.DataFrame,
) -> dict[str, float | int]:
    """Calculate dashboard-level monitoring metrics."""
    operational_alerts = int(
        data["alert_level"].isin(["WARNING", "CRITICAL"]).sum()
    )

    duration_days = max(
        (
            data["timestamp"].max()
            - data["timestamp"].min()
        ).total_seconds()
        / 86400,
        1 / 24,
    )

    valid_pr = data["performance_ratio"].dropna()

    return {
        "records": int(len(data)),
        "inverters": int(data["inverter_id"].nunique()),
        "ml_anomalies": int(data["is_anomaly"].sum()),
        "total_alerts": int(len(alerts)),
        "operational_alerts": operational_alerts,
        "alerts_per_day": round(
            operational_alerts / duration_days,
            3,
        ),
        "mean_performance_ratio": round(
            float(valid_pr.mean()) if not valid_pr.empty else 0.0,
            4,
        ),
    }


def run_monitoring_pipeline(
    data: pd.DataFrame | None = None,
    data_path=DEFAULT_DATA_PATH,
    config: MonitoringConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run validation, physics, ML, alerting, and metrics."""
    config = config or MonitoringConfig()

    if data is None:
        data = load_pv_data(data_path)
    else:
        data = prepare_pv_data(data)

    data = add_performance_features(data, config)

    data = detect_anomalies(
        data,
        contamination=config.contamination,
        random_seed=config.random_seed,
    )

    data, alerts = generate_alerts(data, config)
    metrics = calculate_summary_metrics(data, alerts)

    return data, alerts, metrics

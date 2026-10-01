import pandas as pd

from src.alerts.engine import generate_alerts
from src.config.settings import MonitoringConfig


def test_underperformance_creates_warning():
    config = MonitoringConfig(
        underperformance_ratio=0.80,
        minimum_irradiance_w_m2=200.0,
    )

    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01 12:00"]
            ),
            "inverter_id": ["INV-01"],
            "irradiance_w_m2": [1000.0],
            "ambient_temp_c": [25.0],
            "power_kw": [50.0],
            "expected_power_kw": [100.0],
            "performance_ratio": [0.50],
            "is_anomaly": [False],
            "anomaly_score": [0.1],
        }
    )

    enriched, alerts = generate_alerts(
        data,
        config,
    )

    assert enriched.loc[
        0,
        "alert_level",
    ] == "WARNING"

    assert enriched.loc[
        0,
        "alert_reason",
    ] == "Potential underperformance"

    assert len(alerts) == 1

    assert alerts.loc[
        0,
        "inverter_id",
    ] == "INV-01"


def test_normal_operation_creates_no_alert():
    config = MonitoringConfig(
        underperformance_ratio=0.80,
        minimum_irradiance_w_m2=200.0,
    )

    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01 12:00"]
            ),
            "inverter_id": ["INV-01"],
            "irradiance_w_m2": [1000.0],
            "ambient_temp_c": [25.0],
            "power_kw": [95.0],
            "expected_power_kw": [100.0],
            "performance_ratio": [0.95],
            "is_anomaly": [False],
            "anomaly_score": [0.1],
        }
    )

    enriched, alerts = generate_alerts(
        data,
        config,
    )

    assert enriched.loc[
        0,
        "alert_level",
    ] == "NORMAL"

    assert len(alerts) == 0
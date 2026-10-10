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

    assert "inverter_id" in alerts.columns


def test_scada_columns_keep_detection_and_dc_consistency():
    import numpy as np
    from src.pipeline.run_pipeline import run_monitoring_pipeline

    count = 40
    dc_power = np.linspace(80.0, 100.0, count)
    ac_power = dc_power * 0.95
    ac_power[-1] = 5.0
    raw = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01 10:00", periods=count, freq="min"),
        "inverter_id": ["INV-01"] * count,
        "irradiance_w_m2": np.linspace(800.0, 1000.0, count),
        "ambient_temp_c": [25.0] * count,
        "ac_power_kw": ac_power,
        "dc_power_kw": dc_power,
        "dc_voltage_v": [500.0] * count,
        "dc_current_a": dc_power * 2.0,
    })
    data, alerts, metrics = run_monitoring_pipeline(data=raw)
    assert len(data) == count
    assert data["power_kw"].equals(data["ac_power_kw"])
    assert np.allclose(data["power_consistency_ratio"], 1.0)
    assert data["anomaly_score"].notna().all()
    assert metrics["ml_anomalies"] > 0
    assert data.iloc[-1]["is_anomaly"]
    assert not alerts.empty

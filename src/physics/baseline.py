"""Physics-informed PV performance baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import MonitoringConfig


def calculate_expected_power(
    data: pd.DataFrame,
    config: MonitoringConfig,
) -> pd.Series:
    """Estimate expected PV output from irradiance and temperature."""
    irradiance = data["irradiance_w_m2"].clip(lower=0.0)

    # Below the daylight threshold, the expected PV output is treated as zero.
    daylight = irradiance >= config.minimum_irradiance_w_m2

    irradiance_ratio = (
        irradiance / config.reference_irradiance_w_m2
    ).clip(lower=0.0, upper=1.2)

    temperature_factor = (
        1.0
        + config.temperature_coefficient
        * (
            data["ambient_temp_c"]
            - config.reference_temperature_c
        )
    ).clip(lower=0.0)

    expected = (
        config.rated_power_kw
        * irradiance_ratio
        * temperature_factor
    )

    expected = expected.where(daylight, 0.0)

    return expected.clip(lower=0.0).rename("expected_power_kw")


def add_performance_features(
    data: pd.DataFrame,
    config: MonitoringConfig,
) -> pd.DataFrame:
    """Add expected power, daylight state, PR, and electrical features."""
    result = data.copy()

    result["expected_power_kw"] = calculate_expected_power(
        result,
        config,
    )

    result["is_daylight"] = (
        result["irradiance_w_m2"]
        >= config.minimum_irradiance_w_m2
    )

    denominator = result["expected_power_kw"].replace(
        0,
        np.nan,
    )

    result["performance_ratio"] = (
        result["power_kw"] / denominator
    ).where(
        result["is_daylight"],
        np.nan,
    ).clip(
        lower=0.0,
        upper=1.5,
    )

    result["electrical_power_estimate_kw"] = (
        result["voltage_v"]
        * result["current_a"]
        / 1000.0
    )

    electrical_denominator = (
        result["electrical_power_estimate_kw"]
        .replace(0, np.nan)
    )

    result["power_consistency_ratio"] = (
        result["power_kw"]
        / electrical_denominator
    )

    return result

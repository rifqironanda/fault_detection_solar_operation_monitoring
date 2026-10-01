"""Physics-informed PV performance baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import MonitoringConfig


def calculate_expected_power(
    data: pd.DataFrame,
    config: MonitoringConfig,
) -> pd.Series:
    """Estimate PV output from irradiance and temperature."""
    
    # 1. Irradiance factor
    irradiance_ratio = (
        data["irradiance_w_m2"]
        / config.reference_irradiance_w_m2
    )

    irradiance_ratio = np.clip(irradiance_ratio, 0.0, 1.2)

     # 2. Temperature factor
    temperature_factor = (
        1.0
        + config.temperature_coefficient
        * (data["ambient_temp_c"] - config.reference_temperature_c)
    )

    temperature_factor = np.clip(temperature_factor, 0.0, None)

    # 3. Expected power
    expected = (
        config.rated_power_kw
        * irradiance_ratio
        * temperature_factor
    )
    expected = np.clip(expected, 0.0, None)
    return pd.Series(expected, index=data.index, name="expected_power_kw")


def add_performance_features(
    data: pd.DataFrame,
    config: MonitoringConfig,
) -> pd.DataFrame:
    
    """Add expected power and Performance Ratio features."""
    
    result = data.copy()

     # 1. Physics-informed expected power
    result["expected_power_kw"] = calculate_expected_power(result, config)


    # 2. Performance Ratio
    valid_irradiance = result["irradiance_w_m2"] >= config.minimum_irradiance_w_m2
    denominator = result["expected_power_kw"].replace(0, np.nan)
    
    performance_ratio = np.where(
        valid_irradiance,
        result["power_kw"] / denominator,
        np.nan,
    )

    result["performance_ratio"] = np.clip(performance_ratio, 0.0, 1.5)

    
    # 3. electrical power estimate
    result["electrical_power_estimate_kw"] = (
        result["voltage_v"] 
        * result["current_a"] 
        / 1000.0
    )

    # 4. Power consistency ratio
    electrical_denominator = result["electrical_power_estimate_kw"].replace(0, np.nan)
    result["power_consistency_ratio"] = (
        result["power_kw"] 
        / electrical_denominator
    )

    return result

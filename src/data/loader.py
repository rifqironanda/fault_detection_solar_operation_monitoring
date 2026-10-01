"""Data loading and validation utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "timestamp",
    "inverter_id",
    "irradiance_w_m2",
    "ambient_temp_c",
    "power_kw",
    "voltage_v",
    "current_a",
}


def load_pv_data(path: Path) -> pd.DataFrame:
    """Load, validate, sort, and return PV measurements."""
    if not path.exists():
        raise FileNotFoundError(f"PV data file not found: {path}")

    data = pd.read_csv(path)

    """Validate and clean PV data."""
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

    data["inverter_id"] = (data["inverter_id"].astype(str).str.strip())

    for column in [
        "irradiance_w_m2",
        "ambient_temp_c",
        "power_kw",
        "voltage_v",
        "current_a",
    ]:data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(
        subset=[
            "timestamp", 
            "inverter_id", 
            "irradiance_w_m2", 
            "ambient_temp_c", 
            "power_kw", 
            "voltage_v", 
            "current_a"
        ]
    )

    data["irradiance_w_m2"] = data["irradiance_w_m2"].clip(lower=0)
    data["ambient_temp_c"] = data["ambient_temp_c"].clip(lower=-100, upper=100)
    data["power_kw"] = data["power_kw"].clip(lower=0)
    data["voltage_v"] = data["voltage_v"].clip(lower=0)
    data["current_a"] = data["current_a"].clip(lower=0)

    data = data.sort_values(["timestamp", "inverter_id"]).reset_index(drop=True)

    if data.empty:
        raise ValueError("No valid records remain after data validation.")

    """Diagnostics"""

    inverter_count = data["inverter_id"].nunique()
    duplicate_key = data.duplicated(subset=["timestamp", "inverter_id"]).sum()

    LOGGER.info("Loaded %d PV records from %s", len(data), path)
    LOGGER.info("Detected %d unique inverters(s)", inverter_count)

    if duplicate_key > 0:
        LOGGER.warning(
            "Detected %d duplicate timestamp-inverter_id combinations",
            duplicate_key,
        )
        
    return data

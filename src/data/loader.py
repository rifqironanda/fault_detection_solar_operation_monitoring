"""Data loading and validation utilities."""

from __future__ import annotations

import logging
from io import BytesIO
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

NUMERIC_COLUMNS = [
    "irradiance_w_m2",
    "ambient_temp_c",
    "power_kw",
    "voltage_v",
    "current_a",
]


def validate_pv_dataframe(data: pd.DataFrame) -> dict:
    """Validate a PV measurement DataFrame without running the pipeline."""
    missing = sorted(REQUIRED_COLUMNS.difference(data.columns))

    report = {
        "valid": not missing,
        "missing_columns": missing,
        "records": int(len(data)),
        "inverters": (
            int(data["inverter_id"].nunique())
            if "inverter_id" in data.columns
            else 0
        ),
        "invalid_timestamps": 0,
        "missing_values": 0,
        "duplicate_keys": 0,
        "time_start": None,
        "time_end": None,
    }

    if missing or data.empty:
        report["valid"] = False
        return report

    timestamps = pd.to_datetime(data["timestamp"], errors="coerce")
    report["invalid_timestamps"] = int(timestamps.isna().sum())

    missing_values = int(data[list(REQUIRED_COLUMNS)].isna().sum().sum())
    report["missing_values"] = missing_values

    report["duplicate_keys"] = int(
        data.duplicated(subset=["timestamp", "inverter_id"]).sum()
    )

    if timestamps.notna().any():
        report["time_start"] = timestamps.min()
        report["time_end"] = timestamps.max()

    report["valid"] = (
        not missing
        and not data.empty
        and report["invalid_timestamps"] == 0
        and report["missing_values"] == 0
    )

    return report


def prepare_pv_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate, clean, normalize, and sort PV measurements."""
    report = validate_pv_dataframe(data)

    if report["missing_columns"]:
        raise ValueError(
            f"Missing required columns: {report['missing_columns']}"
        )

    result = data.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    result["inverter_id"] = (
        result["inverter_id"].astype(str).str.strip()
    )

    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result = result.dropna(
        subset=["timestamp", "inverter_id", *NUMERIC_COLUMNS]
    )

    result["irradiance_w_m2"] = result[
        "irradiance_w_m2"
    ].clip(lower=0)

    result["ambient_temp_c"] = result[
        "ambient_temp_c"
    ].clip(lower=-100, upper=100)

    result["power_kw"] = result["power_kw"].clip(lower=0)
    result["voltage_v"] = result["voltage_v"].clip(lower=0)
    result["current_a"] = result["current_a"].clip(lower=0)

    result = result.sort_values(
        ["timestamp", "inverter_id"]
    ).reset_index(drop=True)

    if result.empty:
        raise ValueError("No valid records remain after data validation.")

    return result


def load_pv_data(path: Path) -> pd.DataFrame:
    """Load, validate, clean, and sort PV measurements from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"PV data file not found: {path}")

    data = pd.read_csv(path)
    result = prepare_pv_data(data)

    LOGGER.info("Loaded %d PV records from %s", len(result), path)
    LOGGER.info(
        "Detected %d unique inverter(s)",
        result["inverter_id"].nunique(),
    )

    duplicate_key = int(
        result.duplicated(
            subset=["timestamp", "inverter_id"]
        ).sum()
    )

    if duplicate_key > 0:
        LOGGER.warning(
            "Detected %d duplicate timestamp/inverter_id combinations",
            duplicate_key,
        )

    return result


def load_pv_upload(file) -> pd.DataFrame:
    """Load PV measurements from an uploaded Streamlit file."""
    raw = file.getvalue()
    data = pd.read_csv(BytesIO(raw))
    return prepare_pv_data(data)

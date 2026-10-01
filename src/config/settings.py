"""Central configuration for the Solar Operations Monitor."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "pv_monitoring_dataset.csv"


@dataclass(frozen=True)
class MonitoringConfig:
    """Immutable configuration shared by the monitoring pipeline."""

    rated_power_kw: float = 100.0
    reference_irradiance_w_m2: float = 1000.0
    reference_temperature_c: float = 25.0
    temperature_coefficient: float = -0.005
    minimum_irradiance_w_m2: float = 200.0
    underperformance_ratio: float = 0.80
    random_seed: int = 42
    contamination: float = 0.03

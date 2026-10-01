"""Explainable inverter-level alert generation."""

from __future__ import annotations

import pandas as pd

from src.config.settings import MonitoringConfig


def generate_alerts(
    data: pd.DataFrame,
    config: MonitoringConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate human-readable alerts from physics and ML evidence."""

    result = data.copy()

    # Optional feature.
    # Tidak semua test atau upstream data harus memiliki
    # power_consistency_ratio.
    if "power_consistency_ratio" not in result.columns:
        result["power_consistency_ratio"] = float("nan")

    alert_records = []

    levels = []
    reasons = []
    explanations = []

    for _, row in result.iterrows():

        level = "NORMAL"
        reason = "Normal"

        explanation = (
            "Measured inverter output is consistent "
            "with the available baseline."
        )

        inverter_id = row["inverter_id"]

        irradiance = float(row["irradiance_w_m2"])
        power = float(row["power_kw"])
        expected = float(row["expected_power_kw"])

        performance_ratio = row["performance_ratio"]
        is_anomaly = bool(row["is_anomaly"])

        power_consistency_ratio = row[
            "power_consistency_ratio"
        ]

        # -----------------------------------------------------
        # 1. Low irradiance
        # -----------------------------------------------------
        if irradiance < config.minimum_irradiance_w_m2:

            level = "INFO"
            reason = "Low irradiance"

            explanation = (
                f"Inverter {inverter_id} is operating under "
                "low irradiance conditions. Low power output "
                "may therefore be environmentally driven."
            )

        # -----------------------------------------------------
        # 2. Underperformance
        # -----------------------------------------------------
        elif power < (
            expected * config.underperformance_ratio
        ):

            level = (
                "CRITICAL"
                if is_anomaly
                else "WARNING"
            )

            reason = "Potential underperformance"

            explanation = (
                f"Inverter {inverter_id} power output is "
                "materially below the physics-informed "
                "expected output under the observed "
                "environmental conditions."
            )

            if is_anomaly:
                explanation += (
                    " The anomaly detector also identified "
                    "an unusual operating pattern."
                )

        # -----------------------------------------------------
        # 3. Electrical consistency
        # -----------------------------------------------------
        elif (
            pd.notna(power_consistency_ratio)
            and (
                power_consistency_ratio > 1.5
                or power_consistency_ratio < 0.5
            )
        ):

            level = "WARNING"
            reason = "Electrical consistency anomaly"

            explanation = (
                f"The measured power of inverter "
                f"{inverter_id} differs substantially "
                "from the V × I electrical power estimate."
            )

        # -----------------------------------------------------
        # 4. Statistical anomaly
        # -----------------------------------------------------
        elif is_anomaly:

            level = "WARNING"
            reason = "Statistical anomaly"

            explanation = (
                f"The operating feature combination of "
                f"inverter {inverter_id} differs from "
                "patterns learned by the unsupervised "
                "detector."
            )

        levels.append(level)
        reasons.append(reason)
        explanations.append(explanation)

        # -----------------------------------------------------
        # Alert record
        # -----------------------------------------------------
        if level != "NORMAL":

            alert_records.append(
                {
                    "alert_id": (
                        f"ALERT-{len(alert_records) + 1:05d}"
                    ),
                    "timestamp": row["timestamp"],
                    "inverter_id": inverter_id,
                    "severity": level,
                    "reason": reason,
                    "explanation": explanation,

                    "actual_power_kw": round(
                        power,
                        3,
                    ),

                    "expected_power_kw": round(
                        expected,
                        3,
                    ),

                    "performance_ratio": (
                        round(
                            float(performance_ratio),
                            3,
                        )
                        if pd.notna(performance_ratio)
                        else None
                    ),

                    "irradiance_w_m2": round(
                        irradiance,
                        2,
                    ),

                    "ambient_temp_c": round(
                        float(row["ambient_temp_c"]),
                        2,
                    ),

                    "voltage_v": (
                        round(
                            float(row["voltage_v"]),
                            2,
                        )
                        if "voltage_v" in row
                        else None
                    ),

                    "current_a": (
                        round(
                            float(row["current_a"]),
                            2,
                        )
                        if "current_a" in row
                        else None
                    ),

                    "electrical_power_estimate_kw": (
                        round(
                            float(
                                row[
                                    "electrical_power_estimate_kw"
                                ]
                            ),
                            3,
                        )
                        if "electrical_power_estimate_kw" in row
                        else None
                    ),

                    "power_consistency_ratio": (
                        round(
                            float(
                                power_consistency_ratio
                            ),
                            3,
                        )
                        if pd.notna(
                            power_consistency_ratio
                        )
                        else None
                    ),

                    "anomaly_score": (
                        round(
                            float(
                                row["anomaly_score"]
                            ),
                            3,
                        )
                        if pd.notna(
                            row["anomaly_score"]
                        )
                        else None
                    ),
                }
            )

    result["alert_level"] = levels
    result["alert_reason"] = reasons
    result["alert_explanation"] = explanations

    alerts = pd.DataFrame(alert_records)

    return result, alerts
"""Streamlit entry point for the Solar Operations Monitor."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.run_pipeline import run_monitoring_pipeline


st.set_page_config(
    page_title="Solar Operations Monitor",
    page_icon="☀️",
    layout="wide",
)


@st.cache_data
def load_monitoring_data() -> tuple:
    """Run the monitoring pipeline and cache its output."""

    return run_monitoring_pipeline()


def build_power_chart(data):
    """Create actual-versus-expected power chart."""

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["power_kw"],
            name="Actual Power",
            mode="lines+markers",
            line=dict(width=2),
            marker=dict(size=4),
            connectgaps=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["expected_power_kw"],
            name="Expected Power",
            mode="lines",
            line=dict(
                dash="dash",
                width=2,
            ),
            connectgaps=False,
        )
    )

    fig.update_layout(
        title="Actual vs Expected PV Power",
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        hovermode="x unified",
        height=430,
    )

    return fig


def main() -> None:
    """Render the monitoring dashboard."""

    st.title("Solar Operations Monitor")

    st.caption(
        "Inverter-level monitoring: physics-informed baseline + "
        "machine-learning anomaly detection + explainable alerts"
    )

    # =========================================================
    # Run pipeline
    # =========================================================

    try:
        data, alerts, metrics = load_monitoring_data()

    except Exception as exc:

        st.error(
            f"Pipeline failed: {exc}"
        )

        st.stop()

    # =========================================================
    # Inverter selector
    # =========================================================

    inverter_ids = sorted(
        data["inverter_id"]
        .dropna()
        .unique()
        .tolist()
    )

    if not inverter_ids:
        st.error(
            "No inverter data available."
        )
        st.stop()

    selected_inverter = st.selectbox(
        "Select inverter",
        inverter_ids,
    )

    inverter_data = data[
        data["inverter_id"]
        == selected_inverter
    ].copy()

    inverter_alerts = alerts[
        alerts["inverter_id"]
        == selected_inverter
    ].copy()

    # =========================================================
    # KPI
    # =========================================================

    total_alerts = len(
        inverter_alerts
    )

    anomaly_count = int(
        inverter_data["is_anomaly"].sum()
    )

    valid_pr = inverter_data[
        "performance_ratio"
    ].dropna()

    mean_pr = (
        float(valid_pr.mean())
        if not valid_pr.empty
        else 0.0
    )

    mean_power = float(
        inverter_data["power_kw"].mean()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Inverter",
        selected_inverter,
    )

    col2.metric(
        "Mean Power",
        f"{mean_power:.2f} kW",
    )

    col3.metric(
        "Alerts",
        f"{total_alerts:,}",
    )

    col4.metric(
        "Mean PR",
        f"{mean_pr:.1%}",
    )

    # =========================================================
    # Power chart
    # =========================================================

    st.plotly_chart(
        build_power_chart(
            inverter_data
        ),
        use_container_width=True,
    )

    # =========================================================
    # Environmental and electrical measurements
    # =========================================================

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Environmental Conditions"
        )

        environmental = inverter_data[
            [
                "timestamp",
                "irradiance_w_m2",
                "ambient_temp_c",
            ]
        ].copy()

        st.dataframe(
            environmental.tail(100),
            use_container_width=True,
            hide_index=True,
        )

    with right:

        st.subheader(
            "Electrical Measurements"
        )

        electrical = inverter_data[
            [
                "timestamp",
                "power_kw",
                "voltage_v",
                "current_a",
                "electrical_power_estimate_kw",
                "power_consistency_ratio",
            ]
        ].copy()

        electrical[
            "power_consistency_ratio"
        ] = electrical[
            "power_consistency_ratio"
        ].round(3)

        st.dataframe(
            electrical.tail(100),
            use_container_width=True,
            hide_index=True,
        )

    # =========================================================
    # Recent alerts
    # =========================================================

    st.subheader(
        f"Recent Alerts — {selected_inverter}"
    )

    if inverter_alerts.empty:

        st.success(
            "No alerts detected for this inverter."
        )

    else:

        st.dataframe(
            inverter_alerts.tail(25),
            use_container_width=True,
            hide_index=True,
        )

    # =========================================================
    # Alert explanation
    # =========================================================

    st.subheader(
        "Alert Explanation"
    )

    if inverter_alerts.empty:

        st.info(
            "No alert is available for explanation."
        )

    else:

        selected_id = st.selectbox(
            "Select alert",
            options=inverter_alerts[
                "alert_id"
            ].tolist(),
        )

        selected = inverter_alerts.loc[
            inverter_alerts["alert_id"]
            == selected_id
        ].iloc[0]

        st.write(
            f"**{selected['severity']} — "
            f"{selected['reason']}**"
        )

        st.write(
            selected["explanation"]
        )

        evidence = {
            "Inverter": selected[
                "inverter_id"
            ],
            "Actual power (kW)": selected[
                "actual_power_kw"
            ],
            "Expected power (kW)": selected[
                "expected_power_kw"
            ],
            "Performance ratio": selected[
                "performance_ratio"
            ],
            "Irradiance (W/m²)": selected[
                "irradiance_w_m2"
            ],
            "Temperature (°C)": selected[
                "ambient_temp_c"
            ],
            "Voltage (V)": selected[
                "voltage_v"
            ],
            "Current (A)": selected[
                "current_a"
            ],
            "Electrical power estimate (kW)": (
                selected[
                    "electrical_power_estimate_kw"
                ]
            ),
            "Power consistency ratio": selected[
                "power_consistency_ratio"
            ],
            "Anomaly score": selected[
                "anomaly_score"
            ],
        }

        st.json(evidence)

    # =========================================================
    # Dataset overview
    # =========================================================

    with st.expander(
        "Dataset / Pipeline Information"
    ):

        st.write(
            f"Total records: {len(data):,}"
        )

        st.write(
            f"Number of inverters: "
            f"{data['inverter_id'].nunique():,}"
        )

        st.write(
            f"Selected inverter: "
            f"{selected_inverter}"
        )

        st.json(metrics)

    # =========================================================
    # Prototype warning
    # =========================================================

    st.caption(
        "Prototype only. Physics coefficients, inverter ratings, "
        "thresholds, electrical consistency rules, and anomaly "
        "detection parameters require validation against actual "
        "plant and inverter specifications before operational use."
    )


if __name__ == "__main__":
    main()
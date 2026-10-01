"""Streamlit dashboard for PV operational monitoring."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MonitoringConfig
from src.data.loader import REQUIRED_COLUMNS, load_pv_upload
from src.pipeline.run_pipeline import run_monitoring_pipeline

st.set_page_config(
    page_title="Solar Operations Monitor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        .metric-card {
            padding: 0.8rem 1rem;
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 12px;
            background: rgba(128,128,128,0.06);
        }
        .section-note {
            font-size: 0.9rem;
            opacity: 0.78;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_power_chart(data: pd.DataFrame) -> go.Figure:
    """Create actual versus expected power chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["power_kw"],
            name="Actual Power",
            mode="lines",
            connectgaps=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["expected_power_kw"],
            name="Expected Power",
            mode="lines",
            line={"dash": "dash"},
            connectgaps=False,
        )
    )
    fig.update_layout(
        title="Actual vs Expected Power",
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        hovermode="x unified",
        height=430,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )
    return fig


def build_pr_chart(data: pd.DataFrame) -> go.Figure:
    """Create performance-ratio trend chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["performance_ratio"],
            name="Performance Ratio",
            mode="lines",
            connectgaps=False,
        )
    )
    fig.add_hline(y=0.80, line_dash="dash")
    fig.update_layout(
        title="Performance Ratio",
        xaxis_title="Time",
        yaxis_title="PR",
        yaxis={"range": [0, 1.2]},
        hovermode="x unified",
        height=320,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig


def build_status_table(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize monitoring condition per inverter."""
    records = []

    for inverter_id, group in data.groupby("inverter_id", sort=True):
        alerts = group["alert_level"]
        valid_pr = group["performance_ratio"].dropna()

        if (alerts == "CRITICAL").any():
            status = "CRITICAL"
        elif (alerts == "WARNING").any():
            status = "WARNING"
        elif (alerts == "INFO").any():
            status = "INFO"
        else:
            status = "NORMAL"

        records.append(
            {
                "Inverter": inverter_id,
                "Status": status,
                "Mean Power (kW)": round(float(group["power_kw"].mean()), 2),
                "Mean PR": (
                    round(float(valid_pr.mean()), 3)
                    if not valid_pr.empty
                    else None
                ),
                "ML Anomalies": int(group["is_anomaly"].sum()),
                "Alerts": int((alerts != "NORMAL").sum()),
            }
        )

    return pd.DataFrame(records)


def render_upload_panel() -> pd.DataFrame | None:
    """Render upload and validation workflow."""
    st.sidebar.header("Data Source")

    source = st.sidebar.radio(
        "Select source",
        ["Built-in dataset", "Upload CSV"],
        index=0,
    )

    if source == "Built-in dataset":
        st.sidebar.caption("Use the repository sample dataset.")
        return None

    uploaded = st.sidebar.file_uploader(
        "Upload PV monitoring CSV",
        type=["csv"],
        help=(
            "Required columns: "
            + ", ".join(sorted(REQUIRED_COLUMNS))
        ),
    )

    if uploaded is None:
        st.info(
            "Upload a CSV file to run monitoring on your own PV measurements."
        )
        st.caption(
            "Required fields: timestamp, inverter_id, irradiance_w_m2, "
            "ambient_temp_c, power_kw, voltage_v, current_a."
        )
        return None

    try:
        raw = pd.read_csv(io.BytesIO(uploaded.getvalue()))
    except Exception as exc:
        st.error(f"Unable to read CSV: {exc}")
        return None

    st.sidebar.markdown("### Validation")

    missing = sorted(REQUIRED_COLUMNS.difference(raw.columns))
    if missing:
        st.sidebar.error("Missing columns: " + ", ".join(missing))
        st.error("The uploaded dataset cannot be processed until the required columns are present.")
        with st.expander("Required schema"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "Column": sorted(REQUIRED_COLUMNS),
                        "Required": ["Yes"] * len(REQUIRED_COLUMNS),
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        return None

    timestamps = pd.to_datetime(raw["timestamp"], errors="coerce")
    missing_values = int(raw[list(REQUIRED_COLUMNS)].isna().sum().sum())
    duplicate_keys = int(
        raw.duplicated(subset=["timestamp", "inverter_id"]).sum()
    )

    st.sidebar.success("Required columns detected")
    st.sidebar.write(f"Records: {len(raw):,}")
    st.sidebar.write(
        f"Inverters: {raw['inverter_id'].nunique():,}"
    )
    st.sidebar.write(
        f"Invalid timestamps: {int(timestamps.isna().sum()):,}"
    )
    st.sidebar.write(f"Missing values: {missing_values:,}")
    st.sidebar.write(f"Duplicate timestamp/inverter: {duplicate_keys:,}")

    if timestamps.notna().any():
        st.sidebar.write(
            f"Period: {timestamps.min()} → {timestamps.max()}"
        )

    with st.expander("Uploaded Data Preview", expanded=False):
        st.dataframe(
            raw.head(50),
            use_container_width=True,
            hide_index=True,
        )

    return load_pv_upload(uploaded)


def render_kpis(data: pd.DataFrame, alerts: pd.DataFrame) -> None:
    """Render fleet-level KPI cards."""
    valid_pr = data["performance_ratio"].dropna()
    mean_pr = float(valid_pr.mean()) if not valid_pr.empty else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Inverters", f"{data['inverter_id'].nunique():,}")
    c2.metric("Records", f"{len(data):,}")
    c3.metric("Mean Power", f"{data['power_kw'].mean():.2f} kW")
    c4.metric("Mean PR", f"{mean_pr:.1%}")
    c5.metric(
        "Operational Alerts",
        f"{int(data['alert_level'].isin(['WARNING', 'CRITICAL']).sum()):,}",
    )


def main() -> None:
    inject_css()

    st.title("Solar Operations Monitor")
    st.caption(
        "Physics-informed performance monitoring, per-inverter anomaly detection, "
        "and explainable operational alerts."
    )

    uploaded_data = render_upload_panel()

    try:
        if uploaded_data is None:
            data, alerts, metrics = run_monitoring_pipeline(
                config=MonitoringConfig()
            )
            source_label = "Repository dataset"
        else:
            data, alerts, metrics = run_monitoring_pipeline(
                data=uploaded_data,
                config=MonitoringConfig(),
            )
            source_label = "Uploaded dataset"
    except Exception as exc:
        st.error(f"Monitoring pipeline failed: {exc}")
        st.stop()

    st.caption(
        f"Source: {source_label} · "
        f"{data['timestamp'].min()} → {data['timestamp'].max()} · "
        f"{data['inverter_id'].nunique()} inverter(s)"
    )

    render_kpis(data, alerts)

    st.divider()

    st.subheader("Fleet Status")
    status = build_status_table(data)

    if status.empty:
        st.info("No inverter records available.")
        return

    selected_inverter = st.selectbox(
        "Select inverter for detailed analysis",
        status["Inverter"].tolist(),
    )

    st.dataframe(
        status,
        hide_index=True,
        use_container_width=True,
    )

    inverter_data = data[
        data["inverter_id"] == selected_inverter
    ].copy()

    inverter_alerts = alerts[
        alerts["inverter_id"] == selected_inverter
    ].copy()

    st.subheader(f"Inverter Detail · {selected_inverter}")

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(
            build_power_chart(inverter_data),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            build_pr_chart(inverter_data),
            use_container_width=True,
        )

    e1, e2 = st.columns(2)

    with e1:
        st.markdown("#### Environment")
        environment = inverter_data[
            ["timestamp", "irradiance_w_m2", "ambient_temp_c"]
        ].copy()
        st.dataframe(
            environment.tail(100),
            hide_index=True,
            use_container_width=True,
        )

    with e2:
        st.markdown("#### Electrical")
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
        st.dataframe(
            electrical.tail(100),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Alert Evidence")

    if inverter_alerts.empty:
        st.success("No operational alerts detected for this inverter.")
    else:
        st.dataframe(
            inverter_alerts.tail(25),
            hide_index=True,
            use_container_width=True,
        )

        selected_id = st.selectbox(
            "Select alert",
            inverter_alerts["alert_id"].tolist(),
        )

        selected = inverter_alerts.loc[
            inverter_alerts["alert_id"] == selected_id
        ].iloc[0]

        st.markdown(
            f"**{selected['severity']} · {selected['reason']}**"
        )
        st.write(selected["explanation"])

        evidence = {
            "Timestamp": selected["timestamp"],
            "Inverter": selected["inverter_id"],
            "Actual power (kW)": selected["actual_power_kw"],
            "Expected power (kW)": selected["expected_power_kw"],
            "Performance ratio": selected["performance_ratio"],
            "Irradiance (W/m²)": selected["irradiance_w_m2"],
            "Ambient temperature (°C)": selected["ambient_temp_c"],
            "Voltage (V)": selected["voltage_v"],
            "Current (A)": selected["current_a"],
            "Electrical estimate (kW)": selected[
                "electrical_power_estimate_kw"
            ],
            "Power consistency ratio": selected[
                "power_consistency_ratio"
            ],
            "Anomaly score": selected["anomaly_score"],
        }
        st.json(evidence)

    with st.expander("Pipeline Information"):
        st.json(metrics)

    st.caption(
        "Prototype for research and demonstration. Anomaly scores indicate "
        "statistical unusualness and do not independently establish physical "
        "component failure. Thresholds and model parameters require validation "
        "with plant-specific data."
    )


if __name__ == "__main__":
    main()

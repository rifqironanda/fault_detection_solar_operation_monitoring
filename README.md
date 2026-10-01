# Solar Operations Monitor

A scalable MVP for photovoltaic (PV) operational monitoring.

The application combines:
1. open source or real PV sensor dataset.
2. Data validation and preprocessing.
3. A simple physics-informed expected-power.
4. Performance Ratio (PR) calculation.
5. Isolation Forest anomaly detection.
6. Rule-based, explainable alerts.
7. A Streamlit dashboard.

The project is intentionally modular. The first version uses dummy data so the
pipeline can be validated without access to a production PV system. Real data
can later replace `data/raw/pv_dummy.csv` while preserving the downstream
pipeline contract.

## Python

Target runtime: **Python 3.12**

## Setup

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run:

```powershell
streamlit run app.py
```

Tests:

```powershell
python -m pytest
```

## Data replacement

Replace `data/raw/pv_dummy.csv` with a CSV containing:

```text
timestamp
irradiance_w_m2
ambient_temp_c
power_kw
```

Optional:

```text
site_id
voltage_v
current_a
```

If your real dataset uses different column names, change the mapping in
`src/data/loader.py`. The physics, anomaly, alert, and dashboard layers should
not need to change.

## Method

Expected power:

```text
P_expected =
    P_rated
    × (G / G_reference)
    × [1 + gamma × (T_cell - T_reference)]
```

Performance Ratio:

```text
PR = P_actual / P_expected
```

The MVP uses ambient temperature as a proxy for cell temperature. This is a
demonstration assumption and should be replaced with a validated thermal model
when actual engineering data are available.

Isolation Forest identifies statistical outliers. It is not a physical fault
diagnosis model.

## Evaluation

Use time-based train/validation/test periods for real data. Recommended
metrics:

- Precision
- Recall
- F1
- False alerts per day
- Detection latency
- Missed anomalies

The included dataset contains synthetic fault injections only for demonstration.

## Engineering conventions

The code follows common Python/US/Japanese software-engineering practices:
PEP 8 naming, type hints, docstrings, small single-purpose modules,
configuration separation, deterministic seeds, logging, explicit data
contracts, relative paths, and automated tests.

Comments focus on engineering decisions rather than restating code.

## Limitations

This is a portfolio/research prototype, not an operational control system.
Thresholds, physical coefficients, sensor-quality rules, and fault labels must
be validated against actual plant characteristics before operational use.

# Data contract

Required logical fields:

| Field | Type | Unit | Required | Description |
|---|---|---|---|---|
| timestamp | datetime | ISO 8601 preferred | Yes | Measurement time |
| irradiance_w_m2 | float | W/m² | Yes | PV irradiance |
| ambient_temp_c | float | °C | Yes | Ambient temperature |
| power_kw | float | kW | Yes | Measured PV output |
| site_id | string | - | No | Site identifier |
| voltage_v | float | V | No | Voltage |
| current_a | float | A | No | Current |

For real data, document timezone explicitly. Do not silently mix UTC and local
time.

Recommended future fields:
- module temperature
- inverter status
- availability
- curtailment status
- maintenance events
- weather conditions
- fault labels

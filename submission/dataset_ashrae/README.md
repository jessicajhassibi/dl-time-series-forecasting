---
license: other
tags:
- time-series
- forecasting
- multivariate-time-series
- energy
pretty_name: ASHRAE GEPIII Electricity (converted)
---

# ashrae_gepiii_electricity

Multivariate hourly electricity-load forecasting for ASHRAE GEPIII buildings.

This is **not** an upstream dataset card — it is generated from the raw Kaggle
[ASHRAE Great Energy Predictor III](https://www.kaggle.com/competitions/ashrae-energy-prediction)
files by `submission/convert_ashrae.py`, reshaped into the same on-disk format as the
course benchmark (`submission/dataset`) so the same training/prediction pipeline runs on both.

## Setup
Perform setup steps under `SETUP.md`.

## Target

Predict the future hourly electricity meter reading (kWh) for each series_id.
One series = one building's electricity meter (`meter == 0`) over the year 2016.

## Forecast Contract

- Frequency: `h`
- Series: `200`
- Timesteps per series: `8784` (all hours of leap-year 2016)
- Target column: `target`
- Training history length used by the baseline templates: `168`
- Rollout block length: `24`
- Required prediction horizon: `validation: 336`
- Primary metric: `WAPE`; lower is better.

There is **no private test split** (`test_horizon: 0`) — we own the labels here and score
ourselves against `validation_target.csv`.

## Files

- `train.csv`: rows `0 .. 8447` per series, features + targets.
- `validation_input.csv`: last 336 steps per series, covariates only (parity with the benchmark).
- `forecast_index_validation.csv`: exact validation timestamps to predict.
- `validation_target.csv`: validation ground truth — **our own scoring labels**, no analogue
  in the benchmark dataset (there the labels stay on the leaderboard server).
- `metadata.json`: machine-readable metadata mirrored in this card.
- `raw/`: the three untouched Kaggle downloads (`train.csv`, `building_metadata.csv`,
  `weather_train.csv`). See `SETUP.md` for how to fetch them.

## Schema

- `train`: `series_id`, `timestamp`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos`, `is_weekend`, `air_temperature`, `dew_temperature`, `sea_level_pressure`, `wind_speed`, `wind_dir_sin`, `wind_dir_cos`, `log_square_feet`, `year_built`, `floor_count`, `target`
- `prediction`: `series_id`, `timestamp`, `prediction`
- `labels`: `series_id`, `timestamp`, `target`

Feature groups (17 total):

| Group | Columns | Notes |
| --- | --- | --- |
| Calendar | `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`, `is_weekend` | cyclic encodings, not scaled |
| Weather (per site) | `air_temperature`, `dew_temperature`, `sea_level_pressure`, `wind_speed` | interpolated onto the hourly grid |
| Weather (direction) | `wind_dir_sin/cos` | cyclic, not scaled |
| Static building | `log_square_feet`, `year_built`, `floor_count` | broadcast across time |

`cloud_coverage` and `precip_depth_1_hr` are dropped (~40–50% missing).

## Construction decisions

Documented here because they are ours, not the benchmark's:

- **Series selection.** Only buildings whose electricity meter has all 8784 hourly rows are
  kept, so the held-out horizon contains no interpolated (fake) ground truth. The first 200
  such buildings by id are taken — sites 0, 1, 2, 3, 5; mostly Education (65), Office (48)
  and Lodging/residential (38).
- **Target.** Raw `meter_reading`, unscaled — the models apply RevIN, so per-window scale is
  handled inside the model. Range on the validation horizon: 0 – 2477 kWh, median 85.
- **Feature scaling.** Every non-cyclic feature column is z-scored using **train rows only**
  (no validation leak). This is done here because the pipeline itself does not scale features,
  and raw ASHRAE scales (pressure ≈ 1000 hPa) would otherwise dominate the linear/TCN inputs.
- **Split.** Last 336 steps of each series (2016-12-18 00:00 → 2016-12-31 23:00) held out for
  validation, mirroring the benchmark's horizon.

## Reproduce

```bash
# raw download: see SETUP.md section 3 (Kaggle auth + accept competition rules)
.venv/bin/python submission/convert_ashrae.py --n-series 200
```


## Notes

- `submission/dataset`, `submission/dataset_ashrae`, `submission/logs`,
  `submission/predictions`, and `.venv` are gitignored — each person regenerates them locally.
- Benchmark metric is **WAPE**; forecast horizon 336h per series (24h rollout blocks).


## Links

- Competition: https://www.kaggle.com/competitions/ashrae-energy-prediction
- Converter: `submission/convert_ashrae.py`
- Benchmark counterpart: `submission/dataset/README.md`
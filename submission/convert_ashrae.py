"""Convert raw ASHRAE GEPIII files into the pipeline's on-disk dataset format.

Raw inputs (downloaded into <raw_dir>, see SETUP.md):
  - train.csv             building_id, meter, timestamp, meter_reading   (hourly, year 2016)
  - building_metadata.csv site_id, building_id, primary_use, square_feet, year_built, floor_count
  - weather_train.csv     site_id, timestamp, air_temperature, dew_temperature, ... (per site)

Modelling decisions (documented for the report):
  - Series      = one electricity meter (meter==0) per building -> series_id "building_XXXX".
                  We keep only full-length series (all 8784 hourly steps of 2016) so the held-out
                  target has no interpolated (fake) ground truth, AND drop series containing a
                  zero-run longer than `max_zero_run` hours. The zero filter matters: site 0 (and
                  part of site 5) recorded a literal 0.0 for Jan-Apr 2016 instead of no reading, so
                  the rows exist and pass the length check while the targets are fake. Isolated
                  short zeros are ordinary dropped readings and are kept. 24h keeps 250 of the 423
                  full-length series; tightening to 12h would leave only 181, and a day-long zero
                  can plausibly be a real closure whereas a week-long one cannot.
  - Target      = meter_reading (raw; the models apply RevIN, so per-window scale is handled).
  - Features    = calendar (hour/dow/month cyclic + is_weekend + is_holiday), site weather
                  (air/dew temp, pressure, wind speed + wind-direction cyclic), and static
                  building attributes (log square feet, year built, floor count). Sparse
                  weather cols (cloud_coverage, precip_depth_1_hr) are dropped.
  - is_holiday  = US federal holidays. Buildings shut down on holidays (median load drops to
                  0.67 of surrounding weeks on Memorial Day, 0.71 on July 4th, 0.76 on Labor
                  Day, 0.59 over Dec 24-26), but nothing else in the feature set identifies
                  those days -- without the flag a holiday is just a weekday that behaves like
                  a Sunday for no visible reason. The three summer/autumn holidays fall inside
                  the training range, so the pattern is learnable and transfers to the Christmas
                  dip in the validation window. Note the calendar is US-only while a few ASHRAE
                  sites are not (site 1 is generally believed to be in the UK); per-site
                  calendars would be a refinement.
                  A wider "winter break" flag was deliberately NOT added: it would be 0 across
                  all of training and 1 across validation, so its weight could never be learned.
  - Scaling     = every feature column is z-scored using TRAIN-rows statistics only (no val leak);
                  the pipeline itself does not scale features, and ASHRAE raw scales (e.g. pressure
                  ~1000 hPa) would otherwise dominate the linear/TCN inputs.
  - Split       = last `validation_horizon` (default 336h) of each series is held out for
                  validation, mirroring the benchmark. No separate private test split
                  (test_horizon = 0); we own the ASHRAE labels and score ourselves.

Outputs (into <out_dir>, default submission/dataset_ashrae):
  - train.csv                     training portion, with features + target
  - validation_input.csv          val horizon, features only (parity with benchmark; unused by predict)
  - forecast_index_validation.csv val horizon, series_id + timestamp
  - validation_target.csv         val horizon, series_id + timestamp + target (OUR ground truth for WAPE)
  - metadata.json                 schema/metadata matching Schema.from_metadata

Run:  python convert_ashrae.py --n-series 250   (250 = the full clean pool; see select_series)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from pandas.tseries.holiday import USFederalHolidayCalendar

YEAR_START = "2016-01-01 00:00:00"
YEAR_END = "2016-12-31 23:00:00"
FULL_LEN = 8784  # hourly steps in leap-year 2016

# Weather columns we keep (dropping cloud_coverage / precip_depth_1_hr: ~40-50% missing).
WEATHER_NUM = ["air_temperature", "dew_temperature", "sea_level_pressure", "wind_speed"]

# Feature columns that are already bounded/cyclic and must NOT be z-scored.
NO_SCALE = {"hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
            "wind_dir_sin", "wind_dir_cos", "is_weekend", "is_holiday"}


def _cyclic(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    radians = 2.0 * np.pi * values / period
    return np.sin(radians), np.cos(radians)


# Federal holidays on which these buildings actually shut down.
SHUTDOWN_HOLIDAYS = {"New Year's Day", "Memorial Day", "Independence Day", "Labor Day",
                     "Christmas Day"}

# Fixed-date holidays whose *observed* date shifts when they fall on a weekend. Load follows
# the real date (Christmas 2016 was a Sunday: Dec 25 dips to 0.56, the observed Dec 26 to 0.63),
# so both dates are flagged.
FIXED_HOLIDAY_DATES = [(1, 1), (7, 4), (12, 25)]


def build_holiday_flags(full_index: pd.DatetimeIndex) -> np.ndarray:
    """1.0 for every hour falling on a shutdown holiday, else 0.0. See module docstring."""
    calendar = USFederalHolidayCalendar()
    calendar.rules = [rule for rule in calendar.rules if rule.name in SHUTDOWN_HOLIDAYS]
    observed = calendar.holidays(start=full_index.min(), end=full_index.max())

    dates = full_index.normalize()
    flags = dates.isin(observed)
    for month, day in FIXED_HOLIDAY_DATES:
        flags = flags | ((dates.month == month) & (dates.day == day))
    return flags.astype(float)


def _max_zero_run(values: np.ndarray) -> int:
    """Length of the longest consecutive run of exact zeros in `values`."""
    is_zero = (values == 0).astype(np.int8)
    if not is_zero.any():
        return 0
    edges = np.diff(np.concatenate([[0], is_zero, [0]]))
    return int((np.where(edges == -1)[0] - np.where(edges == 1)[0]).max())


def build_weather(raw_dir: Path, sites: list[int], full_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return per-(site_id, timestamp) weather features, reindexed to a regular hourly grid
    and interpolated. wind_direction is turned into sin/cos and the raw column dropped."""
    weather = pd.read_csv(raw_dir / "weather_train.csv", parse_dates=["timestamp"])
    weather = weather[weather.site_id.isin(sites)]

    frames = []
    for site_id, group in weather.groupby("site_id"):
        group = (group.drop_duplicates("timestamp").set_index("timestamp")
                 .reindex(full_index))
        group["site_id"] = site_id
        # interpolate the numeric weather signals both directions
        cols = WEATHER_NUM + ["wind_direction"]
        group[cols] = group[cols].interpolate(limit_direction="both")
        frames.append(group.reset_index(names="timestamp"))
    out = pd.concat(frames, ignore_index=True)

    wd = out["wind_direction"].fillna(0.0).to_numpy()
    out["wind_dir_sin"], out["wind_dir_cos"] = _cyclic(wd, 360.0)
    out = out.drop(columns=["wind_direction"])
    # any residual gaps (a whole site missing a signal) -> 0 after interpolation
    out[WEATHER_NUM] = out[WEATHER_NUM].fillna(0.0)
    return out


def select_series(raw_dir: Path, meter: int, n_series: int, metadata: pd.DataFrame,
                  max_zero_run: int) -> list[int]:
    """Pick the first `n_series` buildings whose chosen-meter series is full length (8784h),
    has no zero-run longer than `max_zero_run` hours, and has building metadata + a site
    with weather. See the module docstring for why the zero filter is needed."""
    reader = pd.read_csv(raw_dir / "train.csv",
                         usecols=["building_id", "meter", "timestamp", "meter_reading"],
                         dtype={"building_id": "int16", "meter": "int8", "meter_reading": "float32"})
    reader = reader[reader.meter == meter]

    counts = reader.groupby("building_id").size()
    full = counts[counts == FULL_LEN].index

    # timestamps are ISO strings, so lexicographic sort == chronological (no parsing needed)
    reader = reader[reader.building_id.isin(full)].sort_values(["building_id", "timestamp"])
    longest_zero = reader.groupby("building_id").meter_reading.apply(
        lambda s: _max_zero_run(s.to_numpy()))
    clean = longest_zero[longest_zero <= max_zero_run].index

    valid = metadata[metadata.building_id.isin(clean)].sort_values("building_id")
    chosen = valid.building_id.head(n_series).tolist()
    print(f"Candidates for meter {meter}: {counts.size} buildings -> {len(full)} full-length "
          f"-> {len(clean)} with zero-runs <= {max_zero_run}h.")
    if len(chosen) < n_series:
        print(f"WARNING: only {len(chosen)} qualifying series available for meter {meter} "
              f"(requested {n_series}).")
    return chosen


def convert(raw_dir: Path = Path("dataset_ashrae/raw"),
            out_dir: Path = Path("dataset_ashrae"),
            meter: int = 0,
            n_series: int = 250,
            max_zero_run: int = 24,
            validation_horizon: int = 336,
            history_length: int = 168,
            forecast_horizon: int = 24) -> None:
    """Build the ASHRAE dataset in our pipeline's format. See module docstring for details."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    full_index = pd.date_range(YEAR_START, YEAR_END, freq="h")
    assert len(full_index) == FULL_LEN

    metadata = pd.read_csv(raw_dir / "building_metadata.csv")
    building_ids = select_series(raw_dir, meter, n_series, metadata, max_zero_run)
    meta_by_building = metadata.set_index("building_id")
    sites = sorted(meta_by_building.loc[building_ids, "site_id"].unique().tolist())
    print(f"Selected {len(building_ids)} series (meter={meter}) across {len(sites)} sites.")

    # --- meter readings (target) for the chosen buildings ---
    readings = pd.read_csv(raw_dir / "train.csv", parse_dates=["timestamp"],
                           dtype={"building_id": "int16", "meter": "int8"})
    readings = readings[(readings.meter == meter) & (readings.building_id.isin(building_ids))]
    readings = readings[["building_id", "timestamp", "meter_reading"]]

    weather = build_weather(raw_dir, sites, full_index)
    # date-only, so compute once and broadcast to every building
    holiday_flags = build_holiday_flags(full_index)

    # static building attributes, cleaned
    stat = meta_by_building.loc[building_ids].copy()
    stat["log_square_feet"] = np.log1p(stat["square_feet"].astype(float))
    stat["year_built"] = stat["year_built"].fillna(stat["year_built"].median())
    stat["floor_count"] = stat["floor_count"].fillna(stat["floor_count"].median())

    # --- assemble one long frame ---
    frames = []
    for building_id in building_ids:
        df = pd.DataFrame({"timestamp": full_index})
        r = readings[readings.building_id == building_id][["timestamp", "meter_reading"]]
        df = df.merge(r, on="timestamp", how="left")
        df["target"] = df["meter_reading"].astype(float)
        df = df.drop(columns=["meter_reading"])

        # calendar features
        ts = df["timestamp"].dt
        df["hour_sin"], df["hour_cos"] = _cyclic(ts.hour.to_numpy(), 24.0)
        df["dow_sin"], df["dow_cos"] = _cyclic(ts.dayofweek.to_numpy(), 7.0)
        df["month_sin"], df["month_cos"] = _cyclic(ts.month.to_numpy() - 1, 12.0)
        df["is_weekend"] = (ts.dayofweek >= 5).astype(float)
        df["is_holiday"] = holiday_flags

        # weather (join on site)
        site_id = int(meta_by_building.loc[building_id, "site_id"])
        w = weather[weather.site_id == site_id].drop(columns=["site_id"])
        df = df.merge(w, on="timestamp", how="left")

        # static (broadcast)
        df["log_square_feet"] = float(stat.loc[building_id, "log_square_feet"])
        df["year_built"] = float(stat.loc[building_id, "year_built"])
        df["floor_count"] = float(stat.loc[building_id, "floor_count"])

        df["series_id"] = f"building_{building_id:04d}"
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["series_id", "timestamp"]).reset_index(drop=True)

    feature_cols = (["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
                     "is_weekend", "is_holiday"] + WEATHER_NUM + ["wind_dir_sin", "wind_dir_cos",
                     "log_square_feet", "year_built", "floor_count"])
    ordered = ["series_id", "timestamp"] + feature_cols + ["target"]
    data = data[ordered]

    # --- train / validation split (last `validation_horizon` steps per series) ---
    n_steps = FULL_LEN
    split_at = n_steps - validation_horizon
    idx_in_series = data.groupby("series_id").cumcount()
    train_mask = idx_in_series < split_at

    # --- z-score features using TRAIN rows only ---
    scale_cols = [c for c in feature_cols if c not in NO_SCALE]
    means = data.loc[train_mask, scale_cols].mean()
    stds = data.loc[train_mask, scale_cols].std().replace(0.0, 1.0)
    data[scale_cols] = (data[scale_cols] - means) / stds

    train_df = data[train_mask].copy()
    val_df = data[~train_mask].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.drop(columns=["target"]).to_csv(out_dir / "validation_input.csv", index=False)
    val_df[["series_id", "timestamp"]].to_csv(out_dir / "forecast_index_validation.csv", index=False)
    val_df[["series_id", "timestamp", "target"]].to_csv(out_dir / "validation_target.csv", index=False)

    meta = {
        "name": "ashrae_gepiii_electricity",
        "task": "Multivariate hourly electricity-load forecasting for ASHRAE GEPIII buildings.",
        "target_column": "target",
        "target_description": "Hourly electricity meter reading (kWh) per building.",
        "frequency": "h",
        "n_series": len(building_ids),
        "n_steps": n_steps,
        "max_zero_run": max_zero_run,
        "history_length": history_length,
        "forecast_horizon": forecast_horizon,
        "required_prediction_horizon": {"validation": validation_horizon},
        "validation_horizon": validation_horizon,
        "test_horizon": 0,
        "metric": "wape",
        "schema": {
            "labels": ["series_id", "timestamp", "target"],
            "prediction": ["series_id", "timestamp", "prediction"],
            "train": ["series_id", "timestamp"] + feature_cols + ["target"],
        },
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote dataset to {out_dir}")
    print(f"  series={len(building_ids)}  features={len(feature_cols)}  "
          f"train_steps={split_at}  val_steps={validation_horizon}")
    print(f"  train.csv rows={len(train_df)}  val rows={len(val_df)}")


if __name__ == "__main__":
    tyro.cli(convert, use_underscores=True)

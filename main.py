from collections.abc import Hashable, Mapping

import numpy as np
import pandas as pd



# ==================== Load Data ====================

DATA_PATH = r"./datasets/household_power_consumption.txt"

data = pd.read_csv(DATA_PATH, sep=";", na_values=["?"])
data = data.convert_dtypes()

data["Datetime"] = pd.to_datetime(
    data["Date"] + " " + data["Time"], format="%d/%m/%Y %H:%M:%S"
)
data = data.set_index("Datetime")
data.drop(columns=["Date", "Time"], inplace=True)
assert data.index.is_monotonic_increasing

data_raw = data.copy()  # keep the pre-imputation version around for sanity-check plots


# ---------- Check Data
# data.info()
# print(data.head())

# missing_count = data.isna().sum()
# print(missing_count)


# ==================== Train/Test Split ====================

# Hold out the last six months as the test set; split before imputing missing values to avoid data leakage
split_date = data.index.max() - pd.DateOffset(months=6)

train_data = data[data.index <= split_date].copy()
test_data = data[data.index > split_date].copy()


# ==================== Handle Missing Values ====================

LONG_GAP_MINUTES = 60  # gaps longer than this get filled from the same time last week


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = df.columns.tolist()

    is_missing = df[value_cols[0]].isna()
    gap_id = (is_missing != is_missing.shift()).cumsum()
    gap_length = is_missing.groupby(gap_id).transform("sum")
    long_gap = is_missing & (gap_length > LONG_GAP_MINUTES)

    # Long outages: reuse the value from the same time on the previous week to preserve
    # the daily/weekly load shape instead of flattening it
    same_time_last_week = df.shift(freq="7D").reindex(df.index)
    df.loc[long_gap, value_cols] = same_time_last_week.loc[long_gap, value_cols]

    # Short gaps (and any long-gap rows with no data a week earlier within this split)
    # fall back to time-based linear interpolation
    df[value_cols] = df[value_cols].interpolate(method="time", limit_direction="both")

    assert df.isna().sum().sum() == 0
    return df


train_data = fill_missing(train_data)
test_data = fill_missing(test_data)

data = pd.concat([train_data, test_data])


# ==================== Resample to Hourly ====================

HOURLY_AGG: Mapping[Hashable, str] = {
    "Global_active_power": "mean",
    "Global_reactive_power": "mean",
    "Voltage": "mean",
    "Global_intensity": "mean",
    "Sub_metering_1": "sum",
    "Sub_metering_2": "sum",
    "Sub_metering_3": "sum",
}

# data_hourly = data.resample("h").agg(HOURLY_AGG)
# data_hourly["Peak_active_power"] = data["Global_active_power"].resample("h").max()
#
# print(data_hourly.head())

# ---------- Sanity-check visualization (before/after around a known outage)
# window = slice("2007-04-25", "2007-05-02")   # covers the big April 2007 gap
# fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
# data_raw["Global_active_power"][window].plot(ax=ax[0], title="Raw (with gap)")
# data["Global_active_power"][window].plot(ax=ax[1], title="Imputed")
# plt.tight_layout()
# plt.show()

train_data, test_data = (
    train_data.resample("h").agg(HOURLY_AGG),
    test_data.resample("h").agg(HOURLY_AGG),
)


# ==================== Feature Engineering ====================

# 7 days of history to predict the next 24 hours
LOOKBACK = 24 * 7
HORIZON = 24

TARGET_COL = "Global_active_power"
CALENDAR_COLS = [
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "month_sin",
    "month_cos",
]
FEATURE_COLS = [TARGET_COL, *CALENDAR_COLS]


# Adds calendar features to the DataFrame including sin and cos transformations for hour, day of week, and month
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(df.index, pd.DatetimeIndex)
    hour, dayofweek, month = df.index.hour, df.index.dayofweek, df.index.month
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    return df


train_data, test_data = (
    add_calendar_features(train_data),
    add_calendar_features(test_data),
)


# ---------- Feature Scaling
target_min = train_data[TARGET_COL].min()
target_max = train_data[TARGET_COL].max()


def min_max_scale(series: pd.Series) -> pd.Series:
    return (series - target_min) / (target_max - target_min)


train_data[TARGET_COL] = min_max_scale(train_data[TARGET_COL])
test_data[TARGET_COL] = min_max_scale(test_data[TARGET_COL])

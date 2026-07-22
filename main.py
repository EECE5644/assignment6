import pandas as pd


DATA_PATH = r"./datasets/household_power_consumption.txt"
LONG_GAP_MINUTES = 60  # gaps longer than this get filled from the same time last week


# ==================== Load Data ====================

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

data_hourly = data.resample("h").agg(
    {
        "Global_active_power": "mean",
        "Global_reactive_power": "mean",
        "Voltage": "mean",
        "Global_intensity": "mean",
        "Sub_metering_1": "sum",
        "Sub_metering_2": "sum",
        "Sub_metering_3": "sum",
    }
)
data_hourly["Peak_active_power"] = data["Global_active_power"].resample("h").max()

# print(hourly.head())

# ---------- Sanity-check visualization (before/after around a known outage)
# window = slice("2007-04-25", "2007-05-02")   # covers the big April 2007 gap
# fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
# data_raw["Global_active_power"][window].plot(ax=ax[0], title="Raw (with gap)")
# data["Global_active_power"][window].plot(ax=ax[1], title="Imputed")
# plt.tight_layout()
# plt.show()

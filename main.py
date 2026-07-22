from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import override

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


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


# ==================== Dataset Preparation ====================

# 7 days of history to predict the next 24 hours
LOOKBACK = 24 * 7
HORIZON = 24

TARGET_IDX = FEATURE_COLS.index(TARGET_COL)


def make_sequences(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(values) - LOOKBACK - HORIZON + 1):
        X.append(values[i : i + LOOKBACK])
        y.append(values[i + LOOKBACK : i + LOOKBACK + HORIZON, TARGET_IDX])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


X_train, y_train = make_sequences(train_data[FEATURE_COLS].to_numpy())
X_test, y_test = make_sequences(test_data[FEATURE_COLS].to_numpy())


class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self.features)

    @override
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


train_dataset = SequenceDataset(X_train, y_train)
test_dataset = SequenceDataset(X_test, y_test)


# ==================== Model ====================
class LSTMForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        horizon: int = HORIZON,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, horizon)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, c_n) = self.lstm(x)
        output = h_n[-1]  # (batch, hidden_size)
        output = self.drop(output)
        return self.fc(output)  # (batch, HORIZON)


# ==================== Training ====================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 5
LEARNING_RATE = 1e-3

criterion = nn.MSELoss()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    lr: float = LEARNING_RATE,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    patience_counter = 0
    best_loss, best_state = float("inf"), None

    model.to(DEVICE)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            preds = model(features)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(DEVICE), targets.to(DEVICE)
                preds = model(features)
                loss = criterion(preds, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)


CHECKPOINT_PATH = Path("checkpoints/lstm_forecaster.pt")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
model = LSTMForecaster(n_features=len(FEATURE_COLS)).to(DEVICE)

if CHECKPOINT_PATH.exists():
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
else:
    train_model(model, train_loader, test_loader)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)


# ==================== Evaluation ====================
def evaluate_model(model: nn.Module, dataloader: DataLoader) -> tuple[float, float]:
    total_loss = 0.0
    total_ape, n_sample = 0.0, 0

    model.eval()
    with torch.no_grad():
        for features, targets in dataloader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            preds = model(features)
            loss = criterion(preds, targets)
            total_loss += loss.item()

            preds_real = preds.cpu().numpy() * (target_max - target_min) + target_min
            targets_real = (
                targets.cpu().numpy() * (target_max - target_min) + target_min
            )
            total_ape += np.abs((targets_real - preds_real) / targets_real).sum()
            n_sample += targets_real.size

    total_loss /= len(dataloader)
    mape = total_ape / n_sample
    model.train()

    return total_loss, mape


train_loss, train_mape = evaluate_model(model, train_loader)
print(f"Train MSE (scaled): {train_loss:.5f}, Train MAPE: {train_mape:.2%}")
test_loss, test_mape = evaluate_model(model, test_loader)
print(f"Test MSE (scaled): {test_loss:.5f}, Test MAPE: {test_mape:.2%}")


# OUTPUT: [TODO]
# Train MSE (scaled): 0.01184, Train MAPE: 73.42%
# Test MSE (scaled): 0.00767, Test MAPE: 60.81%

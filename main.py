import random
import time
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import override

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm


# ==================== Reproducibility ====================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


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


# ---------- Identify missing values and outage structure
LONG_GAP_MINUTES = 60  # gaps longer than this get filled from the same time last week

# The dataset drops all sensors at once during a meter outage, so every column is
# missing on the same rows; checking one column tells us about the whole row.
_is_missing = data["Global_active_power"].isna()
_gap_id = (_is_missing != _is_missing.shift()).cumsum()
_gap_lengths = _is_missing.groupby(_gap_id).sum().astype(int)
_gap_lengths = _gap_lengths[_gap_lengths > 0]

print(f"Missing rows: {_is_missing.sum()} / {len(data)} ({_is_missing.mean():.2%})")
print(f"Number of separate outages: {len(_gap_lengths)}")
print(
    f"Outage length (minutes) - min:{_gap_lengths.min()}, median:{_gap_lengths.median():.0f},",
    f"max:{_gap_lengths.max()}(~{_gap_lengths.max() / 60:.1f}hours)",
)
print(
    f"Outages longer than {LONG_GAP_MINUTES} min: {(_gap_lengths > LONG_GAP_MINUTES).sum()}",
    f"out of {len(_gap_lengths)} - these are the multi-hour/multi-day outages that a flat",
    "fill or short-window interpolation would visibly distort",
)
print(f"\n{'=' * 80}\n")


# ==================== Train/Val/Test Split ====================

# Hold out the last six months as the test set, and 3 months before that as validation
test_split_date = data.index.max() - pd.DateOffset(months=6)
val_split_date = test_split_date - pd.DateOffset(months=3)

train_data = data[data.index <= val_split_date].copy()
val_data = data[(data.index > val_split_date) & (data.index <= test_split_date)].copy()
test_data = data[data.index > test_split_date].copy()


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
val_data = fill_missing(val_data)
test_data = fill_missing(test_data)

data = pd.concat([train_data, val_data, test_data])


# ==================== Resample to Hourly ====================

# Global_active_power/reactive_power/Voltage/Global_intensity are instantaneous
# rates (1-minute averages), so the hourly value that keeps the same meaning is
# the mean over that hour. Sub_metering_1/2/3 are energy already measured in
# Wh per minute - a "quantity", not a rate - so they should be summed to get the
# hour's total Wh, not averaged.
HOURLY_AGG: Mapping[Hashable, str] = {
    "Global_active_power": "mean",
    "Global_reactive_power": "mean",
    "Voltage": "mean",
    "Global_intensity": "mean",
    "Sub_metering_1": "sum",
    "Sub_metering_2": "sum",
    "Sub_metering_3": "sum",
}

data_hourly = data.resample("h").agg(HOURLY_AGG)
data_hourly["Peak_active_power"] = data["Global_active_power"].resample("h").max()

print(f"Resampled {len(data)} minute-level rows into {len(data_hourly)} hourly rows")
print(data_hourly.head())
print(f"\n{'=' * 80}\n")

train_data, val_data, test_data = (
    train_data.resample("h").agg(HOURLY_AGG),
    val_data.resample("h").agg(HOURLY_AGG),
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


train_data, val_data, test_data = (
    add_calendar_features(train_data),
    add_calendar_features(val_data),
    add_calendar_features(test_data),
)


# ---------- Feature Scaling
target_min = train_data[TARGET_COL].min()
target_max = train_data[TARGET_COL].max()


def min_max_scale(series: pd.Series) -> pd.Series:
    return (series - target_min) / (target_max - target_min)


train_data[TARGET_COL] = min_max_scale(train_data[TARGET_COL])
val_data[TARGET_COL] = min_max_scale(val_data[TARGET_COL])
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
X_val, y_val = make_sequences(val_data[FEATURE_COLS].to_numpy())
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
val_dataset = SequenceDataset(X_val, y_val)
test_dataset = SequenceDataset(X_test, y_test)


# ==================== Model ====================
class LSTMForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        horizon: int = HORIZON,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, horizon)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, c_n) = self.lstm(x)
        output = h_n[-1]  # last layer's final hidden state, (batch, hidden_size)
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
PATIENCE = 10
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
    progress = tqdm(range(epochs), desc="Training", unit="epoch")
    for epoch in progress:
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

        progress.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                progress.write(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)


CHECKPOINT_PATH = Path("checkpoints/lstm_forecaster.pt")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
model = LSTMForecaster(n_features=len(FEATURE_COLS)).to(DEVICE)

if CHECKPOINT_PATH.exists():
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
else:
    train_model(model, train_loader, val_loader)
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

    return total_loss, mape


train_loss, train_mape = evaluate_model(model, train_loader)
print(f"Train MSE (scaled): {train_loss:.5f}, Train MAPE: {train_mape:.2%}")
val_loss, val_mape = evaluate_model(model, val_loader)
print(f"Val MSE (scaled): {val_loss:.5f}, Val MAPE: {val_mape:.2%}")
test_loss, test_mape = evaluate_model(model, test_loader)
print(f"Test MSE (scaled): {test_loss:.5f}, Test MAPE: {test_mape:.2%}")
print(f"\n{'=' * 80}\n")


# ==================== Deployment Considerations ====================

# ---------- Inference Latency
LATENCY_BUDGET_MS = 500.0

model.eval()
with torch.no_grad():
    # Warm-up call so the timed runs don't include one-off kernel init overhead.
    _ = model(test_dataset.features[:1].to(DEVICE))

    start = time.perf_counter()
    _ = model(test_dataset.features[:1].to(DEVICE))
    single_latency_ms = (time.perf_counter() - start) * 1000

    batch_size = min(1000, len(test_dataset))
    batch = test_dataset.features[:batch_size].to(DEVICE)
    start = time.perf_counter()
    _ = model(batch)
    batch_latency_ms = (time.perf_counter() - start) * 1000

print(f"Single-household latency: {single_latency_ms:.2f} ms ")
print(
    f"Batch of {batch_size} households: {batch_latency_ms:.2f} ms total",
    f"{batch_latency_ms / batch_size:.3f} ms/household",
)
print(f"\n{'=' * 80}\n")


# ---------- Anomaly Alerting
# Flag a household when the 1-hour-ahead forecast error exceeds 3 std of the
# historical (validation-set) residuals for 2+ consecutive hours.
ANOMALY_STD_MULTIPLIER = 3.0
MIN_CONSECUTIVE_HOURS = 2


def one_step_residuals(model: nn.Module, dataloader: DataLoader) -> np.ndarray:
    # Each window's first predicted hour is the model's 1-hour-ahead forecast,
    # made 24 hours before the next window makes the same-hour forecast again.
    model.eval()
    residuals = []
    with torch.no_grad():
        for features, targets in dataloader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            preds = model(features)
            preds_real = (
                preds[:, 0].cpu().numpy() * (target_max - target_min) + target_min
            )
            targets_real = (
                targets[:, 0].cpu().numpy() * (target_max - target_min) + target_min
            )
            residuals.append(targets_real - preds_real)
    return np.concatenate(residuals)


val_residuals = one_step_residuals(model, val_loader)
residual_std = val_residuals.std()
alert_threshold = ANOMALY_STD_MULTIPLIER * residual_std

test_residuals = one_step_residuals(model, test_loader)
breach = np.abs(test_residuals) > alert_threshold

# True at hour i only once `breach` has held for MIN_CONSECUTIVE_HOURS in a row.
alert = np.zeros(len(breach), dtype=bool)
consecutive = 0
for i, is_breach in enumerate(breach):
    consecutive = consecutive + 1 if is_breach else 0
    if consecutive >= MIN_CONSECUTIVE_HOURS:
        alert[i] = True

n_alerts = int(alert.sum())
false_alarm_rate = n_alerts / len(test_residuals)

print(f"Residual std (val, 1h-ahead): {residual_std:.4f} kW")
print(f"Alert threshold (±{ANOMALY_STD_MULTIPLIER:.0f}σ): {alert_threshold:.4f} kW")
print(
    f"Test hours flagged ({MIN_CONSECUTIVE_HOURS}+ consecutive breaches):",
    f"{n_alerts} / {len(test_residuals)}",
    f"({false_alarm_rate:.2%})",
)

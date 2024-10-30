# %%
import numpy as np
import pandas as pd
import os
import re
from sklearn.base import clone
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from scipy import stats
from typing import Tuple

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import torch
import torch.nn as nn
import torch.optim as optim

from colorama import Fore, Style
from IPython.display import clear_output
import warnings
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    VotingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
pd.options.display.max_columns = None

SEED = 42
n_splits = 5


# %%
"""
# Actigraphy Files and Field Descriptions

During their participation in the HBN study, some participants were given an accelerometer to wear for up to 30 days continually while at home and going about their regular daily lives.

## File Structure

- **series_{train|test}.parquet/id={id}**  
  Series to be used as training data, partitioned by id. Each series is a continuous recording of accelerometer data for a single subject spanning many days.

## Field Descriptions

- **id**  
  The patient identifier corresponding to the id field in `train/test.csv`.

- **step**  
  An integer timestep for each observation within a series.

- **X, Y, Z**  
  Measure of acceleration, in g, experienced by the wrist-worn watch along each standard axis.

- **enmo**  
  As calculated and described by the wristpy package, ENMO is the Euclidean Norm Minus One of all accelerometer signals (along each of the X, Y, and Z axes, measured in g-force) with negative values rounded to zero. Zero values indicate periods of no motion. While no standard measure of acceleration exists in this space, this is one of the several commonly computed features.

- **anglez**  
  As calculated and described by the wristpy package, Angle-Z is a metric derived from individual accelerometer components and refers to the angle of the arm relative to the horizontal plane.

- **non-wear_flag**  
  A flag (0: watch is being worn, 1: watch is not worn) to help determine periods when the watch has been removed, based on the GGIR definition, which uses the standard deviation and range of the accelerometer data.

- **light**  
  Measure of ambient light in lux. See details [here](link to details).

- **battery_voltage**  
  A measure of the battery voltage in mV.

- **time_of_day**  
  Time of day representing the start of a 5-second window that the data has been sampled over, with format `%H:%M:%S.%9f`.

- **weekday**  
  The day of the week, coded as an integer with 1 being Monday and 7 being Sunday.

- **quarter**  
  The quarter of the year, an integer from 1 to 4.

- **relative_date_PCIAT**  
  The number of days (integer) since the PCIAT test was administered (negative days indicate that the actigraphy data has been collected before the test was administered).
"""


# %%
def extract_advanced_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract advanced features from actigraphy data for SII prediction.

    Parameters:
    data (pd.DataFrame): Input DataFrame with actigraphy measurements

    Returns:
    pd.DataFrame: Single row DataFrame with extracted features
    """
    # Initial data preprocessing
    data = data.copy()
    data["timestamp"] = pd.to_datetime(
        data["relative_date_PCIAT"], unit="D"
    ) + pd.to_timedelta(data["time_of_day"])
    data = data[data["non-wear_flag"] == 0]

    # Calculate basic metrics
    data["magnitude"] = np.sqrt(data["X"] ** 2 + data["Y"] ** 2 + data["Z"] ** 2)
    data["velocity"] = data["magnitude"]
    data["distance"] = data["velocity"] * 5  # 5 seconds per observation
    data["date"] = data["timestamp"].dt.date
    hour = pd.to_datetime(data["time_of_day"]).dt.hour

    # Calculate aggregated distances
    distances = {
        "daily": data.groupby("date")["distance"].sum(),
        "monthly": data.groupby(data["timestamp"].dt.to_period("M"))["distance"].sum(),
        "quarterly": data.groupby("quarter")["distance"].sum(),
    }

    # Initialize features dictionary
    features = {}

    # Time masks for different periods
    time_masks = {
        "morning": (hour >= 6) & (hour < 12),
        "afternoon": (hour >= 12) & (hour < 18),
        "evening": (hour >= 18) & (hour < 22),
        "night": (hour >= 22) | (hour < 6),
    }

    # 1. Activity Pattern Features
    for period, mask in time_masks.items():
        features.update(
            {
                f"{period}_activity_mean": data.loc[mask, "magnitude"].mean(),
                f"{period}_activity_std": data.loc[mask, "magnitude"].std(),
                f"{period}_enmo_mean": data.loc[mask, "enmo"].mean(),
            }
        )

    # 2. Sleep Quality Features
    sleep_hours = time_masks["night"]
    magnitude_threshold = data["magnitude"].mean() + data["magnitude"].std()

    features.update(
        {
            "sleep_movement_mean": data.loc[sleep_hours, "magnitude"].mean(),
            "sleep_movement_std": data.loc[sleep_hours, "magnitude"].std(),
            "sleep_disruption_count": len(
                data.loc[
                    sleep_hours
                    & (
                        data["magnitude"]
                        > data["magnitude"].mean() + 2 * data["magnitude"].std()
                    )
                ]
            ),
            "light_exposure_during_sleep": data.loc[sleep_hours, "light"].mean(),
            "sleep_position_changes": len(
                data.loc[sleep_hours & (abs(data["anglez"].diff()) > 45)]
            ),
            "good_sleep_cycle": int(data.loc[sleep_hours, "light"].mean() < 50),
        }
    )

    # 3. Activity Intensity Features
    features.update(
        {
            "sedentary_time_ratio": (
                data["magnitude"] < magnitude_threshold * 0.5
            ).mean(),
            "moderate_activity_ratio": (
                (data["magnitude"] >= magnitude_threshold * 0.5)
                & (data["magnitude"] < magnitude_threshold * 1.5)
            ).mean(),
            "vigorous_activity_ratio": (
                data["magnitude"] >= magnitude_threshold * 1.5
            ).mean(),
            "activity_peaks_per_day": len(
                data[data["magnitude"] > data["magnitude"].quantile(0.95)]
            )
            / len(data.groupby("relative_date_PCIAT")),
        }
    )

    # 4. Circadian Rhythm Features
    hourly_activity = data.groupby(hour)["magnitude"].mean()
    features.update(
        {
            "circadian_regularity": hourly_activity.std() / hourly_activity.mean(),
            "peak_activity_hour": hourly_activity.idxmax(),
            "trough_activity_hour": hourly_activity.idxmin(),
            "activity_range": hourly_activity.max() - hourly_activity.min(),
        }
    )

    # 5-11. Additional Feature Groups
    weekend_mask = data["weekday"].isin([6, 7])

    features.update(
        {
            # Movement Patterns
            "movement_entropy": stats.entropy(
                pd.qcut(data["magnitude"], q=10, duplicates="drop").value_counts()
            ),
            "direction_changes": len(data[abs(data["anglez"].diff()) > 30]) / len(data),
            "sustained_activity_periods": len(
                data[data["magnitude"].rolling(12).mean() > magnitude_threshold]
            )
            / len(data),
            # Weekend vs Weekday
            "weekend_activity_ratio": data.loc[weekend_mask, "magnitude"].mean()
            / data.loc[~weekend_mask, "magnitude"].mean(),
            "weekend_sleep_difference": data.loc[
                weekend_mask & sleep_hours, "magnitude"
            ].mean()
            - data.loc[~weekend_mask & sleep_hours, "magnitude"].mean(),
            # Non-wear Time
            "wear_time_ratio": (data["non-wear_flag"] == 0).mean(),
            "wear_consistency": len(data["non-wear_flag"].value_counts()),
            "longest_wear_streak": data["non-wear_flag"]
            .eq(0)
            .astype(int)
            .groupby(data["non-wear_flag"].ne(0).cumsum())
            .sum()
            .max(),
            # Device Usage
            "screen_time_proxy": (data["light"] > data["light"].quantile(0.75)).mean(),
            "dark_environment_ratio": (
                data["light"] < data["light"].quantile(0.25)
            ).mean(),
            "light_variation": (
                data["light"].std() / data["light"].mean()
                if data["light"].mean() != 0
                else 0
            ),
            # Battery Usage
            "battery_drain_rate": -np.polyfit(
                range(len(data)), data["battery_voltage"], 1
            )[0],
            "battery_variability": data["battery_voltage"].std(),
            "low_battery_time": (
                data["battery_voltage"] < data["battery_voltage"].quantile(0.1)
            ).mean(),
            # Time-based
            "days_monitored": data["relative_date_PCIAT"].nunique(),
            "total_active_hours": len(
                data[data["magnitude"] > magnitude_threshold * 0.5]
            )
            * 5
            / 3600,
            "activity_regularity": data.groupby("weekday")["magnitude"].mean().std(),
        }
    )

    # Variability Features for multiple columns
    for col in ["X", "Y", "Z", "enmo", "anglez"]:
        features.update(
            {
                f"{col}_skewness": data[col].skew(),
                f"{col}_kurtosis": data[col].kurtosis(),
                f"{col}_trend": np.polyfit(range(len(data)), data[col], 1)[0],
            }
        )

    return pd.DataFrame([features])


def process_file(filename: str, dirname: str) -> Tuple[np.ndarray, str]:
    df = pd.read_parquet(os.path.join(dirname, filename, "part-0.parquet"))
    data = extract_advanced_features(df)
    array_1 = data.values[0]
    array_2 = df.describe().values.reshape(-1), filename.split("=")[1]
    # Combine the two arrays
    combined_array = np.concatenate((array_1, array_2[0]))
    combined_tuple = (array_1, array_2[1])
    return combined_tuple


def load_time_series(dirname: str) -> pd.DataFrame:
    ids = os.listdir(dirname)

    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(lambda fname: process_file(fname, dirname), ids),
                total=len(ids),
            )
        )

    stats, indexes = zip(*results)

    df = pd.DataFrame(stats, columns=[f"stat_{i}" for i in range(len(stats[0]))])
    df["id"] = indexes
    return df


# %%
train_ts = load_time_series(
    r"C:\Users\ragha\Desktop\Competition\child-mind-institute-problematic-internet-use\series_train.parquet"
)


train_ts.to_csv(r"train_ts.csv", index=False)

# %%
test_ts = load_time_series(
    r"C:\Users\ragha\Desktop\Competition\child-mind-institute-problematic-internet-use\series_test.parquet"
)

test_ts.to_csv(r"test_ts.csv", index=False)

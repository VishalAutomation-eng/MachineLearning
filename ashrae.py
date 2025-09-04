# ===================== #
# 1. Import Libraries
# ===================== #
import numpy as np
import pandas as pd
import gc, os
from datetime import datetime
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb


#Load Data

base_path = "/home/vkpande/Documents/machine learning/ashrae-energy-prediction"

train = pd.read_csv(f"{base_path}/train.csv")
building = pd.read_csv(f"{base_path}/building_metadata.csv")
weather_train = pd.read_csv(f"{base_path}/weather_train.csv")


# 3. Reduce Memory Usage

def reduce_memory_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

building = reduce_memory_usage(building)
weather_train = reduce_memory_usage(weather_train)
train = reduce_memory_usage(train)


# 4. Feature Engineering

# Building Age
current_year = datetime.now().year
building["building_age"] = current_year - building["year_built"]
building.drop(columns=["year_built", "floor_count"], inplace=True, errors="ignore")
building.fillna(building["building_age"].mean(), inplace=True)

# Encode primary_use
le = LabelEncoder()
building["primary_use"] = le.fit_transform(building["primary_use"])

# Convert timestamp
train["timestamp"] = pd.to_datetime(train["timestamp"])
weather_train["timestamp"] = pd.to_datetime(weather_train["timestamp"])

# Datetime features
weather_train["month"] = weather_train["timestamp"].dt.month.astype(np.int8)
weather_train["day_of_week"] = weather_train["timestamp"].dt.dayofweek.astype(np.int8)
weather_train["day_of_month"] = weather_train["timestamp"].dt.day.astype(np.int8)
weather_train["hour"] = weather_train["timestamp"].dt.hour.astype(np.int8)
weather_train["is_weekend"] = weather_train["day_of_week"].apply(lambda x: 1 if x >= 5 else 0).astype(np.int8)

# Season
def convert_season(month):
    if (month <= 2) | (month == 12): return 0  # Winter
    elif month <= 5: return 1  # Spring
    elif month <= 8: return 2  # Summer
    elif month <= 11: return 3  # Fall
weather_train["season"] = weather_train["month"].apply(convert_season).astype(np.int8)


# 5. Handle Missing Weather Data

def fill_missing_by_group(df, column):
    filler = df.groupby(["site_id","day_of_month","month"])[column].transform("mean")
    df[column].fillna(filler, inplace=True)
    if df[column].isna().sum() > 0:
        df[column].fillna(df[column].mean(), inplace=True)
    return df

for col in ["air_temperature", "cloud_coverage", "dew_temperature", 
            "precip_depth_1_hr", "sea_level_pressure", "wind_speed"]:
    weather_train = fill_missing_by_group(weather_train, col)

# Wind direction → compass
def convert_direction(degrees):
    if degrees <= 90: return 0
    elif degrees <= 180: return 1
    elif degrees <= 270: return 2
    else: return 3
weather_train["wind_compass_direction"] = weather_train["wind_direction"].apply(convert_direction).astype(np.int8)
weather_train.drop(columns=["wind_direction"], inplace=True)


# 6. Merge Data

train = train.merge(building, on="building_id", how="left")
train = train.merge(weather_train, on=["site_id","timestamp"], how="left")


# 7. Target Transformation

train["log_meter_reading"] = np.log1p(train["meter_reading"])


# 8. Feature Selection (basic correlations)

correlations = train.corr()["log_meter_reading"].reset_index()
correlations.columns = ["Feature", "Correlation"]
selected_features = correlations[
    correlations["Feature"].isin(train.columns) &
    (correlations["Feature"] != "log_meter_reading")
]["Feature"].tolist()

# ✅ Drop raw datetime column if still present
if "timestamp" in selected_features:
    selected_features.remove("timestamp")

X = train[selected_features]
y = train["log_meter_reading"]

print("✅ Preprocessing complete!")
print("Features shape:", X.shape)
print("Target shape:", y.shape)


# 9. Downsample + Train-Test Split


# Downsample for testing (use full dataset later)
sample_frac = 0.02   # 2% of data
train_sample = train.sample(frac=sample_frac, random_state=42)

X = train_sample[selected_features]
y = train_sample["log_meter_reading"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Free up big original df
del train_sample
gc.collect()


# 10. LightGBM with Dataset API

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}

print("✅ Training LightGBM...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=10000,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(100)]
)


# 11. Evaluation

y_pred = model.predict(X_valid)

rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
r2 = r2_score(y_valid, y_pred)

print(f"📊 LightGBM Results:")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score (accuracy): {r2:.4f}")

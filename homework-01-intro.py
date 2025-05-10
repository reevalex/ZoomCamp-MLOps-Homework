from ast import Dict
from numpy import mean
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

"""
Note: This code is run in vscode using jupyter interactive
"""
# ------------------------------------------------------------
# 1. Downloading the data
# ------------------------------------------------------------
df = pd.read_parquet("data/yellow_tripdata_2023-01.parquet")
df.columns
len(df.columns)  # 19
total = len(df)

# -----------------------------------------------------------
# 2. Computing duration
# ------------------------------------------------------------
df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
df.duration = df.duration.apply(lambda x: x.total_seconds() / 60)
df.duration.std()  # 42.59

# -----------------------------------------------------------
# 3. Dropping outliers
# ------------------------------------------------------------
df = df[(df.duration >= 1) & (df.duration <= 60)]
df.duration.describe()
frac = len(df) / total  # 98%

# -----------------------------------------------------------
# 4. One-hot encoding
# ------------------------------------------------------------
df.dtypes
df.PULocationID = df.PULocationID.astype(str)
df.DOLocationID = df.DOLocationID.astype(str)

categorical = ["PULocationID", "DOLocationID"]
numerical = ["trip_distance"]
target = ["duration"]

dv = DictVectorizer()
train_dicts = df[categorical + numerical].to_dict(orient="records")
X_train = dv.fit_transform(train_dicts)  # 515
y_train = df[target].values

# -----------------------------------------------------------
# 5. Training a model
# ------------------------------------------------------------
clf = LinearRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
rmse = root_mean_squared_error(y_train, y_pred)  # 7.64

# -----------------------------------------------------------
# 6. Evaluating the model
# ------------------------------------------------------------
df_val = pd.read_parquet("data/yellow_tripdata_2023-02.parquet")
df_val["duration"] = df_val.tpep_dropoff_datetime - df_val.tpep_pickup_datetime
df_val.duration = df_val.duration.apply(lambda x: x.total_seconds() / 60)
df_val = df_val[(df_val.duration >= 1) & (df_val.duration <= 60)]

val_dicts = df_val[categorical + numerical].to_dict(orient="records")
X_val = dv.transform(val_dicts)
y_val = df_val[target].values
val_preds = clf.predict(X_val)
rmse_val = root_mean_squared_error(y_val, val_preds)  # 11.81

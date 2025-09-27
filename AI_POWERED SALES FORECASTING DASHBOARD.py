#!/usr/bin/env python
# coding: utf-8

# In[619]:


import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# In[620]:


file_path = r"C:\Users\koskr\Desktop\future_inturn\task_3\customer_support1\sample.csv"


# In[621]:


df = pd.read_csv(file_path, parse_dates=['created_at'])


# In[622]:


print(pd.DataFrame(df)) 


# In[623]:


df['date'] = df['created_at'].dt.floor('D')
daily = df.groupby('date').size().rename('y').to_frame()


# In[624]:


daily = daily.asfreq('D', fill_value=0)


# In[625]:


# ---------- 2) Feature engineering ----------
def make_features(df, lags=[1,2,3,7,14,30]):
    X = pd.DataFrame(index=df.index)
    X['dayofweek'] = df.index.dayofweek
    X['day'] = df.index.day
    X['month'] = df.index.month
    X['quarter'] = df.index.quarter
    X['is_month_start'] = df.index.is_month_start.astype(int)
    X['is_month_end'] = df.index.is_month_end.astype(int)
    # lag features
    for lag in lags:
        X[f'lag_{lag}'] = df['y'].shift(lag)
    # rolling features
    X['roll_7_mean'] = df['y'].shift(1).rolling(7).mean()
    X['roll_30_mean'] = df['y'].shift(1).rolling(30).mean()
    X = X.fillna(0)
    return X


# In[626]:


X = make_features(daily)
y = daily['y']


# In[627]:


valid = (X.index >= X.index.min() + pd.Timedelta(days=30))
X = X.loc[valid]
y = y.loc[valid]


# In[628]:


print(daily.shape)
print(X.shape)


# In[629]:


X = make_features(daily, lags=[1, 2, 7])  # instead of 30 days
y = make_features(daily, lags=[1, 2, 7])  # instead of 30 days


# In[630]:


valid = (X.index >= X.index.min() + pd.Timedelta(days=7))
X = make_features(daily, lags=[1, 7])  # only 2 lags
valid = (y.index >= y.index.min() + pd.Timedelta(days=7))
y = make_features(daily, lags=[1, 7])  # only 2 lags


# In[631]:


print("Daily rows:", daily.shape)
print("Features X:", X.shape)
print("Target y:", y.shape)


# In[632]:


print("Daily rows:", daily.shape)
print("Features X:", X.shape)
print("Target y:", y.shape)


# In[633]:


train_end = 14
if train_end == 0:
    test_size = 1  # fallback to at least 1

    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
else:
    X_train, X_test = X.iloc[:train_end], X.iloc[train_end:]
    y_train, y_test = y.iloc[:train_end], y.iloc[train_end:]


# In[634]:


model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1)
model.fit(X_train, y_train)


# In[635]:


train_end = -28
X_train, X_test = X.iloc[:train_end], X.iloc[train_end:]


# In[636]:


# Determine test size based on data length
test_size = min(7, len(X) // 5)  # last 7 rows or 20% of data
if test_size == 0:
    test_size = 1  # ensure at least 1 row

X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)


# In[637]:


X_train, y_train = X, y
X_test, y_test = X.iloc[0:0], y.iloc[0:0]  # empty test set

# Only predict if test set exists
if len(X_test) > 0:
    y_pred = model.predict(X_test)
else:
    print("No test set — model trained on full dataset")


# In[638]:


print("Total rows after feature engineering:", len(X))


# In[ ]:





# In[639]:


test_size = min(7, len(X) // 5)  # last 7 rows or 20% of data
if test_size == 0:  
    test_size = 1  # guarantee at least 1 row

X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]


# In[640]:


X_train, y_train = X, y
X_test, y_test = X.iloc[0:0], y.iloc[0:0]  # empty test set


# In[641]:


if len(X_test) > 0:
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"MAE: {mae:.3f}  RMSE: {rmse:.3f}")
else:
    print("No test set available — trained on full data.")


# In[642]:


print("Length of X:", len(X))
print("Length of y:", len(y))


# In[643]:


# ---- Train/Test Split ----
test_size = min(7, len(X) // 5)  # 7 rows OR 20% of data
if test_size == 0:
    test_size = 1   # guarantee at least 1 row

X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)


# In[644]:


y_pred = model.predict(X_test)


# In[645]:


y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"MAE: {mae:.3f}  RMSE: {rmse:.3f}")


# In[646]:


# Plot actual vs predicted
plt.figure(figsize=(12,5))
plt.plot(y_test.index, y_test.values, label='actual')
plt.plot(y_test.index, y_pred, label='predicted', linestyle='--')
plt.legend()
plt.title('Test: actual vs predicted')
plt.show()


# In[647]:


horizon = 42
last_window = daily.copy()  # we'll iteratively build future rows
future_index = pd.date_range(start=daily.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')

future_rows = []
temp = last_window.copy()


# In[648]:


# row = pd.Series(index=temp.index.append(pd.DatetimeIndex([date])))
row = pd.Series(index=temp.index.append(pd.DatetimeIndex([date])), dtype=float)


# In[649]:


X = make_features(daily, lags=[1,2,7])  # fixed set of lags
X = X[sorted(X.columns)]  # lock column order


# In[650]:


X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

# Save the exact columns used in training
feature_cols = X_train.columns


# In[651]:


feature_cols = X_train.columns  # save the column names

print("Training features:", list(feature_cols))


# In[652]:


X_aligned =  X_test.reindex(columns=feature_cols, fill_value=0)


# In[653]:


feat = feat.reindex(columns=feature_cols, fill_value=0)


# In[654]:


print("Training columns:", feature_cols.tolist())
print("Prediction columns:", feat.columns.tolist())


# In[655]:


X_test_aligned = X_test.reindex(columns=feature_cols, fill_value=0)


# In[656]:


print("Training columns:", list(feature_cols))
print("Prediction columns:", list(X_test_aligned.columns))


# In[657]:


feat = feat.reindex(columns=feature_cols, fill_value=-1)


# In[658]:


# X_train: features used to train
print("Training columns:", list(X_train.columns))
print("Number of training features:", len(X_train.columns))


# In[659]:


print("Prediction columns:", list(X_test.columns))  # or feat.columns
print("Number of prediction features:", len(X_test.columns))


# In[660]:


# Reindex to match training columns
extra_cols = set(X_test.columns) - set(X_train.columns)
missing_cols = set(X_train.columns) - set(X_test.columns)

print("Extra columns in prediction:", extra_cols)
print("Missing columns in prediction:", missing_cols)


# In[661]:


feature_cols = X_train.columns
print("Training features:", list(feature_cols))


# In[662]:


feature_cols = X_train.columns  # columns used in training
print("Training features:", list(feature_cols))


# In[663]:


# X is your new prediction data (may have extra columns)
X_aligned = X_test.reindex(columns=feature_cols, fill_value=0)


# In[664]:


X_aligned = X_test.reindex(columns=feature_cols, fill_value=0)


# In[665]:


X = X.drop(X.columns[0], axis=1)


# In[672]:


print(X_aligned)


# In[ ]:





# In[673]:


extra_cols = set(X.columns) - set(feature_cols)
missing_cols = set(feature_cols) - set(X.columns)

print("Extra columns in prediction:", extra_cols)
print("Missing columns in prediction:", missing_cols)


# In[674]:


extra_cols = set(X.columns) - set(feature_cols)
missing_cols = set(feature_cols) - set(X.columns)

print("Extra columns in prediction:", extra_cols)
print("Missing columns in prediction:", missing_cols)


# In[675]:


# Suppose your DataFrame is X and the extra column is 'extra_feature'
X = X.drop(X_aligned.columns[0], axis=1)


# In[670]:


future_rows = []
temp = daily.copy()

for date in future_index:
    # Add placeholder for this date
    extended = temp.append(pd.DataFrame({'y': [np.nan]}, index=[date]))
    # Build features
    feat = make_features(extended).loc[[date]]
    # Make sure it matches training columns
    feat = feat.reindex(columns=feature_cols, fill_value=0)
    # Predict
    pred = model.predict(feat)[0]
    future_rows.append(pred)
    # Add prediction to temp for next iteration
    temp.loc[date] = pred


# In[610]:


future_rows = [0]
temp = daily.copy()

for date in future_index:
    # add placeholder for this date
    extended = temp.append(pd.DataFrame({'y': [np.nan]}, index=[date]))
    # build features
    feat = make_features(extended).loc[[date]]
    # predict
    pred = model.predict(feat)[0]
    future_rows.append(pred)
    # update temp with predicted value for recursive forecasting
    temp.loc[date] = pred


# In[541]:


for date in future_index:
    # build features for this date using temp
    # construct single-row dataframe
    row = pd.Series(index=temp.index.append(pd.DatetimeIndex([date])))
    # we only need features function to accept this extended series:
    extended = temp.append(pd.DataFrame({'y': [np.nan]}, index=[date]))
    feat = make_features(extended).loc[[date]]
    pred = model.predict(feat)[0]
    future_rows.append(pred)
    # append prediction to temp for next iteration
    temp = temp.append(pd.DataFrame({'y': [pred]}, index=[date]))


# In[542]:


extra_cols = set(X_test.columns) - set(feature_cols)
print("Extra columns removed:", extra_cols)


# In[ ]:





# In[543]:


y_pred = model.predict(X_aligned)


# In[544]:


future = pd.Series(future_rows, index=future_index, name='forecast')


# In[ ]:





# In[ ]:





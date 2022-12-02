import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from PreProcess import PreProcess
import matplotlib.pyplot as plt

from prophet import Prophet

data_path = "../../data/20182019_LondonFirm_SalesTransaction.csv"
train, validation = PreProcess(data_path)

train_prophet = train.reset_index()
train_prophet.columns = ["ds","y"]
valid_prophet = validation.reset_index()
valid_prophet.columns = ["ds","y"]

model = Prophet()
model.fit(train_prophet)

future_index = pd.date_range(start=train_prophet.ds.tail(1).iloc[0], periods=31, freq="D")
future = pd.DataFrame(future_index)
future.columns = ['ds']

forecast = model.predict(future)
valid_prophet = pd.merge(valid_prophet,forecast[['ds', 'yhat']])

y_true = valid_prophet.y
y_pred = valid_prophet.yhat
print('Mae:', mean_absolute_error(y_true, y_pred))
print('Series mean:',np.mean(y_true))
print(mean_absolute_error(y_true, y_pred)/np.mean(y_true))
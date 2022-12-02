import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from PreProcess import PreProcess

from sklearn.multioutput import RegressorChain
from sklearn.ensemble import GradientBoostingRegressor

def Make_Rolling_Features(df):
    df_c = df.copy()
    lag = 30
    rolling_windows = [7,30]
    for window in rolling_windows:
        df_c['rolling_mean_'+str(window)] = df_c['Value'].rolling(window).mean().shift(1)
        df_c['rolling_std_'+str(window)] = df_c['Value'].rolling(window).std().shift(1)
        df_c['rolling_min_'+str(window)] = df_c['Value'].rolling(window).min().shift(1)
        df_c['rolling_max_'+str(window)] = df_c['Value'].rolling(window).max().shift(1)
        df_c['rolling_median_'+str(window)] = df_c['Value'].rolling(window).median().shift(1)
        df_c['rolling_quantile1st_'+str(window)] = df_c['Value'].rolling(window).quantile(0.25).shift(1)
        df_c['rolling_quantile3rd_'+str(window)] = df_c['Value'].rolling(window).quantile(0.75).shift(1)
        
    for i in range(lag):
        df_c['lag_'+str(i+1)] = df_c['Value'].shift(i+1)
    
    df_c['expanding_mean'] = df_c['Value'].expanding().mean().shift(1)
    df_c['expanding_std'] = df_c['Value'].expanding().std().shift(1)
    df_c['expanding_min'] = df_c['Value'].expanding().min().shift(1)
    df_c['expanding_var'] = df_c['Value'].expanding().var().shift(1)
    
    df_c['month'] = df_c['Date'].dt.month
    df_c['day_of_week'] = df_c['Date'].dt.dayofweek
    df_c['day_of_month'] = df_c['Date'].dt.day
              
    return(df_c)

def Make_Y(y,fh):
    y_ = pd.DataFrame()
    for i in range(fh):
        y_[str(i+1)] = y.shift(-1*i)
        
    return y_.iloc[:,:fh]


data_path = "../../data/20182019_LondonFirm_SalesTransaction.csv"
train, validation = PreProcess(data_path)

fh = 30
train_f = train.reset_index()
valid_f = validation.reset_index()

future_date_index = pd.date_range(start=train_f.tail(1).Date.iloc[0], periods=fh+1, freq="D", name="Date")
future = pd.DataFrame(future_date_index)
total_train = pd.merge(train_f,future,how="outer")

df_f = Make_Rolling_Features(total_train)
df_f = df_f.iloc[30:].set_index("Date")

historical = df_f.iloc[:-30]
future = df_f.iloc[-30]

X = historical.iloc[:,1:]
y = historical.iloc[:,0]

y = Make_Y(y,30)
y_train = y.iloc[:-30,:]
X_train = X.iloc[:-30,:]

X_valid = future[1:].to_frame()

gbr_seq = GradientBoostingRegressor(random_state=42)
chained_gbr = RegressorChain(gbr_seq)
chained_gbr.fit(X_train, y_train)
gbr_seq_preds = chained_gbr.predict(X_valid.transpose())

y_true = valid_f.Value
y_pred = gbr_seq_preds.T
print('Mae:', mean_absolute_error(y_true, y_pred))
print('Series mean:',np.mean(y_true))
print(mean_absolute_error(y_true, y_pred)/np.mean(y_true))
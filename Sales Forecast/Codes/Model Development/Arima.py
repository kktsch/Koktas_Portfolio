import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PreProcess import PreProcess

from tqdm import notebook
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error


def optimize_ARIMA(endog, order_list, d):

    results = [] # To store p,q values

    for order in notebook.tqdm(order_list):  # Iterate over p,q values
        try:
            model = SARIMAX(endog, order=(order[0], d, order[1]),
                          simple_differencing=False).fit(disp=False) 
        except:
            continue

        aic = model.aic # Calcute each model's AIC
        results.append([order, aic])  # Store p,q combination and their model's AIC

    result_df = pd.DataFrame(results) # Store p,q combinations and their model's AICs in a frame
    result_df.columns = ['(p,q)', 'AIC'] 

    #Sort in ascending order
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True) # Sort the frame in ascending order to find minimum AIC

    return result_df 

data_path = "../../data/20182019_LondonFirm_SalesTransaction.csv"
train, validation = PreProcess(data_path)

values = train.Value.values
ADF_result = adfuller(values)
print("Series ADF:")
print(f'p-value: {ADF_result[1]}') 

ps = range(0, 10, 1) 
qs = range(0, 10, 1) 
d = 0
order_list = list(product(ps, qs))

result_df = optimize_ARIMA(values, order_list, d) 
print(result_df)

model = SARIMAX(values, order=(7,d,7), simple_differencing=False)
model_fit = model.fit(disp=False)
model_fit.plot_diagnostics()
plt.show()

residuals = model_fit.resid 
acorr_ljungbox(residuals, np.arange(1, 11, 1))

ARIMA_pred = model_fit.get_prediction(314, 343).predicted_mean
pred_table = pd.DataFrame({'Real':validation.Value.values, 'ARIMA':ARIMA_pred})

y_true = pred_table.Real
y_pred = pred_table.ARIMA
print('Mae:', mean_absolute_error(y_true, y_pred))
print('Series mean:',np.mean(pred_table.Real))
print(mean_absolute_error(y_true, y_pred)/np.mean(pred_table.Real))
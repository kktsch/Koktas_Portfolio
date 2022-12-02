import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from PreProcess import PreProcess
import matplotlib.pyplot as plt

data_path = "../../data/20182019_LondonFirm_SalesTransaction.csv"

train, validation = PreProcess(data_path)

forecast = train.iloc[-30:]
pred_table = pd.DataFrame({'Real':validation.Value.values, 'Base_Model':forecast.Value.values})

y_true = pred_table.Real
y_pred = pred_table.Base_Model

print('Mae:', mean_absolute_error(y_true, y_pred))
print('Series mean:',np.mean(pred_table.Real))
print(mean_absolute_error(y_true, y_pred)/np.mean(pred_table.Real))

fig, ax = plt.subplots(figsize=(20,4))
ax.plot(y_true, 'b-', label='actual')
ax.plot(y_pred, 'k--', label='ARIMA')
ax.legend(loc=2)
fig.autofmt_xdate()
plt.tight_layout() 
plt.show()
import pandas as pd 
import matplotlib.pyplot as plt
import pmdarima as pm 
from os import listdir 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller 
from statsmodels.graphics.tsaplots import plot_acf
from numpy import log 



fpath = "datasets/"
files = [f for f in listdir(fpath)]

dataset = files[-1]

df_daily = pd.read_excel(fpath+dataset, sheet_name="Table 1 Daily", header=4) 
# print(df_daily.head())
# df_daily.set_index("Date")["7-day average"].plot()

df_monthly = pd.read_excel(fpath+dataset, sheet_name="Table 2 Monthly", header=4) 
df_monthly.set_index("Date", inplace=True)

#### AD Fuller Test #### 
# result = adfuller(df_monthly["Monthly average"])
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])

#### Plots ### 
fig, axs = plt.subplots(3, 2) 

axs[0, 0].plot(df_monthly)
axs[0, 0].set_title("Original Series")
plot_acf(df_monthly.dropna(), ax=axs[0, 1])

axs[1, 0].plot(df_monthly.diff())
axs[1, 0].set_title("1st Order Diff")
plot_acf(df_monthly.diff().dropna(), ax=axs[1, 1])

axs[2, 0].plot(df_monthly.diff().diff())
axs[2, 0].set_title("2nd Order Diff")
plot_acf(df_monthly.diff().diff().dropna(), ax=axs[2, 1])

plt.show()


train = df_monthly[:int(len(df_monthly)*0.75)]
test = df_monthly[int(len(df_monthly)*0.75):]

# test.plot()

# model = ARIMA(df_monthly, order=(1,1,1))

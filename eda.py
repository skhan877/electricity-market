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
# fig, axs = plt.subplots(3, 2) 

# axs[0, 0].plot(df_monthly)
# axs[0, 0].set_title("Original Series")
# plot_acf(df_monthly.dropna(), ax=axs[0, 1])

# axs[1, 0].plot(df_monthly.diff())
# axs[1, 0].set_title("1st Order Diff")
# plot_acf(df_monthly.diff().dropna(), ax=axs[1, 1])

# axs[2, 0].plot(df_monthly.diff().diff())
# axs[2, 0].set_title("2nd Order Diff")
# plot_acf(df_monthly.diff().diff().dropna(), ax=axs[2, 1])

# plt.show()


train = df_monthly[:int(len(df_monthly)*0.75)]
test = df_monthly[int(len(df_monthly)*0.75):]

model = ARIMA(df_monthly, order=(1,1,1))
fitted = model.fit()
# print(fitted.summary())

forecast = fitted.get_forecast(steps=len(test))
df = pd.DataFrame(forecast.summary_frame(alpha=0.05))
# print(df[["mean", "mean_ci_lower"]])


fc_series = pd.Series(df["mean"].values, index=test.index)
lower_series = pd.Series(df["mean_ci_lower"].values, index=test.index)
upper_series = pd.Series(df["mean_ci_upper"].values, index=test.index)


plt.figure(figsize=(15,10), dpi=100)
plt.plot(train, label="training data")
plt.plot(test, label="actual data")
plt.plot(fc_series, label="forecast")
plt.fill_between(lower_series.index, lower_series, upper_series, color="k", alpha=0.15)
plt.title("Actual vs Forecast")
plt.legend(loc="upper left", fontsize=8)
plt.show() 

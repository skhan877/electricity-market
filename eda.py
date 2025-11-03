import pandas as pd 
import matplotlib.pyplot as plt
from os import listdir 

fpath = "datasets/"
files = [f for f in listdir(fpath)]

dataset = files[-1]

df_daily = pd.read_excel(fpath+dataset, sheet_name="Table 1 Daily", header=4) 
df_monthly = pd.read_excel(fpath+dataset, sheet_name="Table 2 Monthly", header=4) 
# print(df_monthly.head())

df_monthly.set_index("Date").plot()
plt.show()
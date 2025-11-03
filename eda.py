import pandas as pd 
from os import listdir 

fpath = "datasets/"
files = [f for f in listdir(fpath)]

print(files)
import numpy as np
from pandas import read_csv
import pandas as pd 
from matplotlib import pyplot as plt


#DATA MANIPULATION AND CLEANING CELL
series = read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv', index_col = "Country/Region")
series = series.drop(["Province/State","Lat","Long"], axis=1)
series.index.names = ["Country"]
country = pd.Series(series.index.unique().to_numpy())
series = series.groupby(['Country']).sum()
series = series.T
index = pd.date_range(start='1/22/20', end=series.tail(1).index.item(), freq='d')
series.set_index(index, inplace = True) 

#Removing the stationarity from the dataset
series.India = np.sqrt(series.India) #Square root of the data
series.dropna(inplace=True)
data_diff=series.India
datat=data_diff-data_diff.shift() #First difference
data0=datat
series.India=data0-data0.shift()  #Second difference
series.India.dropna(inplace=True)
plt.plot(series.India)


import numpy as np
from pandas import read_csv
import pandas as pd 
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import adfuller, kpss

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


#Test for stationarity
#Dickey Fuller Test
def dickey(data):
    test=adfuller(data, autolag='AIC')
    out=pd.Series(test[0:4],index=['Test Statistics','p','lags used','no. of observation'])
    for key, values in test[4].items():
        out['Critical Value (%s)'%key]=values
    print(out)
    
dickey(series.India)

#KPSS Test
def kps(data):
    test=kpss(data, regression='c')
    out=pd.Series(test[0:3],index=['Test Statistic', 'p','Lags Used'])
    for key, value in test[3].items():
        out['Critical values (%s)'%key]=value
    print(out)
    
kps(series.India)
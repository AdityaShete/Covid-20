import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot

datam=pd.read_excel("covid_19_india.xlsx")
del_col=['Sno','Time','State/UnionTerritory','ConfirmedIndianNational','ConfirmedForeignNational','Cured','Deaths']
datan=datam.drop(del_col,axis=1)
data_real=datan.groupby('Date').sum()
data_real.dropna(inplace=True)
data2=pd.read_excel("INDIA.xlsx")
del_col=['Sno','Time','State/UnionTerritory','ConfirmedIndianNational','ConfirmedForeignNational','Cured','Deaths']
data3=data2.drop(del_col,axis=1)
data_r=data3.groupby('Date').sum()
data_r.dropna(inplace=True)

#Removing the stationarity from the dataset
#data12['Confirmed']=np.sqrt(data12['Confirmed']) #Square root of the data
#data12.dropna(inplace=True)
#data_diff=data12
#datat=data_diff-data_diff.shift() #First difference
#data0=datat
#data=data0-data0.shift()  #Second difference
#data.dropna(inplace=True)

#Decomposing the data
#decompose=seasonal_decompose(data_diff)
#trend=decompose.trend
#seasonal=decompose.seasonal
#residual=decompose.resid
#plt.subplot(411)
#plt.plot(data_diff, label="Original")
#plt.legend(loc='best')
#plt.subplot(412)
#plt.plot(trend, label='Trend')
#plt.legend(loc='best')
#plt.subplot(413)
#plt.plot(seasonal, label='Seasonality')
#plt.legend(loc='best')
#plt.subplot(414)
#plt.plot(residual, label='Residual')
#plt.legend(loc='best')
#plt.tight_layout()

#Test for stationarity
#Dickey Fuller Test
#def dickey(data):
#    test=adfuller(data, autolag='AIC')
#    out=pd.Series(test[0:4],index=['Test Statistics','p','lags used','no. of observation'])
#    for key, values in test[4].items():
#        out['Critical Value (%s)'%key]=values
#    print(out)
#dickey(data['Confirmed'])
#
##KPSS Test
#def kps(data):
#    test=kpss(data, regression='c')
#    out=pd.Series(test[0:3],index=['Test Statistic', 'p','Lags Used'])
#    for key, value in test[3].items():
#        out['Critical values (%s)'%key]=value
#    print(out)
#kps(data['Confirmed'])

#
##ACF Plot
#plot_acf(data, lags=20)
#ACF= acf(data, nlags=20)
#plt.plot(ACF)
#plt.axhline(y=0, linestyle='--', color='gray')
#
##PACF Plot
#plot_pacf(data, lags=20)
#PACF= pacf(data, nlags=20, method='ols')
#plt.plot(PACF)
#plt.axhline(y=0, linestyle='--', color='gray')

##ARIMA Fitting
#models=ARIMA(data_diff, order=(1,2,1))
#results_arima=models.fit(disp=-1)
##plt.plot(data)
##plt.plot(results_arima.fittedvalues, color='red')
#
##Predictions
#predictions_diff2=pd.Series(results_arima.fittedvalues, copy="True")
#predictions_diff1=predictions_diff2.cumsum()
#predictions_cumsum=predictions_diff1.cumsum()
#predictions_sqrt=pd.Series(data['Confirmed'].ix[0], index=data.index)
#predictions_sqrt=predictions_sqrt.add(predictions_cumsum, fill_value=0)
#predictions=np.square(predictions_sqrt)
#x=results_arima.forecast(steps=5)

model=ARIMA(data_real, order=(1,2,0))
model_fit=model.fit(disp=0)
f,g,h=model_fit.forecast(7,alpha=0.05)
fs=pd.Series(f,index = pd.date_range(start = data_real.index.max() + pd.Timedelta(1,unit = 'd'), end = data_real.index.max()+ pd.Timedelta(7, unit= 'd'), freq = 'd'))
plt.plot(data_r)
plt.plot(fs)
print(fs)






















#data0=data[1:len(data)-7]
#data1=data[len(data)-7:len(data)]
#train0=AR(data0)
#train=train0.fit()
#window=train.k_ar
#predictions = train.predict(start=len(data0), end=len(data0)+len(data1)-1, dynamic=False)
#plt.plot(data)
#plt.plot(predictions, color='red')
#plt.plot(data1, color='yellow')
#plot_pacf(data)
#lag_plot(data0)
#autocorrelation_plot(data)
#plt.show()
#



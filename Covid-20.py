#!/usr/bin/env python
# coding: utf-8


from pandas import read_csv
import pandas as pd 
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults

#DATA MANIPULATION AND CLEANING CELL
series = read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv', index_col = "Country/Region")
series = series.drop(["Province/State","Lat","Long"], axis=1)
series.index.names = ["Country"]
country = pd.Series(series.index.unique().to_numpy())
series = series.groupby(['Country']).sum()
series = series.T
index = pd.date_range(start='1/22/20', end=series.tail(1).index.item(), freq='d')
series.set_index(index, inplace = True)

#MODEL IMPLEMENTATION 
model = ARIMA(series.India, order=(1,2,1))
model_fit = model.fit(disp=0)
pre, e, err = model_fit.forecast(steps=4, alpha = .05)
predict = pd.Series(pre, index = pd.date_range(start = series.index.max() + pd.Timedelta(1,unit = 'd'), end = series.index.max()+ pd.Timedelta(4, unit= 'd'), freq = 'd'))
error_lower = pd.Series(err[:,0], index = predict.index)
error_upper = pd.Series(err[:,1], index = predict.index)
print(predict)
print(model_fit.summary())


# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot(figsize = (10,5))
plt.show()
residuals.plot(kind='kde',figsize = (10,5))
axes = plt.gca()
axes.set_xlim([-300,300])
plt.show()
print(residuals.describe())
print(err)


#PLOT

plt.figure(figsize=(10,5))
plt.plot(series.India.tail(20))
plt.plot(predict)
plt.fill_between(predict.index, error_lower, error_upper, color='k', alpha=.15)
plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib 

res = seasonal_decompose(series.India, model='additive')
fig = res.plot()
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(9,9)
plt.show()








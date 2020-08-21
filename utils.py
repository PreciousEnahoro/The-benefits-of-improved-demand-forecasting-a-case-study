#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Installing general public packages

import pandas as pd, numpy as np, datetime, holidays,matplotlib.pyplot as plt,statsmodels.api as sm, seaborn as sns, warnings
from datetime import date
import seaborn as sns
from matplotlib import dates as mdates
sns.set(style="whitegrid")
from sklearn.metrics import mean_absolute_error
from fbprophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fbprophet.plot import plot_plotly
import plotly.offline as py
from statsmodels.tsa.stattools import adfuller


# In[ ]:


def observe_data_structure(data):
    
    """This function uses dataset.info and dataset.describe to know 
        what variables need processing and observe the summary statistics
        of the dataset.
    """ 
    print(data.info())
    print('\n Summary Statistics \n')
    print(data.describe())


# In[ ]:


def split_data_and_drop_train_column(file, number):
    
    """This function splits the dataset into the train and test sets, 
    and drops the train indicator column after. 
    """ 
    x = file[file['train']==number]
    x = x.drop(columns = ['train'])
    file = x
    return file


# In[ ]:


def best_and_worst_selling_items(data):
    
    """This function returns the best selling and worst selling items in the dataset, 
    in terms of quantity of the item sold.
    """  

    sorted_list = data.groupby('item_nbr', as_index=False).sum()["units"].sort_values()

    lowest_5 = sorted_list[:5]
    highest_5 = sorted_list[106:]


    #Combining data in dataframe
    new_item_data = pd.DataFrame(dict(lowest_5 = lowest_5, 
                                            highest_5=highest_5)).reset_index()
    new_item_data = new_item_data.fillna(0)
    new_item_data['Total number sold'] = new_item_data['lowest_5'] + new_item_data['highest_5']
    new_item_data = new_item_data.drop(columns = ['lowest_5', 'highest_5'])


    #Renaming columns, sorting dataframe and adjusting formatting

    new_item_data = new_item_data.rename(columns={"index": "Item_Number"})
    new_item_data = new_item_data.sort_values(['Total number sold'])
    new_item_data['Total number sold'] = new_item_data['Total number sold'].apply(lambda x: '%.1f' % x)
    new_item_data = new_item_data.reset_index(drop=True)

    #Showing table
    return new_item_data


# In[ ]:


def reformatting_data(file, item_number):
    
    """This function reformats the dataset to be an ideal format for time series decomposition and forecasting.
    The data is also filtered for only the entries of the specified item number.
    """  
    file = file.copy()
    x = file.loc[file['item_nbr']==item_number]
    x_new = x.groupby('date').sum()["units"].reset_index()
    return x_new


# In[ ]:


def plot_seasonal_decomposition(dataset,dataset_date,frequency):
    
    """This function plots the observed time series and its trend, seasonality, residuals.""" 
    
    seasonal_decomposition=sm.tsa.seasonal_decompose(dataset,freq=frequency) 
    
    observed = seasonal_decomposition.observed
    trend = seasonal_decomposition.trend
    seasonal = seasonal_decomposition.seasonal
    residual = seasonal_decomposition.resid
    
    df = pd.DataFrame({"observed":observed,"trend":trend, "seasonal":seasonal,"residual":residual})
    df = df.set_index(dataset_date)

    years = mdates.YearLocator()    # only print label for the years
    months = mdates.MonthLocator()  # mark months as ticks
    years_fmt = mdates.DateFormatter('%Y-%b')
    fmt = mdates.DateFormatter('%b')

    _, axes = plt.subplots(nrows=4,ncols=1, figsize=(20, 15))
    _.tight_layout(pad=8.0)
    for i, ax in enumerate(axes):
        ax = df.iloc[:,i].plot(ax=ax)
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)
        ax.xaxis.set_minor_formatter(fmt)
        ax.set_ylabel(df.iloc[:,i].name, fontsize=18)
        ax.set_xlabel('Date', fontsize=16)
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=90, fontsize = 14)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, fontsize = 14)

# In[ ]:
def eda_plots(data, xlabels, x_axis_label, title):
    
    """This function creates plots that show the most profitable day in the week and month in the year in 
       the dataset. For the colour scheme, every bar for each day would be grey.The day with the highest 
       total sales would be green and the day with the lowest total sales would be red.
    """ 
    #Getting colour scheme
    
    clrs = []

    for x in data:
        if x == max(data):
            clrs.append('green')
        elif x == min(data):
            clrs.append('red')
        else:
            clrs.append('grey')
        
    # Plotting
    plt.figure(figsize=(15,5))
    sns.barplot(x=xlabels, y=data, palette=clrs)
    plt.xlabel(x_axis_label,fontsize = 16)
    plt.ylabel('Total Sales', fontsize = 16)
    plt.title(title,fontsize = 16)

# In[ ]:
def test_stationarity(timeseries):

    """This function carries out the Dickey-Fuller Test to check if the time series is stationary
       or not.
    """ 

    #Determing rolling statistics
    rolmean = timeseries.rolling(365).mean()
    rolstd = timeseries.rolling(365).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput) 


# In[ ]:
def acf_pacf_plotter(data):
    
    """This function plots autocorrelation function and partial 
       autocorrelation function plots.
    """ 
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data, lags = 30, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data, lags = 30, ax=ax2)

# In[ ]:
def cost_of_forecast_error(actual, model_forecast):
    """
    Cost of Forecast Error
    
    Calculating the total dollar cost of forecast inaccuracy, either by overstock or
    understock
    """
    costs = []
    
    for i in range(len(actual)):
    
        #Cost of understock
        if round(model_forecast[i]) < actual[i]: 
            costs.append((abs(actual[i] - round(model_forecast[i])) * 12.5))

        #Cost of overstock
        elif round(model_forecast[i]) > actual[i]: 
            costs.append((abs(actual[i] - round(model_forecast[i])) * 5))

        else:
            costs.append(0)

    return sum(costs)

# In[ ]:
    
def evaluator(actual, model_forecast):   
    """
    This function returns the results of the performance of the forecasting models.
    """
    print('Cost of forecast error is: $', cost_of_forecast_error(actual, model_forecast))
    print('Mean absolute error is:', round(mean_absolute_error(actual, model_forecast)))
    
# In[ ]:

def plotting_and_evaluating_results(model_name, actual, dates, model_forecast, \
                                    lower_bound = [], upper_bound = []):
    
    """
    This function shows how well each model performs according to the metrics,
    and plots the forecast against the test set, and the confidence intervals 
    when appropriate.
    """
    
    print ('\033[1m' + model_name + ' Results \033[0m')
    evaluator(actual, model_forecast)
    print('\n')

    #Plotting
    
    plt.figure(figsize=(12,8)) 
    plt.plot(dates, actual, label='Test', alpha=0.8)
    plt.xticks(rotation=90)
    plt.plot(dates,model_forecast, label = model_name, alpha = 1)
    plt.legend(loc='best') 

    ###sort the labelling and axes out
    plt.title("Actual Test Data versus " + model_name) 
    plt.xlabel('Date')
    plt.ylabel('Sales')

    
    #Plotting Confidence Intervals
    if len(lower_bound) > 0:

        plt.fill_between(dates,\
                    lower_bound, \
                    upper_bound,color='g', alpha=0.2)
        
    plt.show()
    
    #Residuals
    
    residuals = actual - model_forecast
    
    #Printing Largest Residual
    df = pd.DataFrame(dates)
    df['residuals'] = residuals
    df = df.sort_values(by='residuals')
    
    print('Date with largest residual is:')
    if abs(df.iloc[0].residuals) > abs(df.iloc[-1].residuals):
        print(df.iloc[0].date)
    else:
        print(df.iloc[-1].date)
    
    print('\n')
    #Autocorrelation and Partial Autocorrelation Plot of Residuals
    print("Autocorrelation and Partial Autocorrelation Plot of Residuals")
    acf_pacf_plotter(residuals)
    plt.show()
    
    print('\n')
    plt.hist(residuals) 
    plt.title("Distribution of residuals from " + model_name)
    plt.show()
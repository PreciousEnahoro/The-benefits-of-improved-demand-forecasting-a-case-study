# Capstone Project
### Precious Enahoro
### March 2020

The full, unprocessed data can be found here: [Walmart item sales/ demand data from 2012 -2014 on Kaggle.](https://www.kaggle.com/c/walmart-recruiting-sales-in-stormy-weather/data) 

The project write-up is titled: The benefits of improved demand forecasting: a case study


## Note: Please ensure to run the notebooks in this exact order: 
1) Data Pre-processing and Exploration
2) Feature Engineering
3) Forecasting and Evaluation


# Files:

1. `utils.py` - Loads all necessary public packages and functions used across all notebooks. If you have not already, pip install *fbprophet and statsmodels.api*.

2. `Data Pre-processing and Exploration.ipynb` - Pre-processing and exploring *sales.csv*.

3. `Feature Engineering.ipynb` - Creating features/ exogenous variables for the time series models.

4. `Forecasting and Evaluation.ipynb`  - Implements and evaluates Holt-Winters and Prophet models.

5. `train_trend.csv` - Training set from 2012 - 2013 Walmart item sales/ demand data, that has been pre-processed and feature engineered. Can directly be used in `Forecasting and Evaluation.ipynb` if the other notebooks have not been run yet.

6. `test_trend.csv` - Test set from 2014 Walmart item sales/ demand data, that has been pre-processed and feature engineered. Can directly be used in `Forecasting and Evaluation.ipynb` if the other notebooks have not been run yet.

7. `weather.csv` - Data containing weather variables that can be added in `Forecasting and Evaluation.ipynb`, as one of the recommended exogenous variables. From this dataset, I only used the *tavg* and *preciptotal* variables, as they made the most sense for the application.


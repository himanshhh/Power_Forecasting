# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 20:13:31 2025

@author: himan
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot, autocorrelation_plot
from pandas import DataFrame, concat, Grouper
from statsmodels.graphics.tsaplots import plot_acf
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.api import AutoReg
import gc
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#%% Data Loading
my_missing = ['NA','NULL','--', ' '] # Load the values in the list as missing values
series = pd.read_csv("Tetuan City power consumption.csv", na_values = my_missing, header=0, index_col=0, parse_dates=True, dayfirst=False)
series.head()
#%% Check for missing values
series.isnull().sum()
#%%
series.info()
#%%
series.shape
#%%
data_description = series.describe()
print(data_description)
#%% Correlation Matrix

#numerical_variables = series.drop(columns=['DateTime'])

plt.figure(figsize=(10, 6))
sns.heatmap(series.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

#%% Boxplots to detect outliers in features
plt.figure(figsize=(12, 6))
sns.boxplot(data=series.iloc[:, 0:-3], orient="h", palette="husl")
plt.title("Box Plots to Detect Outliers", fontsize=16)
plt.show()

#%%Create date-based features
dataframe = DataFrame()
dataframe['month'] = series.index.month
dataframe['day'] = series.index.day
dataframe['hour'] = series.index.hour
dataframe['minute'] = series.index.minute
dataframe['temperature'] = series['Temperature']
dataframe['humidity'] = series['Humidity']
dataframe['wind_speed'] = series['Wind Speed']
dataframe['general_diffuse'] = series['general diffuse flows']
dataframe['diffuse'] = series['diffuse flows']
dataframe['zone1'] = series['Zone 1 Power Consumption']
dataframe['zone2'] = series['Zone 2  Power Consumption']
dataframe['zone3'] = series['Zone 3  Power Consumption']
print(dataframe.head(10))

#%% Set target variables
power_consumption_zones = ['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']

#%%Line plot
for column in power_consumption_zones:
    if column in series.columns:
        series[column].plot()
        plt.xlabel('Month', fontdict={'fontname': 'Arial', 'fontsize': 14, 'color': 'red'})
        plt.ylabel('Power Consumption', fontdict={'fontname': 'Arial', 'fontsize': 14, 'color': 'red'})
        plt.title(f'Time Series Plot for {column}')
        plt.grid(True)
        plt.show()

#%%Histogram
bin_size=20

for column in power_consumption_zones:
    if column in series.columns:
        series[column].hist(color='green', bins=bin_size)
        plt.title(f'Histogram for {column}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

#%%Density plot
for column in power_consumption_zones:
    if column in series.columns:
        series[column].plot(kind='kde')
        plt.title(f'Density plot for {column}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
        
#%%Monthly Box plot
# one_year = series['2017']
# groups = one_year.groupby(Grouper(freq='ME'))
# months = concat([DataFrame(x[1].values) for x in groups], axis=1)
# months.columns = range(1,13)
# months.boxplot()
# plt.show()

#%%Lag plot
lag_plot(series)
plt.show()

#%%Autocorrelation plot
for column in power_consumption_zones:
    if column in series.columns:
        plt.figure(figsize=(10, 5))
        autocorrelation_plot(series[column].dropna())
        plt.title(f"Autocorrelation plot of {column}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

#%% Correlation Plot
for column in power_consumption_zones:
    if column in series.columns:
        values = series[[column]]
        dataframe = pd.concat([values.shift(1), values], axis=1)
        dataframe.columns = ['t', 't+1']
        print(f'\nLagged correlation for {column}:')
        print(dataframe.corr())

#%%ACF Plot Daily
for column in power_consumption_zones:
    if column in series.columns:
        
        plt.figure(figsize=(10, 4))
        plot_acf(series[column].dropna(), lags=144)
        plt.title(f'ACF Plot (144 lags) for {column}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
#%%ACF Plot Weekly
for column in power_consumption_zones:
    if column in series.columns:
        
        plt.figure(figsize=(10, 4))
        plot_acf(series[column].dropna(), lags=864)
        plt.title(f'ACF Plot (864 lags) for {column}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
#%%ACF Plot Monthly
for column in power_consumption_zones:
    if column in series.columns:

        plt.figure(figsize=(10, 4))
        plot_acf(series[column].dropna(), lags=4320)
        plt.title(f'ACF Plot (4320 lags) for {column}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
#%% Predictions using persistence model
for column in series.columns:
    print(f"\n--- Persistence Model for: {column} ---")

    values = series[[column]].dropna().values
    dataframe = pd.concat([series[[column]].shift(1), series[[column]]], axis=1).dropna()
    dataframe.columns = ['t', 't+1']
    data = dataframe.values

    train, test = data[:-7], data[-7:]
    train_X, train_y = train[:, 0], train[:, 1]
    test_X, test_y = test[:, 0], test[:, 1]

    predictions = test_X.tolist()
    rmse = sqrt(mean_squared_error(test_y, predictions))
    
    print(f'Test RMSE (Persistence Model): {rmse:.3f}')

#%%  Plot predictions vs expected
for column in power_consumption_zones: 
    if column in series.columns:
        values = series[[column]].dropna().values
        dataframe = pd.concat([series[[column]].shift(1), series[[column]]], axis=1).dropna()
        dataframe.columns = ['t', 't+1']
        data = dataframe.values
        test = data[-7:]
        test_X, test_y = test[:, 0], test[:, 1]

        predictions = test_X.tolist()

        plt.figure(figsize=(10, 5))
        plt.plot(test_y, label='Expected')
        plt.plot(predictions, color='red', label='Predicted')
        plt.title(f'Actual vs Predicted (Persistence Model) - {column}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
#%%AutoReg Model Optimization
for column in power_consumption_zones:
    if column in series.columns:
        print(f"\n--- AutoReg Model Optimization for {column} ---")

        values = series[[column]].dropna().values
        dataframe = pd.concat([series[[column]].shift(1), series[[column]]], axis=1).dropna()
        dataframe.columns = ['t', 't+1']
        data = dataframe.values
        train, test = data[:-7], data[-7:]
        train_X, train_y = train[:, 0], train[:, 1]
        test_X, test_y = test[:, 0], test[:, 1]

        best_rmse, optimal_lag = float('inf'), None

        for lag in range(1, 32):
            model = AutoReg(train_y, lags=lag).fit()
            predictions = model.predict(start=len(train_y), end=len(train_y) + len(test_y) - 1, dynamic=False)
            rmse = sqrt(mean_squared_error(test_y, predictions))
            print(f'lag: {lag}, Test RMSE: {rmse:.3f}')
            
            if rmse < best_rmse:
                best_rmse, optimal_lag = rmse, lag

        print(f'Optimal Lag Order for {column}: {optimal_lag}, RMSE: {best_rmse:.3f}')
        
#%% Fit optimal AutoReg Model
for column in power_consumption_zones:
    if column in series.columns:
        print(f"\n--- Fitting Optimal AutoReg Model for {column} ---")

        values = series[[column]].dropna().values
        dataframe = pd.concat([series[[column]].shift(1), series[[column]]], axis=1).dropna()
        dataframe.columns = ['t', 't+1']
        data = dataframe.values
        train, test = data[:-7], data[-7:]
        train_X, train_y = train[:, 0], train[:, 1]
        test_X, test_y = test[:, 0], test[:, 1]

        optimal_model = AutoReg(train_y, lags=optimal_lag).fit()

        optimal_predictions = optimal_model.predict(start=len(train_y), end=len(train_y) + len(test_y) - 1, dynamic=False)

        optimal_rmse = sqrt(mean_squared_error(test_y, optimal_predictions))
        print(f'Test RMSE for Optimal AutoReg Model ({column}): {optimal_rmse:.3f}')
        
#%% Plot Results
for column in power_consumption_zones:
    if column in series.columns:
        print(f"\n--- Plotting Results for {column} ---")

        values = series[[column]].dropna().values
        dataframe = pd.concat([series[[column]].shift(1), series[[column]]], axis=1).dropna()
        dataframe.columns = ['t', 't+1']
        data = dataframe.values
        train, test = data[:-7], data[-7:]
        train_X, train_y = train[:, 0], train[:, 1]
        test_X, test_y = test[:, 0], test[:, 1]

        optimal_model = AutoReg(train_y, lags=optimal_lag).fit()

        optimal_predictions = optimal_model.predict(start=len(train_y), end=len(train_y) + len(test_y) - 1, dynamic=False)

        plt.figure(figsize=(10, 5))
        plt.plot(test_y, label='Expected')
        plt.plot(optimal_predictions, color='red', label='Predicted')
        plt.title(f'Actual vs Predicted (AutoReg Model) - {column}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        

#%% Auto ARIMA model with Hyperparameter Tuning and calculate RMSE
results = {}

for column in power_consumption_zones:
    if column in series.columns:
        print(f"\n--- Auto ARIMA Hyperparameter Tuning and ARIMA Model for {column} ---")
        
        # first 1000 rows because otherwise I get a Memory overload Error
        data = series[[column]].dropna().values.flatten()[:1000]

        model = auto_arima(data, start_p=1, max_p=1, start_d=1, max_d=1, start_q=1, max_q=1,
                           m=12, seasonal=True, trace=True, 
                           error_action='ignore', suppress_warnings=True, n_jobs=-1)
        
        order = model.order
        print(f"Optimal ARIMA order for {column}: {order}")

        # Fit ARIMA model
        arima_model = ARIMA(data, order=order)
        arima_fit = arima_model.fit()

        predictions = arima_fit.predict(start=1, end=len(data) + 12)

        rmse = np.sqrt(mean_squared_error(data[1:], predictions[:len(data) - 1]))
        print(f"RMSE for ARIMA Model ({column}): {rmse:.3f}")

        results[column] = {
            "data": data,
            "predictions": predictions,
            "rmse": rmse
        }

        gc.collect()

#%% Plot ARIMA predictions
for column in power_consumption_zones:
    if column in results:
        raw_data = series[[column]].dropna().iloc[:1000]
        data = raw_data.values.flatten()
        index = raw_data.index
        predictions = results[column]["predictions"]

        plt.figure(figsize=(10, 4))
        plt.plot(index, data, label='Actual ' + column)
        plt.plot(index, predictions[:len(data)], color='red', label='ARIMA Predictions')
        plt.title(f'ARIMA Prediction vs Actual for {column}')
        plt.xlabel('Time')
        plt.ylabel('Power Consumption')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
#%% Fit the SARIMA model for Zone 3

data = series['Zone 3  Power Consumption'][:1000].dropna()

sarima_model = auto_arima(
    data,
    seasonal=True,
    m=12,
    start_p=0, max_p=2,
    start_q=0, max_q=2,
    start_P=0, max_P=2,
    start_Q=0, max_Q=2,
    d=None, D=1,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

print(f"Optimal SARIMA order: {sarima_model.order} seasonal_order: {sarima_model.seasonal_order}")

# Fit final SARIMA model
sarimax_model = SARIMAX(
    data,
    order=sarima_model.order,
    seasonal_order=sarima_model.seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarimax_fit = sarimax_model.fit()

pred = sarimax_fit.get_prediction(start=0, end=999)
rmse = np.sqrt(mean_squared_error(data, pred.predicted_mean))
print(f"SARIMA RMSE for Zone 3: {rmse:.4f}")

#%%Plot SARIMA prediction

forecast_steps = 12
pred = sarimax_fit.get_prediction(start=0, end=999 + forecast_steps)
pred_mean = pred.predicted_mean
ci = pred.conf_int()

time_index = series.index[:1000 + forecast_steps]
actual_values = series['Zone 3  Power Consumption'][:1000 + forecast_steps]

plt.figure(figsize=(12, 5))
plt.plot(time_index[:1000], actual_values[:1000], label='Actual Zone 3 Power Consumption')
plt.plot(time_index, pred_mean, color='red', label='SARIMA Predictions')
plt.fill_between(time_index, ci.iloc[:, 0], ci.iloc[:, 1], color='pink', alpha=0.3, label='95% Confidence Interval')

plt.title('SARIMA Forecast vs Actual for Zone 3 Power Consumption')
plt.xlabel('Time')
plt.ylabel('Power Consumption')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% PCA 

df = series.iloc[:, 0:-3]

pca = PCA()
pca.fit(df)

print("Principal Components:")
print(pca.components_)

# Display Scree plot
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(df.columns) + 1), pca.explained_variance_ratio_, marker='o', linestyle='-')
plt.title('Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.show()

# Explained variance ratio table
explained_variance_ratio_table = pd.DataFrame({
    'Component': np.arange(1, len(df.columns) + 1),
    'Explained Variance Ratio': pca.explained_variance_ratio_
})
print("\nExplained Variance Ratio Table:")
print(explained_variance_ratio_table)

# Find important components (95% cumulative variance threshold)
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
# Use 95% as the threshold variance
num_important_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"\nNumber of important components: {num_important_components}")

#%% ARIMA after PCA

X = series.iloc[:, :5]  # First 5 are: temperature, humidity, wind speed, general diffuse flow, diffuse flow (all features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA with 2 components
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)

# Add PCA components to the dataframe
series['PC1'] = pca_components[:, 0]
series['PC2'] = pca_components[:, 1]

# fit ARIMA model
for zone in power_consumption_zones:
    print(f"\n--- ARIMA for {zone} ---")
    
    subset = series[zone].dropna().values[:1000]

    model = auto_arima(subset, start_p=1, max_p=2, start_q=1, max_q=2,
                       start_d=1, max_d=1, seasonal=False, trace=True,
                       error_action='ignore', suppress_warnings=True)

    order = model.order
    print(f"Optimal ARIMA order for {zone}: {order}")

    arima_model = ARIMA(subset, order=order)
    arima_fit = arima_model.fit()

    predictions = arima_fit.predict(start=1, end=len(subset)+12)

    rmse = np.sqrt(mean_squared_error(subset[1:], predictions[:len(subset)-1]))
    print(f"RMSE for ARIMA Model ({zone}): {rmse:.3f}")

    plt.figure(figsize=(10, 4))
    plt.plot(subset, label='Actual ' + zone)
    plt.plot(predictions[:len(subset)], color='red', label='ARIMA Predictions')
    plt.title(f'ARIMA Prediction vs Actual for {zone} after PCA')
    plt.xlabel('Time Steps')
    plt.ylabel('Power Consumption')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    gc.collect()
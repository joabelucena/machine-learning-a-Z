# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:06:20 2018

@author: Joabe
"""

# Multiple Linear Regression
#%%
## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
## Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#%%
## Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, -1] = labelencoder_X.fit_transform(X[:, -1])

onehotencoder = OneHotEncoder(categorical_features = [-1])
X = onehotencoder.fit_transform(X).toarray()

#%%
## Avoiding the Dummy Variable Trap
# This is not necessary because Python's Regression library
# already takes care of this but, this is just for showing
# how we do this.
X = X[:, 1:]

#%%
## Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#%%
## Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#%%
## Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#%%
## Predicting the Test set results
y_pred = regressor.predict(X_test)

#%%
## Building the optimal model using Backward Elimination method
import statsmodels.formula.api as sm
# StatsModels doesn't assume beta0. So, we have to add a column of ones to 
# associate with the constant 0.
# So, in this point we're appending on a array of ones, our X matrix in order
# to the ones becomes first and work as beta0 X
X = np.append(arr = np.ones(shape = (50, 1)).astype(int), values = X, axis = 1)
#%%
# Manual Backward Elimination process
# It 1: Full variables
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#%%
# It 2: Minus x1 and x2 (Dummy Variables for States)
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#%%
# It 3: Minus actual x1
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#%%
# It 4: Minus actual x1
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#%%
# It 5: Minus actual x1
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# After running backward elimination, the remaining variable was "R & D Spend".
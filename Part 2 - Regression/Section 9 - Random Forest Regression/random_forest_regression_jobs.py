# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 23:20:08 2018

@author: Joabe
"""

# Random Forest Random Forest Regression

#%%
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
# Importing the dataset
dataset = pd.read_csv('Position_salaries.csv')
X = dataset.iloc[:, 1:2].values # 1:2 for X be considered as a matrix
y = dataset.iloc[:, -1:].values

#%%
# Splitting the dataset into the Training set and Test set
# Very small dataset, doesn't make sense split it
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

#%%
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#%%
## Fitting the Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X, y)


#%%
## Predict a new result with Random Forest Regression
y_pred = regressor.predict(6.5)


#%%
## Visualising the Random Forest Regression results (for higher resolution and smoother curve)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1) #reshapes the array into a matrix
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

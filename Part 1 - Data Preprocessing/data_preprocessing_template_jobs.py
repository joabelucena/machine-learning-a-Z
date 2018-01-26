# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Data.csv")
# .iloc[<lines_index_to_retrieve> , <columns_index_to_retrieve>]

X = dataset.iloc[:, :-1].values # Independent variables (Country, Age and Salary)
y = dataset.iloc[:, -1].values # Dependent variables (Purchased)

# taking care of missing data - from missing_data.py
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:])
X[:,1:] = imputer.transform(X[:, 1:])

# Encoding categorical data
# 1 - Just replaces the values for numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

# 2 - Creates dummy variables for each category
# Is necessary to convert String for numbers using LabelEncoder before
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Now we create the dummy columns using OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()


# Splittin the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split

# seeding (random_state parameter) means that you will have a pre determined set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

'''
    ## Important Point here ##
    from docs:
    fit() - Compute the mean and std to be used for later scaling.
    transform() - Perform standardization by centering and scaling.
    fit_transform() - fit() + transform()
    
    As fit() method coputes mean and std, it must be applied on training set only which
    has more observations. So test set must only be transformed with mean and std computed
    on training set.
'''
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
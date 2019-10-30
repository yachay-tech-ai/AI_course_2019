# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../Datasets/regresion_multiple.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
columnTransformer = ColumnTransformer([('oh_enc',OneHotEncoder(sparse=False),[3])],remainder='passthrough')
X = columnTransformer.fit_transform(X)
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test, y_pred)
print(f"The RMSE is {rmse}")

import statsmodels.api as sm
X = np.concatenate((np.ones((50,1)),X),axis=1).astype(float)

X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(y, X_opt).fit() #ordinary least squares
regressor_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

# Backward Elimination function
def backwardElimination(X, sl):
    while len(X) != 0:
        regressor_OLS = sm.OLS(y, X).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            index = np.argmax(regressor_OLS.pvalues)
            X = np.delete(X, index, 1)
        else:
            break
    return X
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

X_train, X_test, y_train, y_test = train_test_split(X_Modeled, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
print(f"The RMSE is {rmse}")
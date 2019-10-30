# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../Datasets/regresion_simple.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regresor = LinearRegression()
regresor.fit(X_train, y_train)

y_pred = regresor.predict(X_test)

from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test, y_pred)
print(f"The RMSE is {rmse}")

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regresor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regresor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
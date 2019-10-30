# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

dataset = pd.read_csv("../Datasets/data.csv")
X = dataset.iloc[:,:3].values
y = dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X[:,[1,2]] = imputer.fit_transform(X[:,[1,2]])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
columTransformer = ColumnTransformer([("oh_enc", OneHotEncoder(), [0])], remainder="passthrough")
X = columTransformer.fit_transform(X)
labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
stadardScaler = StandardScaler()
X_train = stadardScaler.fit_transform(X_train)
X_test = stadardScaler.transform(X_test)

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 00:50:54 2022

@author: LEO
"""
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Creating our dataset
dataset = pd.read_csv("50_Startups.csv")

# Selecting our x features and y vector
x = dataset.iloc[:, :4].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
x[:, 3] = le.fit_transform(x[:, 3])

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = "passthrough")
x = ct.fit_transform(x)

# Avoid the dummy variable trap
x = x[:, 1:].astype(np.float64)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_

# Creating our regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict Y
y_pred = regressor.predict(x_test)
y_pred1 = regressor.predict(x_train)

# Plotting our graph for Y train against Y pred
plt.scatter(y_pred1, y_train, color = "red")
plt.ylabel("Actual Y")
plt.xlabel("Predicted Y")
plt.show()


# -*- coding: utf-8 -*-
"""
Created on Sat May 28 09:35:25 2022

@author: LEO
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Create the regressor that fits the POLYNOMIAL FEATURES
from sklearn.preprocessing import PolynomialFeatures
regressor_POLY = PolynomialFeatures(degree = 4)
x_POLY = regressor_POLY.fit_transform(x)

# Create your Linear Regression that fits X_POLY and Y
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_POLY, y)

# Create a list to serve our smooth graph
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(900, 1)

# Predicting X_POLY
y_pred = regressor.predict(x_POLY)

# Plotting your graph
plt.scatter(x, y, color = "red")
plt.plot(x_grid, regressor.predict(regressor_POLY.fit_transform(x_grid)), color = "green")
plt.show()

# Getting a Salary at a point on our graph
salary = regressor.predict(regressor_POLY.fit_transform([[6.5]]))
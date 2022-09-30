# -*- coding: utf-8 -*-
"""
Created on Sat May 28 09:36:07 2022

@author: LEO
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor
regressor_DT = DecisionTreeRegressor(random_state = 0)
regressor_DT.fit(x, y)

y_pred = regressor_DT.predict(x)
salary = regressor_DT.predict([[6.5]])

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(900, 1)

plt.scatter(x, y, color = "red")
plt.plot(x_grid, regressor_DT.predict(x_grid), color = "green")
plt.show()
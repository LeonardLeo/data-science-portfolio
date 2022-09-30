# -*- coding: utf-8 -*-
"""
Created on Sun May 29 23:42:43 2022

@author: LEO
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.ensemble import RandomForestRegressor
regressor_RF = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor_RF.fit(x, y)

y_pred = regressor_RF.predict(x)
salary = regressor_RF.predict([[6.5]])

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(900, 1)

plt.scatter(x, y, color = "red")
plt.plot(x_grid, regressor_RF.predict(x_grid), color = "green")
plt.show()
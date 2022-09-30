# -*- coding: utf-8 -*-
"""
Created on Sat May 28 09:35:49 2022

@author: LEO
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = y.reshape(x.shape)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor_SVR = SVR(kernel = "rbf")
regressor_SVR.fit(x, y)

y_pred = sc_y.inverse_transform(regressor_SVR.predict(x))
salary = sc_y.inverse_transform(regressor_SVR.predict(sc_x.transform([[6.5]])))

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)

plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = "red")
plt.plot(sc_x.inverse_transform(x_grid), sc_y.inverse_transform(regressor_SVR.predict(x_grid)), color ="blue")
plt.show()
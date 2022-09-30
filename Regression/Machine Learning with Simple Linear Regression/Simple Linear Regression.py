# -*- coding: utf-8 -*-
"""
Created on Sat May 28 09:18:31 2022

@author: LEO
"""
# Importing our libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Creating our dataset
dataset = pd.read_csv("Salary_Data.csv")

# Creating the x features and y vector
x = dataset.iloc[:, :1].values
y = dataset.iloc[:, 1].values

# Splitting our data into a train set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Building the regression model
from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
Regressor.fit(x_train, y_train)

# Predicting the estimated y_train and y_test
y_pred1 = Regressor.predict(x_train)
y_pred2 = Regressor.predict(x_test)

# Visualizing the training data
plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, y_pred1, c = "green")
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the test data
plt.scatter(x_test, y_test, color = "red")
plt.plot(x_test, y_pred2, c = "green")
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
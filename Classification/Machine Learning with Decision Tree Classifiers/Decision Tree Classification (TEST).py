# -*- coding: utf-8 -*-
"""
Created on Sat May 28 09:35:05 2022

@author: LEO
"""
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Creating our dataset
dataset = pd.read_csv("Data.csv")

# Selecting our x features and y vector
x = dataset.iloc[:, :3].values
y = dataset.iloc[:, 3].values

# Turning categorical variables to dummy variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
x[:, 0] = le.fit_transform(x[:, 0])

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = "passthrough")
x = ct.fit_transform(x)

# Filling empty values with mean stategy
from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values = np.nan, strategy = "mean")
x = impute.fit_transform(x)

# Avoiding the dummy variable trap
x = x.astype(np.float64)
x = x[:, 1:]

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 1)
x = pca.fit_transform(x)
explained_variance = pca.explained_variance_ratio_

# Building our classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
classifier.fit(x, y)

y_pred = classifier.predict(x)

from sklearn.metrics import confusion_matrix, classification_report
metric = confusion_matrix(y, y_pred)
print(metric)
print(classification_report(y, y_pred))

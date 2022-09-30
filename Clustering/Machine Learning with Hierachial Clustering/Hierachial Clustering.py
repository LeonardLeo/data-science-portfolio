# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:30:42 2022

@author: LEO
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, 3:5].values

plt.scatter(x[:, 0], x[:, 1])
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Consumer Income Vs Spending Score (Without Analysis)")
plt.show()

# (1)
# The first thing to note is that Hierachial Clustering is of 2 types
# AGGLOMERATIVE and DIVISIVE 
import scipy.cluster.hierarchy as sch
link = sch.linkage(x, "ward")
dendrogram = sch.dendrogram(link)
plt.show()

# (2)
# Create a LINK for the dendrogram using SCH
# Create a DENDROGRAM from SCH using LINK
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 5, linkage = "ward")

# (3)
# Predicting x
y_pred = cluster.fit_predict(x)

# (4)
# Plot the scatter diagram
plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], color = "red", label = "Disciplined")
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], color = "purple", label = "Standard")
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], color = "orange", label = "Target")
plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], color = "blue", label = "Reckless")
plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], color = "green", label = "Sensible")
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Consumers Income Vs Spending Score")
plt.legend()
plt.show()
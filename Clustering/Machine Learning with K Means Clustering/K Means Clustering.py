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

# We plot the Elbow Diagram first
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11, 1):
    cluster = KMeans(n_clusters = i, init = "k-means++", random_state = 0)
    cluster.fit(x)
    a = cluster.inertia_
    wcss.append(a)
print(wcss)

plt.plot(range(1, 11, 1), wcss)
plt.show()

# Now we fit our derived number of clusters to X
cluster = KMeans(n_clusters = 5, init = "k-means++", random_state = 0)
cluster.fit(x)

# We then predict the category/grouping in which each person falls
y_pred = cluster.predict(x)

plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s = 10, c = "green", label = "")
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s = 10, c = "red", label = "")    
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], s = 10, c = "brown", label = "")  
plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], s = 10, c = "blue", label = "")    
plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], s = 10, c = "pink", label = "")
plt.scatter(cluster.cluster_centers_[:, 0], cluster.cluster_centers_[:, 1], s = 40, label = "Centroids")
plt.legend()
plt.show()
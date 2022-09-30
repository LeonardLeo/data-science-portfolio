# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 11:54:30 2022

@author: LEO
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/4, random_state = 0)

# Building our classifier
from sklearn.svm import SVC
classifier = SVC(kernel = "poly", degree = 5, random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the results of our classifier
y_pred = classifier.predict(x_test)

# Creating our Confusion Matrix to evaluate model performance
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calculating the accuracy and error rate from the confusion matrix
def accuracy_rate():
    a = cm[0, 1]
    b = cm[1, 0]
    c = cm.sum()
    d = (a + b)/c
    e = 1 - d
    d = str(d)
    e = str(e)
    print("Error rate = " + d)
    print("Accuray rate = " + e)
    print("")

# Calculating the recall and precision
def recall_precision():
    a = cm[0, 0]
    b = cm[0, 0] + cm[0, 1]
    c = cm[0, 0] + cm[1, 0]
    d = a/b
    e = a/c
    d = str(d)
    e = str(e)
    print("Recall = " + d)
    print("Precision = " + e)
    print("")

# Evaluating the models performance
acc = accuracy_rate()
recall_precision = recall_precision()
    
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
mean = accuracies.mean()
std = accuracies.std()
print("KFold Evaluation")
print("Mean accuracy under KFold Cross Validation = " + str(mean))
print("Standard Deviation of accuracies under KFold Cross Validation = " + str(std))


# Visualizing the training set
from matplotlib.colors import ListedColormap
x_set = x_train
y_set = y_train

x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                      np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))

plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), 
              alpha = 0.25, cmap = ListedColormap(("red", "green")))

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], 
                c = ListedColormap(("red", "green"))(i), label = j)
    
plt.legend()
plt.show()


# Visualizing the test set
x_set = x_test
y_set = y_test

x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01), 
                      np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))

plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), 
              cmap = 'Accent_r', alpha = 0.25)

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], label = j, 
                c = ListedColormap(("blue", "brown"))(i))

plt.legend()
plt.show()
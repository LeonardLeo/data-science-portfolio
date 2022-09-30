# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 19:31:25 2022

@author: LEO
"""
# Upper Confidence Bound (UCB)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Implementing UCB
import math
N = 10000
d = 10
ad_selected = []
total_reward = 0
sum_of_reward = [0]*10
number_of_selection = [0]*10

for n in range(0, N):
    ad = 0
    max_UCB = 0
    for j in range(0, d):
        if number_of_selection[j] > 0:
            average = sum_of_reward[j]/number_of_selection[j]
            deviation = math.sqrt(3/2 * math.log(n + 1)/number_of_selection[j])
            UCB = average + deviation
        else:
            UCB = 1e400
        if UCB > max_UCB:
            max_UCB = UCB
            ad = j
    ad_selected.append(ad)
    reward = dataset.values[n, ad]
    number_of_selection[ad] = number_of_selection[ad] + 1
    sum_of_reward[ad] = sum_of_reward[ad] + reward
    total_reward = total_reward + reward
 
# Visualising the results
plt.hist(ad_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
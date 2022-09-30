# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:49:44 2022

@author: LEO
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Implementing random sampling
import random
N = 10000
d = 10
ad_selected = []
total_reward = 0
sum_of_reward = [0]*d
number_of_selection = [0]*d

for n in range(0, N):
    ad = random.randrange(0, d)
    ad_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        sum_of_reward[ad] = sum_of_reward[ad] + 1
    number_of_selection[ad] = number_of_selection[ad] + 1
    total_reward = total_reward + reward

# Visualizing the results - Histogram 
plt.hist(ad_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
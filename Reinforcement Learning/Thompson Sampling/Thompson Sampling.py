# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 19:31:25 2022

@author: LEO
"""
# Thompson Sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Implementing Thompson Sampling
import random
N = 10000
d = 10
ad_selected = []
total_reward = 0
number_of_reward_1 = [0]*10
number_of_reward_0 = [0]*10

for n in range(0, N):
    ad = 0
    max_random = 0
    for j in range(0, d):
        random_beta = random.betavariate(number_of_reward_1[j] + 1, number_of_reward_0[j] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = j
    ad_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        number_of_reward_1[ad] = number_of_reward_1[ad] + 1
    else:
        number_of_reward_0[ad] = number_of_reward_0[ad] + 1
    total_reward = total_reward + reward

# Visualising the results - Histogram
plt.hist(ad_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
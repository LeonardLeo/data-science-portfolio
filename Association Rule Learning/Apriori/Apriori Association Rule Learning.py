# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 21:09:02 2022

@author: LEO
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)

transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results     
result = list(rules)
list_result = [list(result[i][0]) for i in range(0, len(result))]






























# # CREATE A LISTTTTTTTTTTTT where you'll arrange all the transactions
# transactions = []

# for i in range(0, 7501):
#     transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# from apyori import apriori
# rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, max_length = 2)

# # result = list(rules)
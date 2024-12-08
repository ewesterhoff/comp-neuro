import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt


# Load the data
task = 'PDMa'
pickle_file1 = f'data/{task}_training_results.pkl'
pickle_file2 = f'{task}_training_results.pkl'

with open(pickle_file1, 'rb') as f:
    data1 = pickle.load(f)
with open(pickle_file2, 'rb') as f:
    data2 = pickle.load(f)
data = data1 + data2

# Transform the data into a list of dictionaries
rows = []
for i in range(len(data)):
    params = data[i][0]
    print(params)
    if 'dim_rings' not in params:
        params['dim_rings'] = 13
    if 'hidden_layer' not in params:
        params['hidden_layer'] = 100
    data[i] = (params, data[i][1])

with open(pickle_file1, 'wb') as f:
    pickle.dump(data, f)
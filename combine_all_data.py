import os
from scipy.io import loadmat
import pickle
import numpy as np


# Load the data
task = 'PDMa'
pickle_file = f'combined_data/{task}_training_results.pkl'

pickle_file0 = f'temp_data0/{task}_training_results.pkl'
pickle_file1 = f'temp_data1/{task}_training_results.pkl'
pickle_file2 = f'temp_data2/{task}_training_results.pkl'
pickle_file3 = f'temp_data3/{task}_training_results.pkl'

new_data = []
if task == 'PDMa':

    data_grid = loadmat('grid_search_results_consolidated.mat')
    keys = list(data_grid.keys())
    for key in keys[3:]:
        parts = key.split('_')

        e_prop = int(parts[0][5:]) 
        density = int(parts[1][7:]) 
        params = {
                'e_prop': e_prop,
                'density': density,
                'dim_ring': 11,
                'hidden_size': 100,
                'graph_type': 'ws',
                'ii_conn': 0,
            }
        
        results = {'W': data_grid[key][0][0][0],
            'evals': data_grid[key][0][0][1],
            'evecs': data_grid[key][0][0][2],
            'performance': data_grid[key][0][0][3][0],
            } 
        
        new_data.append((params, results))

    data_network = loadmat('network_sweep_results.mat')
    keys = list(data_network.keys())
    for key in keys[3:]:
        parts = key.split('_')

        density = int(parts[1])
        dim_ring = int(parts[3])
        hidden_size = int(parts[5])
        params = {
                'e_prop': 0.8,
                'density': density,
                'dim_ring': dim_ring,
                'hidden_size': hidden_size,
                'graph_type': 'ws',
                'ii_conn': 0,
            }
        
        results = {'W': data_network[key][0][0][0],
            'evals': data_network[key][0][0][1],
            'evecs': data_network[key][0][0][2],
            'performance': data_network[key][0][0][3][0],
            } 

        new_data.append((params, results))


    data_network_full = loadmat('network_sweep_results_full.mat')
    keys = list(data_network.keys())
    for key in keys[3:]:
        parts = key.split('_')

        density = int(parts[1])
        dim_ring = int(parts[3])
        hidden_size = int(parts[5])
        params = {
                'e_prop': 0.8,
                'density': density,
                'dim_ring': dim_ring,
                'hidden_size': hidden_size,
                'graph_type': 'ws',
                'ii_conn': 0,
            }
        
        results = {'W': data_network[key][0][0][0],
            'evals': data_network[key][0][0][1],
            'evecs': data_network[key][0][0][2],
            'performance': data_network[key][0][0][3][0],
            } 
        
        new_data.append((params, results))


if os.path.exists(pickle_file0):
    with open(pickle_file0, 'rb') as f:
        data0 = pickle.load(f)
else:
    data0 = []
if os.path.exists(pickle_file1):
    with open(pickle_file1, 'rb') as f:
        data1 = pickle.load(f)
else:
    data1 = []
if os.path.exists(pickle_file2):
    with open(pickle_file2, 'rb') as f:
        data2 = pickle.load(f)
else:
    data2 = []
data_list = new_data + data0 + data1 + data2
if os.path.exists(pickle_file3):
    with open(pickle_file3, 'rb') as f:
        data3 = pickle.load(f)
else:
    data3 = []
data_list = new_data + data0 + data1 + data2 + data3


for i, data in enumerate(data_list):
    params = data[0]
    results = data[1]

    if 'dim_rings' in params:
        del params['dim_rings']
    if 'hidden_layer' in params:
        del params['hidden_layer']

    if 'dim_ring' not in params:
        params['dim_ring'] = 13
    if 'hidden_size' not in params:
        params['hidden_size'] = 100

    data_list[i] = (params, results)

unique_data = []
deleted = 0
for i, data1 in enumerate(data_list):
    is_unique = True
    for j in range(i + 1, len(data_list)):
        data2 = data_list[j]

        try:
            if data1[0] == data2[0] and data1[1]['performance'] == data2[1]['performance'] and \
                    np.array_equal(data1[1]['W'], data2[1]['W']) and \
                    np.array_equal(data1[1]['evals'], data2[1]['evals']) and \
                    np.array_equal(data1[1]['evecs'], data2[1]['evecs']):
                is_unique = False
                deleted += 1
                break
        except Exception as e:
            # If an exception occurs, print data1, data2, and the exception message
            print("An error occurred:")
            print("data1:", data1)
            print("data2:", data2)
            print("Error:", e)
    if is_unique:
        unique_data.append(data1)


print(f"num deleted: {deleted}")

for data in data_list:
    print(data[0])
print(len(data_list))
os.makedirs('combined_data', exist_ok=True)
with open(pickle_file, 'wb') as f:
    pickle.dump(data_list, f)
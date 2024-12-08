from scipy.io import loadmat
import pickle

# data = loadmat('network_sweep_results.mat')
# data = loadmat('network_sweep_results_full.mat')
data = loadmat('grid_search_results.mat')

print(data)

keys = list(data.keys())

# Loop through the keys starting from the 4th key (index 3)
new_data = []
for key in keys[3:]:

    print(key)

    parts = key.split('_')

    e_prop = int(parts[1])  # "0"
    density = int(parts[3])      # "2"
    graph_type = str(parts[5])  # "50"
    ii_conn = int(parts[8])  # "50"
    params = {
            'e_prop': e_prop,
            'density': density,
            'dim_ring': 11,
            'hidden_size': 100,
            'graph_type': graph_type,
            'ii_conn': ii_conn,
        }
    
    results = {'W': data[key][0][0][0],
        'evals': data[key][0][0][1],
        'evecs': data[key][0][0][2],
        'performance': data[key][0][0][3][0],
        } 
    
    print(results)

    new_data.append((params, results))

pickle_file = f'combined_data/PDMa_training_results.pkl'

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

data = data + new_data

import os
os.makedirs('combined_data', exist_ok=True)
# with open('combined_data/PDMa_training_results.pkl', 'wb') as f:
#     pickle.dump(data, f)
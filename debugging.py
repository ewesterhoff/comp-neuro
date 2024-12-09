#%%
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Load the data
task = 'PDMa'
pickle_file = f'combined_data/{task}_training_results.pkl'
figname = f'results/{task}_results'

os.makedirs('results', exist_ok=True)

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

# Transform the data into a list of dictionaries
rows = []
for params, results in data:
    row = {
        'e_prop': params['e_prop'],
        'density': params['density'],
        'dim_ring': params['dim_ring'],
        'hidden_size': params['hidden_size'],
        'graph_type': params['graph_type'],
        'ii_conn': params['ii_conn'],
        'performance': results['performance'],
    }
    rows.append(row)

# Create a DataFrame
df = pd.DataFrame(rows)

# Lists of unique values in each column
e_props_list = df['e_prop'].unique().tolist()
densities_list = df['density'].unique().tolist()
graph_types_list = df['graph_type'].unique().tolist()
ii_conns_list = df['ii_conn'].unique().tolist()
dim_ring_list = sorted(df['dim_ring'].unique().tolist())
hidden_size_list = df['hidden_size'].unique().tolist()
# %%
densityfiltered_df = df[df['density'] == 1]
eprop_filtered_df = densityfiltered_df[densityfiltered_df['e_prop'] == 1]
iiconn_filtered_df = eprop_filtered_df[eprop_filtered_df['ii_conn'] == 1]


print(iiconn_filtered_df)
         
grouped_df = iiconn_filtered_df.groupby(['hidden_size', 'dim_ring', 'graph_type']).agg({'performance': 'mean'}).reset_index()

print(grouped_df)
# %%

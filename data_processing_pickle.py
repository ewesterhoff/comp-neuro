import pickle
import pandas as pd
import matplotlib.pyplot as plt


# Load the data
pickle_file = 'PDMa_training_results.pkl'
figname = 'PDMa_results'

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

# Transform the data into a list of dictionaries
rows = []
for params, results in data:
    row = {
        'e_prop': params['e_prop'],
        'density': params['density'],
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

for density in densities_list:
    # Filter rows where the density matches the current density value
    density_df = df[df['density'] == density]
    
    # Group by 'e_prop', 'graph_type', and 'ii_conn' to calculate the average performance
    grouped_df = density_df.groupby(['e_prop', 'graph_type', 'ii_conn']).agg({'performance': 'mean'}).reset_index()

    # Create the plot for the current density
    plt.figure(figsize=(8, 6))

    # Loop through each combination of graph_type and ii_conn
    for graph_type in graph_types_list:
        for ii_conn in ii_conns_list:
            # Filter the data for the current graph_type and ii_conn
            subset_df = grouped_df[(grouped_df['graph_type'] == graph_type) & (grouped_df['ii_conn'] == ii_conn)]

            # Set the line style and color based on graph_type and ii_conn
            if graph_type == 'er':
                color = 'red'
            else:  # graph_type == 'ws'
                color = 'blue'

            line_style = '-' if ii_conn == 1 else '--'  # Solid for ii_conn=1, dotted for ii_conn=0

            # Plot the data
            plt.plot(subset_df['e_prop'], subset_df['performance'], label=f'{graph_type}, ii_conn={ii_conn}', 
                     color=color, linestyle=line_style, marker='o')

    # Customize the plot
    plt.title(f'Density = {density}')
    plt.xlabel('e_prop')
    plt.ylabel('Performance')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.savefig(f"{figname}_density{density}.png")

for e_prop in e_props_list:
    # Filter rows where the e_prop matches the current e_prop value
    e_prop_df = df[df['e_prop'] == e_prop]
    
    # Group by 'density', 'graph_type', and 'ii_conn' to calculate the average performance
    grouped_df = e_prop_df.groupby(['density', 'graph_type', 'ii_conn']).agg({'performance': 'mean'}).reset_index()

    # Create the plot for the current e_prop
    plt.figure(figsize=(8, 6))

    # Loop through each combination of graph_type and ii_conn
    for graph_type in graph_types_list:
        for ii_conn in ii_conns_list:
            # Filter the data for the current graph_type and ii_conn
            subset_df = grouped_df[(grouped_df['graph_type'] == graph_type) & (grouped_df['ii_conn'] == ii_conn)]

            # Set the line style and color based on graph_type and ii_conn
            if graph_type == 'er':
                color = 'red'
            else:  # graph_type == 'ws'
                color = 'blue'

            line_style = '-' if ii_conn == 1 else '--'  # Solid for ii_conn=1, dotted for ii_conn=0

            # Plot the data
            plt.plot(subset_df['density'], subset_df['performance'], label=f'{graph_type}, ii_conn={ii_conn}', 
                     color=color, linestyle=line_style, marker='o')

    # Customize the plot
    plt.title(f'e_prop = {e_prop}')
    plt.xlabel('density')
    plt.ylabel('Performance')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.savefig(f"{figname}_e_prop{e_prop}.png")
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Load the data
task = 'PDMa'
pickle_file = f'combined_data/{task}_training_results.pkl'
figname = f'results/{task}/{task}_results'

os.makedirs(f'results/{task}', exist_ok=True)

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

# i = len(data)
# while i > 0:
#     i -= 1
#     datai = data[i]
#     results = datai[1]
#     performance = results['performance']

#     if performance < 0.01:
#         data.pop(i)


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

cmap = cm.get_cmap('viridis', len(dim_ring_list))
color_mapping = {dim_ring: cmap(i) for i, dim_ring in enumerate(dim_ring_list)}



for density in densities_list:
    densityfiltered_df = df[df['density'] == density]
    for e_prop in e_props_list:
        epropfiltered_df = densityfiltered_df[densityfiltered_df['e_prop'] == e_prop]
        for ii_conn in ii_conns_list:
            iiconnfiltered_df = epropfiltered_df[epropfiltered_df['ii_conn'] == ii_conn]
            
            grouped_df = iiconnfiltered_df.groupby(['hidden_size', 'dim_ring', 'graph_type']).agg({'performance': 'mean'}).reset_index()


            num_data_points = 0
            plt.figure(figsize=(8, 6))

            for graph_type in graph_types_list:
                for dim_ring in dim_ring_list:
                    subset_df = grouped_df[(grouped_df['graph_type'] == graph_type) & (grouped_df['dim_ring'] == dim_ring)]
                    if len(subset_df) < 2:
                        continue
                    num_data_points += len(subset_df)

                    # Set the line style and marker based on graph_type
                    if graph_type == 'er':
                        line_style = '--'
                        marker = "^"
                    elif graph_type == 'ws':
                        line_style = '-'
                        marker = "o"

                    color = color_mapping[dim_ring]

                    # Plot the data
                    plt.plot(subset_df['hidden_size'], subset_df['performance'], label=f'{graph_type}, dim_ring={dim_ring}', 
                            color=color, linestyle=line_style, marker=marker)

            # Customize the plot
            plt.title(f'Density = {density}, E Proportion = {e_prop}, I-I Connections = {ii_conn}')
            plt.xlabel('Num Neurons')
            plt.ylabel('Performance')
            plt.legend()
            plt.grid(True)

            # Show the plot
            if num_data_points > 5:
                plt.savefig(f"{figname}_density{density}_e_prop{e_prop}_ii_conn{ii_conn}.png")
            plt.close('all')


for hidden_size in hidden_size_list:
    hiddensizefiltered_df = df[df['hidden_size'] == hidden_size]
    for e_prop in e_props_list:
        epropfiltered_df = hiddensizefiltered_df[hiddensizefiltered_df['e_prop'] == e_prop]
        for ii_conn in ii_conns_list:
            iiconnfiltered_df = epropfiltered_df[epropfiltered_df['ii_conn'] == ii_conn]
            
            grouped_df = iiconnfiltered_df.groupby(['density', 'dim_ring', 'graph_type']).agg({'performance': 'mean'}).reset_index()


            num_data_points = 0
            plt.figure(figsize=(8, 6))

            for graph_type in graph_types_list:
                for dim_ring in dim_ring_list:
                    subset_df = grouped_df[(grouped_df['graph_type'] == graph_type) & (grouped_df['dim_ring'] == dim_ring)]
                    if len(subset_df) < 2:
                        continue
                    num_data_points += len(subset_df)

                    # Set the line style and marker based on graph_type
                    if graph_type == 'er':
                        line_style = '--'
                        marker = "^"
                    elif graph_type == 'ws':
                        line_style = '-'
                        marker = "o"

                    color = color_mapping[dim_ring]

                    # Plot the data
                    plt.plot(subset_df['density'], subset_df['performance'], label=f'{graph_type}, dim_ring={dim_ring}', 
                            color=color, linestyle=line_style, marker=marker)

            # Customize the plot
            plt.title(f'Num Neurons = {hidden_size}, E Proportion = {e_prop}, I-I Connections = {ii_conn}')
            plt.xlabel('Density')
            plt.ylabel('Performance')
            plt.legend()
            plt.grid(True)

            # Show the plot
            if num_data_points > 5:
                plt.savefig(f"{figname}_hidden_size{hidden_size}_e_prop{e_prop}_ii_conn{ii_conn}.png")
            plt.close('all')

for density in densities_list:
    densityfiltered_df = df[df['density'] == density]
    for hidden_size in hidden_size_list:
        hiddensizefiltered_df = densityfiltered_df[densityfiltered_df['hidden_size'] == hidden_size]
        for ii_conn in ii_conns_list:
            iiconnfiltered_df = hiddensizefiltered_df[hiddensizefiltered_df['ii_conn'] == ii_conn]
            
            grouped_df = iiconnfiltered_df.groupby(['e_prop', 'dim_ring', 'graph_type']).agg({'performance': 'mean'}).reset_index()


            num_data_points = 0
            plt.figure(figsize=(8, 6))

            for graph_type in graph_types_list:
                for dim_ring in dim_ring_list:
                    subset_df = grouped_df[(grouped_df['graph_type'] == graph_type) & (grouped_df['dim_ring'] == dim_ring)]
                    if len(subset_df) < 2:
                        continue
                    num_data_points += len(subset_df)

                    # Set the line style and marker based on graph_type
                    if graph_type == 'er':
                        line_style = '--'
                        marker = "^"
                    elif graph_type == 'ws':
                        line_style = '-'
                        marker = "o"

                    color = color_mapping[dim_ring]

                    # Plot the data
                    plt.plot(subset_df['e_prop'], subset_df['performance'], label=f'{graph_type}, dim_ring={dim_ring}', 
                            color=color, linestyle=line_style, marker=marker)

            # Customize the plot
            plt.title(f'Density = {density}, Num Neurons = {hidden_size}, I-I Connections = {ii_conn}')
            plt.xlabel('e_prop')
            plt.ylabel('Performance')
            plt.legend()
            plt.grid(True)

            # Show the plot
            if num_data_points > 5:
                plt.savefig(f"{figname}_density{density}_hidden_size{hidden_size}_ii_conn{ii_conn}.png")
            plt.close('all')









# Now switch graph type and ii connections
for density in densities_list:
    densityfiltered_df = df[df['density'] == density]
    for e_prop in e_props_list:
        epropfiltered_df = densityfiltered_df[densityfiltered_df['e_prop'] == e_prop]
        for graph_type in graph_types_list:
            graphtypefiltered_df = epropfiltered_df[epropfiltered_df['graph_type'] == graph_type]
            
            grouped_df = graphtypefiltered_df.groupby(['hidden_size', 'dim_ring', 'ii_conn']).agg({'performance': 'mean'}).reset_index()


            num_data_points = 0
            plt.figure(figsize=(8, 6))

            for ii_conn in ii_conns_list:
                for dim_ring in dim_ring_list:
                    subset_df = grouped_df[(grouped_df['ii_conn'] == ii_conn) & (grouped_df['dim_ring'] == dim_ring)]
                    if len(subset_df) < 2:
                        continue
                    num_data_points += len(subset_df)

                    # Set the line style and marker based on ii_conn
                    if ii_conn == 0:
                        line_style = '--'
                        marker = "^"
                    elif ii_conn == 1:
                        line_style = '-'
                        marker = "o"

                    color = color_mapping[dim_ring]

                    # Plot the data
                    plt.plot(subset_df['hidden_size'], subset_df['performance'], label=f'ii_conn={ii_conn}, dim_ring={dim_ring}', 
                            color=color, linestyle=line_style, marker=marker)

            # Customize the plot
            plt.title(f'Density = {density}, E Proportion = {e_prop}, Graph Type = {graph_type}')
            plt.xlabel('Num Neurons')
            plt.ylabel('Performance')
            plt.legend()
            plt.grid(True)

            # Show the plot
            if num_data_points > 5:
                plt.savefig(f"{figname}_density{density}_e_prop{e_prop}_graphtype{graph_type}.png")
            plt.close('all')


for hidden_size in hidden_size_list:
    hiddensizefiltered_df = df[df['hidden_size'] == hidden_size]
    for e_prop in e_props_list:
        epropfiltered_df = hiddensizefiltered_df[hiddensizefiltered_df['e_prop'] == e_prop]
        for graph_type in graph_types_list:
            graphtypefiltered_df = epropfiltered_df[epropfiltered_df['graph_type'] == graph_type]
            
            grouped_df = graphtypefiltered_df.groupby(['density', 'dim_ring', 'ii_conn']).agg({'performance': 'mean'}).reset_index()


            num_data_points = 0
            plt.figure(figsize=(8, 6))

            for ii_conn in ii_conns_list:
                for dim_ring in dim_ring_list:
                    subset_df = grouped_df[(grouped_df['ii_conn'] == ii_conn) & (grouped_df['dim_ring'] == dim_ring)]
                    if len(subset_df) < 2:
                        continue
                    num_data_points += len(subset_df)

                    # Set the line style and marker based on ii_conn
                    if ii_conn == 0:
                        line_style = '--'
                        marker = "^"
                    elif ii_conn == 1:
                        line_style = '-'
                        marker = "o"

                    color = color_mapping[dim_ring]

                    # Plot the data
                    plt.plot(subset_df['density'], subset_df['performance'], label=f'ii_conn={ii_conn}, dim_ring={dim_ring}', 
                            color=color, linestyle=line_style, marker=marker)

            # Customize the plot
            plt.title(f'Num Neurons = {hidden_size}, E Proportion = {e_prop}, Graph Type = {graph_type}')
            plt.xlabel('Density')
            plt.ylabel('Performance')
            plt.legend()
            plt.grid(True)

            # Show the plot
            if num_data_points > 5:
                plt.savefig(f"{figname}_hidden_size{hidden_size}_e_prop{e_prop}_graphtype{graph_type}.png")
            plt.close('all')

for density in densities_list:
    densityfiltered_df = df[df['density'] == density]
    for hidden_size in hidden_size_list:
        hiddensizefiltered_df = densityfiltered_df[densityfiltered_df['hidden_size'] == hidden_size]
        for graph_type in graph_types_list:
            graphtypefiltered_df = hiddensizefiltered_df[hiddensizefiltered_df['graph_type'] == graph_type]
            
            grouped_df = graphtypefiltered_df.groupby(['e_prop', 'dim_ring', 'ii_conn']).agg({'performance': 'mean'}).reset_index()


            num_data_points = 0
            plt.figure(figsize=(8, 6))

            for ii_conn in ii_conns_list:
                for dim_ring in dim_ring_list:
                    subset_df = grouped_df[(grouped_df['ii_conn'] == ii_conn) & (grouped_df['dim_ring'] == dim_ring)]
                    if len(subset_df) < 2:
                        continue
                    num_data_points += len(subset_df)

                    # Set the line style and marker based on graph_type
                    if ii_conn == 0:
                        line_style = '--'
                        marker = "^"
                    elif ii_conn == 1:
                        line_style = '-'
                        marker = "o"

                    color = color_mapping[dim_ring]

                    # Plot the data
                    plt.plot(subset_df['e_prop'], subset_df['performance'], label=f'ii_conn={ii_conn}, dim_ring={dim_ring}', 
                            color=color, linestyle=line_style, marker=marker)

            # Customize the plot
            plt.title(f'Density = {density}, Num Neurons = {hidden_size}, Graph Type = {graph_type}')
            plt.xlabel('e_prop')
            plt.ylabel('Performance')
            plt.legend()
            plt.grid(True)

            # Show the plot
            if num_data_points > 5:
                plt.savefig(f"{figname}_density{density}_hidden_size{hidden_size}_graphtype{graph_type}.png")
            plt.close('all')
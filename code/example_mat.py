import osmnx as ox
import matplotlib.pyplot as plt

# Use the 'Agg' backend for headless environments
import matplotlib
matplotlib.use('Agg')

# Get a street network for a place name
G = ox.graph_from_place('Piedmont, California, USA', network_type='drive')

# Plot the street network
fig, ax = ox.plot_graph(G, show=False, close=False)

# Save the plot to a file
fig.savefig('street_network.png', format='png', dpi=300)

# Optionally close the figure to free up memory
plt.close(fig)


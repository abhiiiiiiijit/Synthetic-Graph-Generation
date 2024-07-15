import osmnx as ox

# Get a street network for a place name
G = ox.graph_from_place('Piedmont, California, USA', network_type='drive')

# Plot the street network
ox.plot_graph(G)


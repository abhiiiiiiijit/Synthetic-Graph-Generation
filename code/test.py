# import torch
# from torch_geometric.data import Data

# # Example existing node features (4 nodes with 3 features each)
# existing_features = torch.randn(4, 3)

# # Example latitude and longitude data
# latitude = torch.tensor([37.7749, 34.0522, 40.7128, 51.5074])
# longitude = torch.tensor([-122.4194, -118.2437, -74.0060, -0.1278])

# # Reshape latitude and longitude
# latitude = latitude.view(-1, 1)
# longitude = longitude.view(-1, 1)

# # Concatenate new features to existing ones
# updated_features = torch.cat((existing_features, latitude, longitude), dim=1)

# # Create a dummy edge index (fully connected for 4 nodes)
# edge_index = torch.tensor([[0, 1, 2, 3, 0, 1, 2],
#                            [1, 2, 3, 0, 2, 3, 0]], dtype=torch.long)

# # Create the Data object
# data = Data(x=updated_features, edge_index=edge_index)

# print("Updated Node Features:\n", data.x)
# print("Edge Index:\n", data.edge_index)

# import csv

# # Open the CSV file
# with open('./data/city_lat_long.csv', mode='r') as file:
#     # Create a CSV reader object
#     csv_reader = csv.reader(file)
    
#     # print(type(csv_reader))
#     # # Iterate over each row in the CSV file
#     for row in csv_reader:
#         print(type(row))

import networkx as nx
import pickle

# # Example: Creating a list of graphs
# G1 = nx.Graph()
# G1.add_edges_from([(1, 2), (2, 3)])

# G2 = nx.Graph()
# G2.add_edges_from([(3, 4), (4, 5)])

# graphs = [G1, G2]

# # Save the list of graphs to a file
# with open("./data/networkx_cities_graph/graphs.pkl", "wb") as f:
#     pickle.dump(graphs, f)

# with open("./data/networkx_cities_graph/cities_graphs.pkl", "rb") as f:
#     loaded_graphs = pickle.load(f)

# # Now you can use the loaded graphs
# # G1_loaded = loaded_graphs[0]
# # G2_loaded = loaded_graphs[1]

# print(len(loaded_graphs))
# import osmnx as ox
# from pyproj import Proj, transform
# # Define the latitude, longitude, and distance (in meters)
# lat, lon = 52.5200, 13.4050  # Example coordinates (Berlin, Germany)
# distance = 500  # 500 meters

# # Get the graph for the given location
# G = ox.graph_from_point((lat, lon), dist=distance, network_type='all')

# G_proj = ox.project_graph(G)



# # Define the projection (e.g., UTM zone for Berlin, which is zone 33U)
# utm_proj = Proj(proj="utm", zone=33, datum="WGS84")

# # Iterate over nodes to transform coordinates
# for node, data in G.nodes(data=True):
#     x, y = utm_proj(data['x'], data['y'])  # Project lon, lat to x, y
#     G.nodes[node]['x'] = x
#     G.nodes[node]['y'] = y

# for node, data in G_proj.nodes(data=True):
#     print(f"Node {node}: x={data['x']}, y={data['y']}")

import osmnx as ox
import networkx as nx
import numpy as np

# Step 1: Get the graph and project it
lat, lon = 52.5200, 13.4050  # Example coordinates (Berlin, Germany)
distance = 500  # 500 meters
G = ox.graph_from_point((lat, lon), dist=distance, network_type='all')
G_proj = ox.project_graph(G)  # Project to UTM

# Step 2: Extract x and y coordinates
x_values = np.array([data['x'] for node, data in G_proj.nodes(data=True)])
y_values = np.array([data['y'] for node, data in G_proj.nodes(data=True)])

# Step 3: Normalize the coordinates between 0 and 1
x_min, x_max = x_values.min(), x_values.max()
y_min, y_max = y_values.min(), y_values.max()

x_norm = (x_values - x_min) / (x_max - x_min)
y_norm = (y_values - y_min) / (y_max - y_min)

# Step 4: Update the graph with normalized coordinates
for i, (node, data) in enumerate(G_proj.nodes(data=True)):
    data['x_norm'] = x_norm[i]
    data['y_norm'] = y_norm[i]

# Now each node has 'x_norm' and 'y_norm' as normalized coordinates between 0 and 1
for node, data in G_proj.nodes(data=True):
    print(f"Node {node}: x={data['x_norm']}, y={data['y_norm']}")
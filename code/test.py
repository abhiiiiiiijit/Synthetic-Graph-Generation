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
# from pyproj import Proj, 
# # Define the latitude, longitude, and distance (in meters)
# lat, lon = 52.5200, 13.4050  # Example coordinates (Berlin, Germany)
# distance = 500  # 500 meters

# # Get the graph for the given location
# G = ox.graph_from_point((lat, lon), dist=distance, network_type='all')

# G_proj = ox.project_graph(G)



# # Define the projection (e.g., UTM zone for Berlin, which is zone 33U)
# utm_proj = Proj(proj="utm", zone=33, datum="WGS84")

# # Iterate over nodes to  coordinates
# for node, data in G.nodes(data=True):
#     x, y = utm_proj(data['x'], data['y'])  # Project lon, lat to x, y
#     G.nodes[node]['x'] = x
#     G.nodes[node]['y'] = y

# for node, data in G_proj.nodes(data=True):
#     print(f"Node {node}: x={data['x']}, y={data['y']}")

# import osmnx as ox
# import networkx as nx
# import numpy as np

# # Step 1: Get the graph and project it
# lat, lon = 52.5200, 13.4050  # Example coordinates (Berlin, Germany)
# distance = 500  # 500 meters
# G = ox.graph_from_point((lat, lon), dist=distance, network_type='all')
# G_proj = ox.project_graph(G)  # Project to UTM

# # Step 2: Extract x and y coordinates
# x_values = np.array([data['x'] for node, data in G_proj.nodes(data=True)])
# y_values = np.array([data['y'] for node, data in G_proj.nodes(data=True)])

# # Step 3: Normalize the coordinates between 0 and 1
# x_min, x_max = x_values.min(), x_values.max()
# y_min, y_max = y_values.min(), y_values.max()

# x_norm = (x_values - x_min) / (x_max - x_min)
# y_norm = (y_values - y_min) / (y_max - y_min)

# # Step 4: Update the graph with normalized coordinates
# for i, (node, data) in enumerate(G_proj.nodes(data=True)):
#     data['x_norm'] = x_norm[i]
#     data['y_norm'] = y_norm[i]

# # Now each node has 'x_norm' and 'y_norm' as normalized coordinates between 0 and 1
# for node, data in G_proj.nodes(data=True):
#     print(f"Node {node}: x={data['x_norm']}, y={data['y_norm']}")

# import osmnx as ox
# import geopandas as gpd
# from shapely.geometry import Point
# import numpy as np
# # Define the central point and the distance (in meters)
# point = (37.7749, -122.4194)  # Example: San Francisco, CA
# distance = 500  # meters

# # Get the bounding box coordinates
# bbox = ox.utils_geo.bbox_from_point(point, dist=distance)

# # bbox = ox.project_gdf(bbox)
   
# # Unpack the bounding box coordinates
# north, south, east, west = bbox

# # Project each corner of the bounding box to UTM
# west_south_utm = (west, south)
# east_south_utm = ( east, south)
# east_north_utm = ( east, north)
# west_north_utm = ( west, north)

# print(west_south_utm,east_north_utm)

# # utm_coords_list = [north, south, east, west]
# utm_coords_list = [west_south_utm, east_north_utm,west_north_utm,east_south_utm]

# # Define the UTM CRS (for example, UTM zone 10N)
# utm_crs = "EPSG:3857"
# # Retrieve the graph from the point
# G_proj = ox.graph_from_point(point, dist=distance, network_type='drive')

# G_proj = ox.project_graph(G_proj, to_crs='epsg:3857') 

# # Step 2: Extract x and y coordinates
# x_values = np.array([data['x'] for node, data in G_proj.nodes(data=True)])
# y_values = np.array([data['y'] for node, data in G_proj.nodes(data=True)])

# # Step 3: Normalize the coordinates between 0 and 1
# x_min, x_max = x_values.min(), x_values.max()
# y_min, y_max = y_values.min(), y_values.max()



# # Create Point objects for each UTM tuple
# points = [Point(coords) for coords in utm_coords_list]

# # Create a GeoDataFrame from the list of Point objects with the UTM CRS
# gdf = gpd.GeoDataFrame(index=range(len(points)), crs=utm_crs, geometry=points)


# bbox2 = ox.project_gdf(gdf)

# from pyproj import Proj, transform

# # Define the EPSG:4326 (WGS84) and EPSG:3857 (Web Mercator) projections
# wgs84 = Proj(init='epsg:4326')  # Lat/Long
# web_mercator = Proj(init='epsg:3857')  # Web Mercator

# # Example coordinates (longitude, latitude)
# lon, lat = -122.42508885165161, 37.77939660167747 # New York City

# # Transform the coordinates to EPSG:3857
# x, y = transform(wgs84, web_mercator, lon, lat)


# # # Display the GeoDataFrame
# print(G_proj.nodes(data=True))
# # print(G)
# print(utm_coords_list)
# print(f"Projected coordinates: x={x}, y={y}")
# # print(bbox2)
# print(x_max,x_min,y_max,y_min)

# # Print the coordinates
# print(f"North: {north}, South: {south}, East: {east}, West: {west}")



# import osmnx as ox
# import matplotlib.pyplot as plt

# # Use the 'Agg' backend for headless environments
# import matplotlib
# matplotlib.use('Agg')

# # Get a street network for a place name
# G = ox.graph_from_place('Piedmont, California, USA', network_type='drive')

# # Plot the street network
# fig, ax = ox.plot_graph(G, show=False, close=False)

# # Save the plot to a file
# fig.savefig('street_network.png', format='png', dpi=300)

# # Optionally close the figure to free up memory
# plt.close(fig)

# import torch

# def replace_top_x_with_1_ignore_diag(mat, x):
#     # Clone the original matrix to avoid in-place modifications
#     result = torch.zeros_like(mat)
    
#     # Create a mask to ignore diagonal elements
#     diag_mask = torch.eye(mat.size(0), mat.size(1), device=mat.device).bool()
    
#     # Apply the mask to set diagonal elements to -inf, so they are not considered
#     masked_mat = mat.masked_fill(diag_mask, float('-inf'))
    
#     # Get the top x indices along each row ignoring diagonal
#     top_x_indices = torch.topk(masked_mat, x, dim=1).indices
    
#     # Scatter 1s into the result tensor at the top x indices
#     result.scatter_(1, top_x_indices, 1)
    
#     return result

# # Example usage
# mat = torch.tensor([[0.1, 0.3, 0.6, 0.9],
#                     [0.5, 0.2, 0.7, 0.1],
#                     [0.8, 0.6, 0.4, 0.2],
#                     [0.9, 0.5, 0.3, 0.98]])

# x = 2  # Number of highest values to replace with 1
# result = replace_top_x_with_1_ignore_diag(mat, x)
# print(result)

import torch

# Example tensor with n points in d-dimensional space
coords = torch.tensor([
    [0.1, 0.2],
    [0.4, 0.4],
    [0.8, 0.9]
])  # This is a (3, 2) tensor for 3 points in 2D space

# Compute pairwise distances
# (n, d) -> (n, 1, d) and (1, n, d) to perform broadcasting subtraction
diffs = coords.unsqueeze(1) - coords.unsqueeze(0)
  
# Square the differences, sum over the coordinate dimensions, and take the square root
dist_matrix = torch.sqrt(torch.sum(diffs**2, dim=-1))

print(dist_matrix)

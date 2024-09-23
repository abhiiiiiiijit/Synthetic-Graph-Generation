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

# import torch

# # Example tensor with n points in d-dimensional space
# coords = torch.tensor([
#     [0.1, 0.2],
#     [0.4, 0.4],
#     [0.8, 0.9]
# ])  # This is a (3, 2) tensor for 3 points in 2D space

# # Compute pairwise distances
# # (n, d) -> (n, 1, d) and (1, n, d) to perform broadcasting subtraction
# diffs = coords.unsqueeze(1) - coords.unsqueeze(0)
  
# # Square the differences, sum over the coordinate dimensions, and take the square root
# dist_matrix = torch.sqrt(torch.sum(diffs**2, dim=-1))

# # print(dist_matrix)

# # import networkx as nx
# # import matplotlib.pyplot as plt
# # import osmnx as ox
# # import random

# # # Step 1: Define node coordinates
# # nodes = {
# #     0: (0, 0),
# #     1: (1, 0),
# #     2: (1, 1),
# #     3: (0, 1),
# # }

# # # Step 2: Define edges with node indices
# # edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]

# # # Step 3: Create a NetworkX graph
# # G = nx.Graph()
# # for node, coord in nodes.items():
# #     G.add_node(node, x=coord[0], y=coord[1])

# # # Step 4: Randomly generate edge geometries
# # for u, v in edges:
# #     # Random edge shape: straight line, or a line with a midpoint
# #     if random.random() > 0.5:
# #         # Straight line
# #         geom = [(nodes[u][0], nodes[u][1]), (nodes[v][0], nodes[v][1])]
# #     else:
# #         # Line with a midpoint (random perturbation)
# #         mid_x = (nodes[u][0] + nodes[v][0]) / 2 + random.uniform(-0.1, 0.1)
# #         mid_y = (nodes[u][1] + nodes[v][1]) / 2 + random.uniform(-0.1, 0.1)
# #         geom = [(nodes[u][0], nodes[u][1]), (mid_x, mid_y), (nodes[v][0], nodes[v][1])]

# #     # Add edge with geometry
# #     G.add_edge(u, v, geometry=geom)

# # # Step 5: Set the CRS to a generic value
# # G.graph['crs'] = 'epsg:3857'
# import networkx as nx
# import pickle

# import osmnx as ox
# import matplotlib.pyplot as plt
# from gen_syn_graph import visualise_graph


# # with open("./data/networkx_cities_graph/ccs_cities_graphs.pkl", "rb") as f:
# #     l_netx_cities = pickle.load(f)

# # G = l_netx_cities[0]

# # for u, v, attrs in G.edges(data=True):
# #     # Access edge attributes:

# #     # weight = attrs.get('weight', None)  # Default to None if weight doesn't exist
# #     # color = attrs.get('color', None)  # Default to None if color doesn't exist
# #     print(f"Edge: ({u}, {v}), attributes={attrs['geometry']}")

# # for node, data in G.edges(data=True):
# #     print(f"Node {node}: {data}")
# # Step 6: Plot the graph using OSMNX
# # fig, ax = ox.plot_graph(G, show=False, close=False)

# # # Optionally, show the plot
# # plt.show()
# import osmnx as ox
# import networkx as nx
# import matplotlib.pyplot as plt

# # Step 1: Define a location (latitude, longitude) and set the distance
# point = (51.233334, 6.783333)#(37.7749, -122.4194)  # Example coordinates (San Francisco, CA)
# distance = 500  # in meters

# # Step 2: Download the graph from OSMnx
# G = ox.graph_from_point(point, dist=distance, network_type='drive')

# # Step 3: Project the graph to EPSG:3857
# G = ox.project_graph(G, to_crs='EPSG:3857')

# G = ox.convert.to_undirected(G)

# # Step 3: Remove all edge attributes except 'geometry'
# # Create a new edge attribute dictionary containing only 'geometry'
# for u, v, k in G.edges(keys=True):
#     geometry = G.edges[u, v, k].get('geometry')
#     G[u][v][k].clear() 
#     # Update the edge data to retain only the geometry attribute
#     nx.set_edge_attributes(G, {(u, v, k): {'geometry': geometry}})

# # for n, a in G.nodes(data=True):
# #     print(a)

# for u, v, attrs in G.edges(data=True):
#     # Access edge attributes:

#     # weight = attrs.get('weight', None)  # Default to None if weight doesn't exist
#     # color = attrs.get('color', None)  # Default to None if color doesn't exist
#     print(f"Edge: ({u}, {v}), attributes={attrs}")

# # Step 4: Plot the graph
# fig, ax = ox.plot_graph(G, show=False, close=False)
# plt.show()
# import pickle

# distance = 500
# country = "Germany"
# out_feat_dim = 16
# pyg_version = 1 
# pyg_file_path = f'./data/tg_graphs/{country}_pyg_graphs_d_{distance}_v_{pyg_version}.pkl'
# pyg_file_path2 = "./data/tg_graphs/tg_graphs_all.pkl"

# with open(pyg_file_path, "rb") as f:
#     data = pickle.load(f)

# with open(pyg_file_path2, "rb") as f:
#     data2 = pickle.load(f)


# print(data.x)
# print(data2.x)
import pandas as pd
from geopy.geocoders import Nominatim
import time
import osmnx as ox
import matplotlib.pyplot as plt
import csv
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import pickle
from pyproj import Transformer
# from pyproj import Proj, transform
import numpy as np
write_networkx_graph = False
write_ll = False
# normalise_coordinate = False
write_pyg_graph = True
distance = 500
precision = 2
country = "Germany"
base_file_path = "./data/"
pyg_version = 1 # 1= wo_edge



def agg_all_graph(g_list):
    data1 = remove_edge_features(g_list[0])
    # for u, v, key in data1.edges(keys=True):
    #     data1.clear() 
    data1 = from_networkx(data1)
    data1 = add_edge_attr(data1)
    
    for i in range(1, len(g_list)):
        data2 = remove_edge_features(g_list[i])
        data2 = from_networkx(data2)
        data2 = add_edge_attr(data2)
        # for u, v, key in data2.edges(keys=True):
        #     data2.clear() 
        x = torch.cat([data1.x, data2.x], dim=0)
        edge_features = torch.cat([data1.edge_attr, data2.edge_attr], dim=0)
        edge_index = torch.cat([data1.edge_index, data2.edge_index + data1.num_nodes], dim=1)
        data1 = Data(x=x, edge_index=edge_index, edge_attr = edge_features)
    return data1

def add_edge_attr(pyg_data):
    x = pyg_data.x
    num_nodes = x.size(0)
    distances = torch.cdist(x, x, p=2)

    # Convert to edge index format (if you're using a fully connected graph, otherwise use your own edge indices)
    edge_index = pyg_data.edge_index # Assuming fully connected
    edge_features = distances[edge_index[0], edge_index[1]]  # Extract distances for edges
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_features)
    return data
def remove_edge_features(G):
    for u, v, key in G.edges(keys=True):
        G[u][v][key].clear() 
    # with open("./data/networkx_cities_graph/ccs_cities_graphs_wo_edge_a.pkl", "wb") as f:
    #      pickle.dump(graph_list, f)
    return G


w_city_nx_file_path = base_file_path + f'networkx_cities_graph/{country}_ccs_cities_nx_graphs_d_{distance}.pkl'
w_city_pyg_file_path = base_file_path + f'tg_graphs/{country}_pyg_graphs_d_{distance}_v_{pyg_version}.pkl'
if write_pyg_graph:
    with open(w_city_nx_file_path, "rb") as f:
        l_netx_cities = pickle.load(f)
    # print(l_netx_cities[0].edges(data=True)) 
    data = agg_all_graph(l_netx_cities)
    print(data.x)
    print(data.edge_index)
    print(data.edge_attr)
    # x = pyg_data.x
    # num_nodes = x.size(0)
    # distances = torch.cdist(x, x, p=2)

    # # Convert to edge index format (if you're using a fully connected graph, otherwise use your own edge indices)
    # edge_index = pyg_data.edge_index # Assuming fully connected
    # edge_features = distances[edge_index[0], edge_index[1]]  # Extract distances for edges
    # data = Data(x=x, edge_index=edge_index, edge_attr=edge_features)
    # print(pyg_data.edge_index.size())
    # print(pyg_data.edge_attr.size())
    # with open(w_city_pyg_file_path, "wb") as f:
    #     pickle.dump(pyg_data, f)
    # if bool(pyg_data):
    #     print("pyg section working fine")
else:
    with open(w_city_pyg_file_path, "rb") as f:
        pyg_data = pickle.load(f)
    print(pyg_data.x)
    # print(l_netx_cities[0].edges(data=True))
    if bool(pyg_data):
            print(f"pyg graphs {pyg_version} section of {country} cities is working fine")
    else:
        print(f"Something is wrong with {country} pyg graphs {pyg_version}")
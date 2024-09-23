# # # import osmnx as ox

# # # # Get a street network for a place name
# # # G = ox.graph_from_place('Piedmont, California, USA', network_type='drive')

# # # # Plot the street network
# # # ox.plot_graph(G)
# # import chardet

# # # Read the raw bytes first
# # with open('./data/12411-0015.csv', 'rb') as f:
# #     raw_data = f.read()

# # # Detect encoding
# # result = chardet.detect(raw_data)
# # encoding = result['encoding']
# # print(encoding)

# # # Now read the file with the detected encoding
# # with open('./data/12411-0015.csv', encoding=encoding) as f:
# #     content = f.read()

# # import torch
# # x = torch.rand(5, 3)
# # print(x)

# # import torch

# # # Check if CUDA is available
# # if torch.cuda.is_available():
# #     print("CUDA is available")
    
# #     # Get the number of CUDA devices
# #     num_devices = torch.cuda.device_count()
# #     print(f"Number of CUDA devices: {num_devices}")

# #     # Print the name and properties of each CUDA device
# #     for i in range(num_devices):
# #         print(f"Device {i}: {torch.cuda.get_device_name(i)}")
# #         device_properties = torch.cuda.get_device_properties(i)
# #         print(f"  Compute capability: {device_properties.major}.{device_properties.minor}")
# #         print(f"  Total memory: {device_properties.total_memory / (1024 ** 3):.2f} GB")
# # else:
# #     print("CUDA is not available")


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


# # utm_coords_list = [north, south, east, west]
# utm_coords_list = [west_south_utm, east_north_utm,west_north_utm,east_south_utm]

# # Define the UTM CRS (for example, UTM zone 10N)
# utm_crs = "EPSG:3857"
# # Retrieve the graph from the point
# G = ox.graph_from_point(point, dist=distance, network_type='drive')

# G_proj = ox.project_graph(G, to_crs='epsg:3857')

# # Step 2: Extract x and y coordinates
# x_values = np.array([data['x'] for node, data in G_proj.nodes(data=True)])
# y_values = np.array([data['y'] for node, data in G_proj.nodes(data=True)])

# # Step 3: Normalize the coordinates between 0 and 1
# x_min, x_max = x_values.min(), x_values.max()
# y_min, y_max = y_values.min(), y_values.max()


# # # Create Point objects for each UTM tuple
# # points = [Point(coords) for coords in utm_coords_list]

# # # Create a GeoDataFrame from the list of Point objects with the UTM CRS
# # gdf = gpd.GeoDataFrame(index=range(len(points)), crs=utm_crs, geometry=points)


# # bbox2 = ox.project_gdf(gdf)

# from pyproj import Proj, transform
# from pyproj import Transformer

# # Define the EPSG:4326 (WGS84) and EPSG:3857 (Web Mercator) projections
# # wgs84 = Proj(init='epsg:4326')  # Lat/Long
# # web_mercator = Proj(init='epsg:3857')  # Web Mercator

# # Example coordinates (longitude, latitude)
# x, y = -122.42508885165161, 37.77939660167747 # New York City

# # Transform the coordinates to EPSG:3857
# # x, y = transform(wgs84, web_mercator, lon, lat)
# # Get the bounding box coordinates
# bbox = ox.utils_geo.bbox_from_point((y,x), dist=distance)

# # bbox = ox.project_gdf(bbox)
   
# # Unpack the bounding box coordinates
# north, south, east, west = bbox

# # # Display the GeoDataFrame
# # print(G_proj.nodes(data=True))
# # print(G.nodes(data=True))
# # # print(G)
# # # print(utm_coords_list)
# # print(f"Projected coordinates: x={x}, y={y}")
# # # print(bbox2)
# # print(x_max,x_min,y_max,y_min)

# def get_ccs_of_nodes(x, y, north, south, east, west):
#     transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857",always_xy=True)
#     x_ccs, y_ccs = transformer.transform( x, y)
#     x_min, y_min = transformer.transform( west, south)
#     x_max, y_max = transformer.transform( east, north)



#     x_ccs = round((x_ccs - x_min) / (x_max - x_min),2)
#     y_ccs = round((y_ccs - y_min) / (y_max - y_min),2)

#     print(x_ccs,y_ccs,x_min,y_min,x_max,y_max)
#     return [ x_ccs, y_ccs]
#     # x_ccs, y_ccs = transform(wgs84, web_mercator, x, y)
#     # x_min, y_min = transform(wgs84, web_mercator, west, south)
#     # x_max, y_max = transform(wgs84, web_mercator, east, north)
#     # print(x_ccs,y_ccs,x_min,y_min,x_max,y_max)


# get_ccs_of_nodes(x, y, north, south, east, west)

import pandas as pd
from geopy.geocoders import Nominatim
# import time
import osmnx as ox
import matplotlib.pyplot as plt
# import csv
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx,  remove_self_loops, coalesce, to_networkx
import pickle
# from pyproj import Transformer
import torch_geometric.transforms as T
# import os.path as osp
import time
import torch
import torch_geometric.transforms as T
# from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv
import random
from sklearn.metrics import average_precision_score, roc_auc_score
import torch.nn.functional as F
from ae_model import GCNEncoder, GCNEncoder12, GATEncoder, GraphSAGEEncoder

distance = 500
country = "Germany"
out_feat_dim = 16
pyg_version = 2
pyg_file_path = f'./data/tg_graphs/{country}_pyg_graphs_d_{distance}_v_{pyg_version}.pkl'#
pyg_file_path_o = f'./data/tg_graphs/tg_graphs_all.pkl'
encoder_name = "gcn"
model_version = 1
write_model = False
base_file_path = "./data/"
nx_fp = base_file_path + f'networkx_cities_graph/{country}_ccs_cities_nx_graphs_d_{distance}.pkl'
nx_fp_o = base_file_path + f'networkx_cities_graph/ccs_cities_graphs_wo_edge_a.pkl'
##################Load data#########################################################
# with open(nx_fp, "rb") as f:
#     data = pickle.load(f)

# with open(nx_fp_o, "rb") as f:
#     data_o = pickle.load(f)

# def count_edges(data):
#     count = 0
#     for d in data:
#         count += len(d.edges(data=True))
#     return count

# print(count_edges(data[:-2]))
# print(count_edges(data_o))
# print(len(data[0].edges(data=True)))
# print(len(data_o[0].edges(data=True)))
# print(data[0].edges(data=True))
# print(data_o[0].edges.size())


with open(pyg_file_path, "rb") as f:
    data = pickle.load(f)

with open(pyg_file_path_o, "rb") as f:
    data_o = pickle.load(f)

print(data.x)
print(data.edge_index)
print(data.edge_attr)
# print(data.x.size())
# print(data_o.x.size())

# print(data.edge_index.size())
# print(data_o.edge_index.size())
import pandas as pd
from geopy.geocoders import Nominatim
import time
import osmnx as ox
import matplotlib.pyplot as plt

def main():
    #get the city population data > 100k
    df_city_p = get_cities_w_pop_gt_100k()

    print(df_city_p.shape)


def get_cities_w_pop_gt_100k():
    df_city_p = pd.read_csv('./data/Germany_cities_pop.csv',delimiter=';'
                            ,encoding='ISO-8859-1',header=None)
    
    #drop other columns, only take the 2023 data
    df_city_p = df_city_p.drop(columns=[0,2,3,4,5])

    #change column name
    col_name = {1:'city', 6:'population'}

    df_city_p.rename(columns=col_name,inplace=True)

    #remove - from population column
    df_city_p = df_city_p[df_city_p['population']!='-']

    # change data type of population to int
    df_city_p['population'] = df_city_p['population'].astype('Int64')

    #Consider only the cities with population more than 100000
    df_city_p = df_city_p[df_city_p['population']>=100000]

    # extract only the city name
    df_city_p['city'] = df_city_p['city'].str.split(',',expand=True)[0]

    return df_city_p


if __name__ == '__main__':
    main()












# import torch
# from torch_geometric.data import Data

# def generate_dataset(num_nodes, features_dim, edge_prob, max_distance):
#   """
#   Generates a sample GNN dataset with embedded edge distance and location.

#   Args:
#       num_nodes: Number of nodes in the graph.
#       features_dim: Dimensionality of node features.
#       edge_prob: Probability of an edge existing between two nodes.
#       max_distance: Maximum distance used for edge distance embedding.

#   Returns:
#       A PyTorch Geometric Data object representing the graph.
#   """
#   # Generate random node features
#   node_features = torch.randn(num_nodes, features_dim)

#   # Create empty edge lists and edge attributes
#   edge_index = []
#   edge_attr = []

#   # Generate random edges with probability and calculate distances
#   for i in range(num_nodes):
#     for j in range(i + 1, num_nodes):
#       if torch.rand(1) < edge_prob:
#         # Calculate random distance between nodes (0 to max_distance)
#         distance = torch.rand(1) * max_distance
#         edge_index.append([i, j])
#         edge_attr.append([distance])
#     # Convert edge lists and attributes to tensors
#   edge_index = torch.tensor(edge_index, dtype=torch.long).t()
#   edge_attr = torch.tensor(edge_attr, dtype=torch.float)

#   # Generate random node locations (optional, can be replaced with actual data)
#   node_locations = torch.randn(num_nodes, 2)  # 2D location for example

#   # Combine edge distance and node locations into a single edge attribute
#   combined_edge_attr = torch.cat((edge_attr, node_locations[edge_index[1]] - node_locations[edge_index[0]]), dim=1)

#   # Create Data object with features, edge indices, and edge attributes
#   data = Data(x=node_features, edge_index=edge_index, edge_attr=combined_edge_attr)
#   return data  


    # print("Hello")
    ########################


    # # Initialize the geolocator
    # geolocator = Nominatim(user_agent="city_locator")

    # # List of cities
    # cities = ["Berlin", "Munich", "Hamburg", "Cologne", "Frankfurt"]

    # # Function to get latitude and longitude
    # def get_lat_lon(city):
    #     try:
    #         location = geolocator.geocode(city + ", Germany")
    #         if location:
    #             return (location.latitude, location.longitude)
    #         else:
    #             return None
    #     except Exception as e:
    #         print(f"Error getting coordinates for {city}: {e}")
    #         return None

    # # Iterate through the cities and get their coordinates
    # city_coords = {}
    # for city in cities:
    #     coords = get_lat_lon(city)
    #     if coords:
    #         city_coords[city] = coords
    #     time.sleep(1)  # To avoid hitting the API rate limit

    # # Print the results
    # # for city, coords in city_coords.items():
    # #     print(f"{city}: Latitude = {coords[0]}, Longitude = {coords[1]}")



    # # Define the place name
    # place_name = "Trier, Germany"

    # # Fetch the street network for Düsseldorf
    # G = ox.graph_from_place(place_name, network_type='drive')

    
    # dataset1 = generate_dataset(num_nodes=50, features_dim=10, edge_prob=0.3, max_distance=10)
    # dataset2 = generate_dataset(num_nodes=100, features_dim=16, edge_prob=0.5, max_distance=20)

    # Print information about the datasets
    # print(f"Dataset 1: Nodes: {dataset1.num_nodes}, Edges: {dataset1.num_edges}")
    # print(f"Dataset 2: Nodes: {dataset2.num_nodes}, Edges: {dataset2.num_edges}")
    # print(dataset1)

# def decode_value(value):
#     try:
#         return value.encode('utf-8').decode('utf-8')
#     except (UnicodeEncodeError, UnicodeDecodeError):
#         return value

# def get_city_from_df(value, delimiter, part_index):
#     try:
#         parts = value.split(delimiter)
#         return parts[part_index].strip() if len(parts) > part_index else None
#     except Exception as e:
#         print(f"Error processing value '{value}': {e}")
#         return None


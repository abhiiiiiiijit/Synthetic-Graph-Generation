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

with open("./data/networkx_cities_graph/graphs.pkl", "rb") as f:
    loaded_graphs = pickle.load(f)

# Now you can use the loaded graphs
G1_loaded = loaded_graphs[0]
G2_loaded = loaded_graphs[1]

print(G1_loaded)
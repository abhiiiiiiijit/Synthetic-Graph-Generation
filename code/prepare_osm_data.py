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
# cd let do it, need to import

def main():
    write_networkx_graph = False
    write_ll = False
    # normalise_coordinate = False
    write_pyg_graph = False
    distance = 1000
    precision = 4
    country = "Germany"
    base_file_path = "./data/"
    pyg_version = 2 # 1= wo_edge 2= w_edge # 1.2 nodefeat+edgefeat
##### city name list >>>>>>>>>> lat long
    #get or write Longitude Latitude
    r_city_pop_file_path = base_file_path+ country + '_cities_pop.csv'
    w_city_long_lat_file_path = f'./data/{country}_cities_lat_long.csv'
    if write_ll:
        #get the city population data > 100k
        df_city_p = get_cities_w_pop_gt_100k(r_city_pop_file_path, country)
        # city names list
        cities = df_city_p['city'].to_list()
        # write the lat long of the cities
        write_lat_long(cities, country) #suffix: '_cities_pop.csv'
        print('lat long written sucessfully')
    else: 
        pddf_lat_long = pd.read_csv(w_city_long_lat_file_path, delimiter=',',header = None)
        print(pddf_lat_long.head())
        if pddf_lat_long.shape[0]>0:
            print(f"Long and Latitude of {country} cities is working fine")
        else:
            print(f"Something is wrong with {country} long latude code")
###### lat, lon >>>>>>>> networkx
    w_city_nx_file_path = base_file_path + f'networkx_cities_graph/{country}_ccs_cities_nx_graphs_d_{distance}.pkl'
    if write_networkx_graph:
        l_netx_cities = get_networkx_data_from_coords(w_city_long_lat_file_path,w_city_nx_file_path, distance, country,precision)
        if bool(l_netx_cities):
             print(f"Sucessfully written networkx graphs section of {country} cities")
    else:
        with open(w_city_nx_file_path, "rb") as f:
            l_netx_cities = pickle.load(f)
        # print(l_netx_cities[0].nodes(data=True))
        print(l_netx_cities[0].edges(data=True))
        if bool(l_netx_cities):
             print(f"networkx graphs section of {country} cities is working fine")
        else:
            print(f"Something is wrong with {country} networkx graphs section  code")  

######  networkx >>>>>>>>>> py torch geo
    w_city_pyg_file_path = base_file_path + f'tg_graphs/{country}_pyg_graphs_d_{distance}_v_{pyg_version}.pkl'
    if write_pyg_graph:
        with open(w_city_nx_file_path, "rb") as f:
            l_netx_cities = pickle.load(f)
        if pyg_version == 1:
            pyg_data = agg_all_graph(l_netx_cities)
        elif pyg_version == 2:
            pyg_data = agg_all__pyg_graph_w_edge_attr(l_netx_cities)
        elif pyg_version == 1.2:
            pyg_data = agg_all_pyg_graphs_v12(l_netx_cities)
        with open(w_city_pyg_file_path, "wb") as f:
            pickle.dump(pyg_data, f)
        if bool(pyg_data):
            print("pyg section working fine")
    else:
        with open(w_city_pyg_file_path, "rb") as f:
            pyg_data = pickle.load(f)
        print(pyg_data.x)
        # print(l_netx_cities[0].edges(data=True))
        if bool(pyg_data):
             print(f"pyg graphs {pyg_version} section of {country} cities is working fine")
        else:
            print(f"Something is wrong with {country} pyg graphs {pyg_version}")  
    
#################################networkx >>>>>>>>>> py torch geo##########################################################

def agg_all_graph(g_list):
    data1 = remove_edge_features(g_list[0])
    # for u, v, key in data1.edges(keys=True):
    #     data1.clear() 
    data1 = from_networkx(data1)
    
    for i in range(1, len(g_list)):
        data2 = remove_edge_features(g_list[i])
        data2 = from_networkx(data2)
        
        # for u, v, key in data2.edges(keys=True):
        #     data2.clear() 
        x = torch.cat([data1.x, data2.x], dim=0)
        edge_index = torch.cat([data1.edge_index, data2.edge_index + data1.num_nodes], dim=1)
        data1 = Data(x=x, edge_index=edge_index)
    return data1

def agg_all__pyg_graph_w_edge_attr(g_list):
    data1 = remove_edge_features(g_list[0])
    data1 = from_networkx(data1)
    data1 = add_edge_attr(data1)
    data1 = Data(x=data1.x.float(), edge_index=data1.edge_index, edge_attr = data1.edge_attr.float())
    for i in range(1, len(g_list)):
        data2 = remove_edge_features(g_list[i])
        data2 = from_networkx(data2)
        data2 = add_edge_attr(data2)
        x = torch.cat([data1.x, data2.x], dim=0)
        edge_index = torch.cat([data1.edge_index, data2.edge_index + data1.num_nodes], dim=1)
        edge_features = torch.cat([data1.edge_attr, data2.edge_attr], dim=0)
        data1 = Data(x=x.float(), edge_index=edge_index, edge_attr = edge_features.float())
    return data1

def agg_all_pyg_graphs_v12(g_list):
    data1 = remove_edge_features(g_list[0])
    data1 = from_networkx(data1)
    data1 = add_edge_attr_2_nodes(data1)
    # data1 = Data(x=data1.x.float(), edge_index=data1.edge_index)
    for i in range(1, len(g_list)):
        data2 = remove_edge_features(g_list[i])
        data2 = from_networkx(data2)
        data2 = add_edge_attr_2_nodes(data2)
        x = torch.cat([data1.x, data2.x], dim=0)
        edge_index = torch.cat([data1.edge_index, data2.edge_index + data1.num_nodes], dim=1)
        pos = torch.cat([data1.pos, data2.pos], dim=0)
        data1 = Data(x=x, edge_index=edge_index, pos=pos )
    return data1

def add_edge_attr_2_nodes(pyg_data):
    x = pyg_data.x
    edge_index = pyg_data.edge_index
    # Number of nodes
    N = x.size(0)

    # Initialize a tensor to store the sum of distances and number of neighbors
    z_sum = torch.zeros(N, dtype=torch.float32)
    neighbor_count = torch.zeros(N, dtype=torch.float32)

    # Loop over all edges
    for i in range(edge_index.size(1)):
        src = edge_index[0, i]  # Source node index
        dst = edge_index[1, i]  # Destination node index
        
        # Calculate the Euclidean distance between src and dst
        dist = torch.norm(x[src] - x[dst], p=2)
        
        # Add distance to both source and destination node
        z_sum[src] += dist
        z_sum[dst] += dist
        
        # Increment neighbor count for both nodes
        neighbor_count[src] += 1
        neighbor_count[dst] += 1

    # Avoid division by zero for isolated nodes
    mean_distances = z_sum / neighbor_count
    mean_distances[neighbor_count == 0] = 0  # Set to 0 for nodes with no neighbors

        # Add the mean distance as the new feature z1
    x_new = torch.cat([x, mean_distances.unsqueeze(1)], dim=1)
    data = Data(x=x_new.float(), edge_index=edge_index, pos=x.float())
    return data

def add_edge_attr(pyg_data):
    x = pyg_data.x
    distances = torch.cdist(x, x, p=2)
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
##############################################################################
def get_ccs_of_nodes(x, y, north, south, east, west, precision):# returns list of (x, y)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857",always_xy=True)
    # x_ccs, y_ccs = transformer.transform( x, y)
    x_ccs, y_ccs = x, y
    x_min, y_min = transformer.transform( west, south)
    x_max, y_max = transformer.transform( east, north)

    #precison of 2, considering a nodes area to be 10m
    x_ccs = round((x_ccs - x_min) / (x_max - x_min),precision)
    y_ccs = round((y_ccs - y_min) / (y_max - y_min),precision)

    # print(x_ccs,y_ccs,x_min,y_min,x_max,y_max)
    return [ x_ccs, y_ccs]


def get_networkx_data_from_coords(r_file_path, w_file_path, distance, country, precision):
    
    l_netx_cities = []
    l_city_coord = []
    try:
            
        with open(r_file_path, mode='r') as file:
            cities_lat_long = csv.reader(file)
            for city in cities_lat_long:
                l_city_coord.append(city)
        for city in l_city_coord:
            try:
                lat, long, distance = round(float(city[0]), precision), round(float(city[1]), precision), distance
            # 1. Obtain the graph from OpenStreetMap

                G = ox.graph_from_point((lat, long), dist=distance, network_type='drive')
                #latitude is y, long is x in nodes

                G = ox.convert.to_undirected(G)

                # project the whole graph
                G = ox.project_graph(G, to_crs='epsg:3857')
                # Get the bounding box coordinates
                bbox = ox.utils_geo.bbox_from_point((lat, long), dist=distance)
                
                # Unpack the bounding box coordinates
                north, south, east, west = bbox
            except Exception:
                continue

            # Filter nodes and update features
            nodes_to_remove = []

            #convert the lat and longitude of the nodes as vector
            for node in G.nodes():
                node_data = G.nodes[node]
                if 'x' in list(node_data.keys()) and 'y' in list(node_data.keys()):
                    # xy_not_none = node_data['x'] is  None or node_data['y'] is  None
                    # xy_not_0 = node_data['x'] == 0 or node_data['y'] == 0
                    if not (bool(node_data['x']) or bool(node_data['y'])):
                        nodes_to_remove.append(node)
                    else:
                        G.nodes[node]['x'] = get_ccs_of_nodes(node_data['x'], node_data['y'], north, south, east, west, precision) #[ node_data['x'],node_data['y']]
                        for key in list(node_data.keys()):
                            if key != 'x':
                                del node_data[key]
                        # node_data = {'lat_long' : [node_data['x'],node_data['y']]}

                elif 'x' not in list(node_data.keys()) or 'y' not in list(node_data.keys()):
                    nodes_to_remove.append(node)
            # print(nodes_to_remove)
            G.remove_nodes_from(nodes_to_remove)

            # clear edge features
            # for u, v in G.edges():
            #     # print(G.edges)
            #     G.edges[(u,v,0)].clear()
            l_netx_cities.append(G)
            # print(G.edges)

            # pyg_data = from_networkx(G)
            # print(pyg_data.x)
    except Exception as e:
        print(f"An unexpected error occurred: {type(e)}")
        

    # return l_netx_cities
    with open(w_file_path, "wb") as f:
        pickle.dump(l_netx_cities, f)
    return l_netx_cities

# takes lot of time to get the lat_long so I have written down a csv file
def write_lat_long(cities, country):
    lat_long = get_lat_long(cities)
    file_path = f'./data/{country}_cities_lat_long.csv'
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(lat_long)

def get_lat_long(cities):
    geolocator = Nominatim(user_agent="city_locator")
    cities_w_ll = []
    for city in cities:
        try:
            location = geolocator.geocode(city + ", Germany")
            # print(location)
            if location:
                cities_w_ll.append((location.latitude, location.longitude)) 
        except Exception as e:
            print(f"Error getting coordinates for {city}: {e}")
            # return None
    return cities_w_ll

def get_cities_w_pop_gt_100k(r_file_path, country):
    df_city_p = pd.read_csv(r_file_path,delimiter=';'
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

#################################################

    # with open("./data/networkx_cities_graph/cities_graphs.pkl", "wb") as f:
    #     pickle.dump(l_netx_cities, f)
    # print(len(l_netx_cities))
    #

    # print(lat_long)
    # print(pyg_data.nodes())

    # for node in pyg_data.nodes(data=True):
    #     print(f"Node: {node[1]}")
    #     for key, value in node[1].items():
    #         print(f"  {key}: {value}")
    #         print()
    # attr_x = nx.get_node_attributes(G, 'x')
    # attr_y = nx.get_node_attributes(G, 'y')
    # attr = { k:{"x":[v,attr_y[k]]} for k,v in attr_x.items()}
    # nx.set_node_attributes(G, attr)
    # print(attr)
    # for node, data in G.nodes(data=True):
    #     lat_long = nx.get_node_attributes(node,'x')
    #     # nb = nx.get_node_attributes(G,'x')
    #     print(lat_long)
        # print(node)
        # if all(key in features for key in required_keys):
        #     updated_features.append([features[key] for key in required_keys])
        # else:
        #     nodes_to_remove.append(node)
    # print(nx.is_directed(G))
    # print(G.nodes(data=True))
    # 3. Convert NetworkX graph to PyTorch Geometric data object
    # pyg_data = from_networkx(G)
    # print(pyg_data.y)


    # for u, v in G.edges():
    #     # print(G.edges)
    #     G.edges[(u,v,0)].clear()

    # print(G.edges[(26815185, 28805947, 0)])
    # print(G.nodes(data=True))

    # 2. Extract node and edge information
    # Clean the graph
    # G = ox.utils_graph.remove_isolated_nodes(G)
    # G = ox.utils_graph.get_largest_component(G, strongly=True)

    # 3. Convert NetworkX graph to PyTorch Geometric data object

    # # Adding node features (optional)
    # # Example: Adding latitude and longitude as node features
    # coords = [G.nodes[n]['y'] for n in G.nodes], [G.nodes[n]['x'] for n in G.nodes]
    # coords_tensor = torch.tensor(list(zip(*coords)), dtype=torch.float)
    # pyg_data.x = coords_tensor
    






# def get_osm_map_1km(lat, long):
#     # Define the central point and distance around it (in meters)
#     center_point = (lat, long)  # Latitude and longitude of Düsseldorf city center
#     distance = 1000  # Distance in meters

#     # Fetch the street network within the distance from the center point
#     G = ox.graph.graph_from_point(center_point, dist=distance, network_type='drive')

#     # Plot the street network
#     fig, ax = ox.plot.plot_graph(G)
#     plt.title('Street Network within 1km of Düsseldorf City Center')
#     plt.show()















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


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

def main():
    #get the city population data > 100k
    # df_city_p = get_cities_w_pop_gt_100k()

    # city names list
    # cities = df_city_p['city'].to_list()
    # write the lat long of the cities
    #   write_lat_long(cities)

    #lets get network x data for all the graphs in a list
    # file_path = './data/city_lat_long.csv'
    # distance = 500
    # l_netx_cities = get_networkx_data_from_coords(file_path, distance)
    with open("./data/networkx_cities_graph/ccs_cities_graphs_wo_edge_a.pkl", "rb") as f:
        l_netx_cities = pickle.load(f)

    # ox.plot_graph(l_netx_cities[0])
    # print(type(l_netx_cities[0]))
    # pos = {e[0]:tuple(e[1]['x']) for e in l_netx_cities[0].nodes(data=True)}
    # nx.draw(l_netx_cities[0],pos=pos)
    # plt.show()
    # G = l_netx_cities[0]
    # l_netx_cities = remove_edge_features(l_netx_cities)
    print(l_netx_cities[0].edges(data=True))
    print(len(l_netx_cities))

def remove_edge_features(graph_list):
    for G in graph_list:
            for u, v, key in G.edges(keys=True):
                G[u][v][key].clear() 
    # with open("./data/networkx_cities_graph/ccs_cities_graphs_wo_edge_a.pkl", "wb") as f:
    #      pickle.dump(graph_list, f)
    return graph_list


if __name__ == "__main__":
    main()
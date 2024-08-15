import pandas as pd
from geopy.geocoders import Nominatim
# import time
import osmnx as ox
import matplotlib.pyplot as plt
# import csv
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import pickle
# from pyproj import Transformer
import torch_geometric.transforms as T
# import os.path as osp
import time
import torch
import torch_geometric.transforms as T
# from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv


def main():

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
  
    # l_netx_cities = remove_edge_features(l_netx_cities)
    is_variational = False
    is_linear = False
    iteration = 10

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    transform = T.Compose([
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      split_labels=True, add_negative_train_samples=False),
    ])
    G = l_netx_cities[0]
    data = from_networkx(G)
    # print(l_netx_cities[0].nodes(data=True))
    # print(g1.pos_edge_label_index)
    # print(g1.get_summary())
    train_data, val_data, test_data = transform(data)
    in_channels, out_channels = data.num_features, 16
    # print(data, train_data, val_data, test_data)

    # elif not is_variational and is_linear:
    #     model = GAE(LinearEncoder(in_channels, out_channels))
    # elif is_variational and not is_linear:
    #     model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
    # elif is_variational and is_linear:
    #     model = VGAE(VariationalLinearEncoder(in_channels, out_channels))
###################################
    if not is_variational and not is_linear:
        model = GAE(GCNEncoder(in_channels, out_channels))

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    times = []
    for epoch in range(1, iteration + 1):
        start = time.time()
        loss = train(model, optimizer, train_data, is_variational)
        auc, ap = test(test_data, model)
        print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
        times.append(time.time() - start)
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

    # save the model
    torch.save(model, './code/models/gae_model_v1.pth')

    #test the saved model
    model = torch.load('./code/models/gae_model_v1.pth')
    auc, ap = test(val_data, model)
    print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')

    # for epoch in range(1, iteration + 1):
    #     start = time.time()
    #     loss = train(model,optimizer,train_data,is_variational)
    #     auc, ap = test(test_data, model)
    #     print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
    #     times.append(time.time() - start)
    # print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
    # print(data.num_features)

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

def train(model, optimizer, train_data, is_variational):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    # print(len(z))
    # dc = torch.sigmoid( model.decode(z,train_data.edge_index))
    # print(len(dc))
    # print(dc)   
    if is_variational:
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data, model):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    # print(z)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

def remove_edge_features(graph_list):
    for G in graph_list:
            for u, v, key in G.edges(keys=True):
                G[u][v][key].clear() 
    # with open("./data/networkx_cities_graph/ccs_cities_graphs_wo_edge_a.pkl", "wb") as f:
    #      pickle.dump(graph_list, f)
    return graph_list


if __name__ == "__main__":
    main()
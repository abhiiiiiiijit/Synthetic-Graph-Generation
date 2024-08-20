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


def main():
  
    with open("./data/tg_graphs/tg_graphs_all.pkl", "rb") as f:
        data = pickle.load(f)

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

    train_data, val_data, test_data = transform(data)   
    model = torch.load('./code/models/gae_model_v1.pth')
    # auc, ap = test(test_data, model)
    # print(f' AUC: {auc:.4f}, AP: {ap:.4f}')

    #Graph coordinate positions are saved here
    data.pos = data.x


    model.eval()
    z = model.encode(data.x, data.edge_index)

    print(z.size()[0])

    no_nodes = z.size(0)
    sample_size = 10
    sampled_indices = random.sample(range(no_nodes), sample_size)

    # z_sampled_nodes = z[sampled_indices]

    new_pyg_graph = gen_new_pyg_graph(z, data, sampled_indices)

    # print(new_pyg_graph.edge_index)

    visualise_graph(new_pyg_graph)

    # print(data.x[n])
    # print(threshold)
    # print(A_prob)
    # print(pred_edges)
    # # print(data.x[0:5])
    # print(len(test_data.x))
    # #test the saved model
    # model = torch.load('./code/models/gae_model_v1.pth')
    # edge_index = test_data.pos_edge_label_index
    # model.eval()
    # z = model.encode(test_data.x, test_data.edge_index)
    # value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    # value = torch.sigmoid(value)
    # adj = torch.matmul(z, z.t())
    
    # adj = torch.sigmoid(adj)
    # pred = adj.detach().cpu().numpy()
    # pos_y = z.new_ones(test_data.pos_edge_label_index.size(1))
    # neg_y = z.new_zeros(test_data.neg_edge_label_index.size(1))
    # y = torch.cat([pos_y, neg_y], dim=0)
    # y = y.detach().cpu().numpy()

    # print(len(y),len(pred))
    # # edge_index = model.decode(z)
    # print(roc_auc_score(y, pred, multi_class='ovo'))


def visualise_graph(data):
    # Convert the PyG Data object to a NetworkX graph
    G = to_networkx(data,  to_undirected=True)

    # Visualize the graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos=data.pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=16)
    plt.show()

def gen_new_pyg_graph(z, data, sampled_indices):
    z_sampled_nodes = z[sampled_indices]

    A_prob = torch.sigmoid(torch.matmul(z_sampled_nodes, z_sampled_nodes.t()))

    threshold = 0.65 #A_prob.mean().item()
    A_pred = (A_prob > threshold).int()

    # pred_edges = from_scipy_sparse_matrix(A_pred)

    pred_edges = A_pred.nonzero(as_tuple=False) 

    # pred_edges = torch.concat(pred_edges[0],pred_edges[1],dim =1)
    # Convert pred_edges to edge_index format
    edge_index = pred_edges.t().contiguous()

    # Remove self-loops
    edge_index, _ = remove_self_loops(edge_index)

    # Sort and remove duplicate edges for undirected graph
    edge_index = coalesce(edge_index, None, num_nodes=z_sampled_nodes.size(0))[0]

    x = data.x[sampled_indices]
    pos = data.pos[sampled_indices]
    new_graph = Data(x= x, edge_index=edge_index, pos=pos)

    return new_graph


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

@torch.no_grad()
def test(data, model):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    # pos_edge_index = data.pos_edge_label_index
    # neg_edge_index = data.neg_edge_label_index
    # pos_y = z.new_ones(data.pos_edge_label_index.size(1))
    # neg_y = z.new_zeros(data.neg_edge_label_index.size(1))
    # y = torch.cat([pos_y, neg_y], dim=0)
    # pos_pred = model.decode(z, pos_edge_index, sigmoid=True)
    # neg_pred = model.decode(z, neg_edge_index, sigmoid=True)
    # pred = torch.cat([pos_pred, neg_pred], dim=0)
    # pred = (pred >= 0.68).int()
    # y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
    # print(roc_auc_score(y, pred), average_precision_score(y, pred))
    # # print(pos_edge_index)
    # print(y)
    # # print(y.s)
    # print(pred)
    # print(pred.size())
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

if __name__ == "__main__":
    main()
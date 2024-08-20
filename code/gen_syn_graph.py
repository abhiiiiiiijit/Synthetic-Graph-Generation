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
    auc, ap = test(test_data, model)
    print(f' AUC: {auc:.4f}, AP: {ap:.4f}')

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
    pos_edge_index = data.pos_edge_label_index
    neg_edge_index = data.neg_edge_label_index
    pos_y = z.new_ones(data.pos_edge_label_index.size(1))
    neg_y = z.new_zeros(data.neg_edge_label_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0)
    pos_pred = model.decode(z, pos_edge_index, sigmoid=True)
    neg_pred = model.decode(z, neg_edge_index, sigmoid=True)
    pred = torch.cat([pos_pred, neg_pred], dim=0)
    pred = (pred >= 0.68).int()
    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
    print(roc_auc_score(y, pred), average_precision_score(y, pred))
    # print(pos_edge_index)
    print(y)
    # print(y.s)
    print(pred)
    # print(pred.size())
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

if __name__ == "__main__":
    main()
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
from ae_model import GCNEncoder, GCNEncoder12, GATEncoder, GraphSAGEEncoder, GCNEncoder2


def main():
######initialization##################
    distance = 500
    country = "Germany"
    out_feat_dim = 16
    pyg_version = 1 
    pyg_file_path = f'./data/tg_graphs/{country}_pyg_graphs_d_{distance}_v_{pyg_version}.pkl'
    encoder_name = "gcn"
    model_version = 2
    write_model = False

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
##################Load data#########################################################
    with open(pyg_file_path, "rb") as f:
        data = pickle.load(f)
    data.x = data.x.float()

    train_data, val_data, test_data = transform(data)   
    model = torch.load(f'./code/models/gae_{encoder_name}_model_v{model_version}.pth')
    # auc, ap = test(test_data, model)
    # print(f' AUC: {auc:.4f}, AP: {ap:.4f}')
##############################################################
    #Graph coordinate positions are saved here
    data.pos = data.x
    # print(data.x[0:3])

    model.eval()
    z = model.encode(data.x, data.edge_index)

    # print(z.size()[0])

    no_nodes = z.size(0)
    sample_size = 400
    sampled_indices = random.sample(range(no_nodes), sample_size)

    # z_sampled_nodes = z[sampled_indices]

    new_pyg_graph = gen_new_pyg_graph(z, data, sampled_indices)

    # print(new_pyg_graph.edge_index)

    visualise_graph(new_pyg_graph)
#####################################################


def visualise_graph(data):
    # Convert the PyG Data object to a NetworkX graph
    G = to_networkx(data,  to_undirected=True)

    # Visualize the graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos=data.pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=50, font_size=5)
    plt.show()

def replace_top_x_with_1_ignore_diag(mat, x):
    # Clone the original matrix to avoid in-place modifications
    result = torch.zeros_like(mat)
    
    # Create a mask to ignore diagonal elements
    diag_mask = torch.eye(mat.size(0), mat.size(1), device=mat.device).bool()
    
    # Apply the mask to set diagonal elements to -inf, so they are not considered
    masked_mat = mat.masked_fill(diag_mask, float('-inf'))
    
    # Get the top x indices along each row ignoring diagonal
    top_x_indices = torch.topk(masked_mat, x, dim=1).indices
    
    # Scatter 1s into the result tensor at the top x indices
    result.scatter_(1, top_x_indices, 1)
    
    return result

def create_dist_matrix(coords):
    # Compute pairwise distances
    # (n, d) -> (n, 1, d) and (1, n, d) to perform broadcasting subtraction
    diffs = coords.unsqueeze(1) - coords.unsqueeze(0)

    # Square the differences, sum over the coordinate dimensions, and take the square root
    dist_matrix = torch.sqrt(torch.sum(diffs**2, dim=-1))
    return dist_matrix

def gen_new_pyg_graph(z, data, sampled_indices):
    z_sampled_nodes = z[sampled_indices]
    x = data.x[sampled_indices]
    A_prob = torch.sigmoid(torch.matmul(z_sampled_nodes, z_sampled_nodes.t()))

    # threshold = 0.5 #A_prob.mean().item()
    # A_pred = (A_prob > threshold).int()

    A_dist = create_dist_matrix(x)

    A_prob = A_prob -  A_dist

    # A_prob = torch.triu(A_prob)

    A_pred = replace_top_x_with_1_ignore_diag(A_prob, 2)

    # pred_edges = from_scipy_sparse_matrix(A_pred)

    pred_edges = A_pred.nonzero(as_tuple=False)

    # pred_edges = torch.concat(pred_edges[0],pred_edges[1],dim =1)
    # Convert pred_edges to edge_index format
    edge_index = pred_edges.t().contiguous()

    # Remove self-loops
    edge_index, _ = remove_self_loops(edge_index)

    # Sort and remove duplicate edges for undirected graph
    edge_index = coalesce(edge_index, None, num_nodes=z_sampled_nodes.size(0))[0]


    pos = data.pos[sampled_indices]
    new_graph = Data(x= x, edge_index=edge_index, pos=pos)

    return new_graph


# class GCNEncoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, 2 * out_channels)
#         self.conv2 = GCNConv(2 * out_channels, out_channels)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index).relu()
#         return self.conv2(x, edge_index)

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
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
from torch_geometric.nn import GAE, VGAE, GCNConv, GATConv, SAGEConv
import random
import torch.nn.functional as F

def main():

    #lets get network x data for all the graphs in a list
    # file_path = './data/city_lat_long.csv'
    # distance = 500
    # l_netx_cities = get_networkx_data_from_coords(file_path, distance)
    # with open("./data/networkx_cities_graph/ccs_cities_graphs_wo_edge_a.pkl", "rb") as f:
    #     l_netx_cities = pickle.load(f)

    # ox.plot_graph(l_netx_cities[0])
    # print(type(l_netx_cities[0]))
    # pos = {e[0]:tuple(e[1]['x']) for e in l_netx_cities[0].nodes(data=True)}
    # nx.draw(l_netx_cities[0],pos=pos)
    # plt.show()

##Save the graph
    # data = agg_all_graph(l_netx_cities)
    # with open("./data/tg_graphs/tg_graphs_all.pkl", "wb") as f:
    #     pickle.dump(data, f)

########################################################## testing

    # with open("./data/tg_graphs/tg_graphs_all.pkl", "rb") as f:
    #     data = pickle.load(f)

    # # print(data.x[0:5])
    # data = select_random_nodes(data, 100)
    # print(len(data.x))
    # #test the saved model
    # model = torch.load('./code/models/gae_model_v1.pth')

    # model.eval()
    # z = model.encode(data.x)
    # edge_index = model.decode(z)
    # print(edge_index)
    # auc, ap = test(data, model)
    # print(f'AUC: {auc:.4f}, AP: {ap:.4f}')
#################################################################
    # print(data1.edge_index)
    # print(data2.edge_index)
    # print(data.edge_index)

    # print(data1.x[0],data1.x[-1])
    # print(data2.x[0],data2.x[-1])
    # print(data.x[0],data.x[-1])

    # print(l_netx_cities[0].nodes(data=True))
    # print(g1.pos_edge_label_index)
    # print(g1.get_summary())

    # print(data, train_data, val_data, test_data)

    # elif not is_variational and is_linear:
    #     model = GAE(LinearEncoder(in_channels, out_channels))
    # elif is_variational and not is_linear:
    #     model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
    # elif is_variational and is_linear:
    #     model = VGAE(VariationalLinearEncoder(in_channels, out_channels))
 ###########################################################initializing 
    # l_netx_cities = remove_edge_features(l_netx_cities)
    is_variational = True
    is_linear = False
    iteration = 200

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

###################################training

    with open("./data/tg_graphs/tg_graphs_all.pkl", "rb") as f:
        data = pickle.load(f)

    train_data, val_data, test_data = transform(data)
    in_channels, out_channels = data.num_features, 16    
    if not is_variational and not is_linear:
        model = GAE(GraphSAGEEncoder())
        # model = GAE(GCNEncoder12(in_channels, out_channels))
    elif is_variational and not is_linear:
        model = VGAE(VGCNEncoder(in_channels, out_channels))


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
###################################################
    # save the model
    torch.save(model, './code/models/vgae_model_v1.pth') # Epoch: 200, AUC: 0.8713, AP: 0.8252

#####################################################


def select_random_nodes(data: Data, num_nodes: int) -> Data:
    # Check that the number of nodes to select is less than or equal to the total number of nodes
    if num_nodes > data.num_nodes:
        raise ValueError("num_nodes cannot be greater than the total number of nodes in the data.")

    # Randomly select `num_nodes` indices from the available nodes
    selected_nodes = random.sample(range(data.num_nodes), num_nodes)

    # Create a mask to filter out the selected nodes
    node_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    node_mask[selected_nodes] = True

    # Select the edges that connect the selected nodes
    # edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]

    # Create the new Data object with the selected nodes and edges
    selected_data = Data(
        x=data.x[node_mask],
        # edge_index=data.edge_index[:, edge_mask],
        # edge_attr=data.edge_attr[edge_mask] if data.edge_attr is not None else None,
        # y=data.y[node_mask] if data.y is not None else None,
        pos=data.x[node_mask] #if data.pos is not None else None
    )

    return selected_data

def agg_all_graph(g_list):
    data1 = from_networkx(g_list[0])
    for i in range(1, len(g_list)):
        data2 = from_networkx(g_list[i])
        x = torch.cat([data1.x, data2.x], dim=0)
        edge_index = torch.cat([data1.edge_index, data2.edge_index + data1.num_nodes], dim=1)
        data1 = Data(x=x, edge_index=edge_index)
    return data1

class VGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class GCNEncoder12(torch.nn.Module):#has 64 dim z space
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=64, num_layers=3):
        super(GCNEncoder12, self).__init__()
        
        self.num_layers = num_layers
        
        # Define the first GCN layer (input -> hidden)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Define additional hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Define the final GCN layer (hidden -> output)
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # Optional: Apply dropout between layers for regularization
        self.dropout = torch.nn.Dropout(p=0.5)
    
    def forward(self, x, edge_index):
        # Pass through each GCN layer with ReLU activation
        for i in range(self.num_layers - 1):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.dropout(x)
        
        # The final layer typically doesn't have an activation function
        x = self.convs[-1](x, edge_index)
        
        return x

class GATEncoder(torch.nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=64, num_layers=3, heads=4):
        super(GATEncoder, self).__init__()
        
        self.num_layers = num_layers
        
        # First GAT layer
        self.gats = torch.nn.ModuleList()
        self.gats.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True))
        
        # Additional GAT layers
        for _ in range(num_layers - 2):
            self.gats.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))
        
        # Final GAT layer (concatenation of heads disabled)
        self.gats.append(GATConv(hidden_dim * heads, output_dim, heads=1, concat=False))
        
        self.dropout = torch.nn.Dropout(p=0.5)
    
    def forward(self, x, edge_index):
        # Pass through each GAT layer with LeakyReLU activation
        for i in range(self.num_layers - 1):
            x = F.leaky_relu(self.gats[i](x, edge_index))
            x = self.dropout(x)
        
        # The final layer
        x = self.gats[-1](x, edge_index)
        
        return x

class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=64, num_layers=3):
        super(GraphSAGEEncoder, self).__init__()
        
        self.num_layers = num_layers
        
        # First GraphSAGE layer
        self.sages = torch.nn.ModuleList()
        self.sages.append(SAGEConv(input_dim, hidden_dim))
        
        # Additional GraphSAGE layers
        for _ in range(num_layers - 2):
            self.sages.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Final GraphSAGE layer
        self.sages.append(SAGEConv(hidden_dim, output_dim))
        
        self.dropout = torch.nn.Dropout(p=0.5)
    
    def forward(self, x, edge_index):
        # Pass through each GraphSAGE layer with ReLU activation
        for i in range(self.num_layers - 1):
            x = F.relu(self.sages[i](x, edge_index))
            x = self.dropout(x)
        
        # The final layer
        x = self.sages[-1](x, edge_index)
        
        return x

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
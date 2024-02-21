import torch
import torch.nn as nn
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
from torch_geometric.data import Data as GraphData
from torch_geometric.nn import RGCNConv, TransformerConv

from dgl import laplacian_pe
import corect

class GNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, num_relations, num_modals, args):
        super(GNN, self).__init__()
        self.args = args

        self.num_modals = num_modals
        
        if args.use_graph_pe=="laplacian":
            self.pe = AddLaplacianEigenvectorPE(args.laplacian_k)
            print("GNN --> Use Laplacian PE")
        if args.use_graph_pe == "rw":
            self.pe = AddRandomWalkPE(args.walk_length)
            print("GNN --> Use RW PE")
        if args.gcn_conv == "rgcn":
            print("GNN --> Use RGCN")
            self.conv1 = RGCNConv(g_dim, h1_dim, num_relations)

        if args.use_graph_transformer:
            print("GNN --> Use Graph Transformer")
           
            in_dim = h1_dim
                
            self.conv2 = TransformerConv(in_dim, h2_dim, heads=args.graph_transformer_nheads, concat=True)
            self.bn = nn.BatchNorm1d(h2_dim * args.graph_transformer_nheads)
            

    def forward(self, node_features, node_type, edge_index, edge_type):
        graph = GraphData(node_features, edge_index)

        if self.args.use_graph_pe=="laplacian":
            graph = self.pe(graph)
        if self.args.use_graph_pe == "rw":
            graph = self.pe(graph)
        if self.args.gcn_conv == "rgcn":
            # x = self.conv1(node_features, edge_index, edge_type)
            x = self.conv1(graph.x, graph.edge_index, edge_type)
        if self.args.use_graph_transformer:
            x = nn.functional.leaky_relu(self.bn(self.conv2(x, edge_index)))
        
        return x
        

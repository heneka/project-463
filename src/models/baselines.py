import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, SAGEConv

class GATModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim=64, num_layers=2, heads=2):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        
        self.convs = nn.ModuleList()
        # Initialize GAT layers. Note: GAT is feature transforming and non-linear.
        self.convs.append(GATConv(embedding_dim, embedding_dim // heads, heads=heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(embedding_dim, embedding_dim // heads, heads=heads))
        
        self.activation = nn.ReLU()

    def forward(self, edge_index):
        x = self.embedding.weight
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.activation(x)
        return x

    def decode(self, user_emb, movie_emb):
        return (user_emb * movie_emb).sum(dim=-1)

    def get_bpr_loss(self, edge_index, pos_edge_index, neg_edge_index):
        out = self.forward(edge_index)
        
        pos_user_emb = out[pos_edge_index[0]]
        pos_movie_emb = out[pos_edge_index[1]]
        neg_user_emb = out[neg_edge_index[0]]
        neg_movie_emb = out[neg_edge_index[1]]
        
        pos_scores = self.decode(pos_user_emb, pos_movie_emb)
        neg_scores = self.decode(neg_user_emb, neg_movie_emb)
        
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        return bpr_loss


class GraphSAGEModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim=64, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(embedding_dim, embedding_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(embedding_dim, embedding_dim))
        
        self.activation = nn.ReLU()

    def forward(self, edge_index):
        x = self.embedding.weight
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.activation(x)
        return x

    def decode(self, user_emb, movie_emb):
        return (user_emb * movie_emb).sum(dim=-1)

    def get_bpr_loss(self, edge_index, pos_edge_index, neg_edge_index):
        out = self.forward(edge_index)
        
        pos_user_emb = out[pos_edge_index[0]]
        pos_movie_emb = out[pos_edge_index[1]]
        neg_user_emb = out[neg_edge_index[0]]
        neg_movie_emb = out[neg_edge_index[1]]
        
        pos_scores = self.decode(pos_user_emb, pos_movie_emb)
        neg_scores = self.decode(neg_user_emb, neg_movie_emb)
        
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        return bpr_loss


class NCFModel(nn.Module):
    """
    Neural Collaborative Filtering. 
    Does not use the Graph structure (edge_index) for message passing.
    Only uses the embeddings and passes concatenated user/movie through an MLP.
    """
    def __init__(self, num_nodes, embedding_dim=64, hidden_dims=[64, 32, 16]):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        
        layers = []
        input_dim = embedding_dim * 2
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, edge_index):
        return self.embedding.weight

    def get_bpr_loss(self, edge_index, pos_edge_index, neg_edge_index):
        emb = self.embedding.weight
        
        pos_u = emb[pos_edge_index[0]]
        pos_m = emb[pos_edge_index[1]]
        neg_u = emb[neg_edge_index[0]]
        neg_m = emb[neg_edge_index[1]]
        
        pos_cat = torch.cat([pos_u, pos_m], dim=-1)
        neg_cat = torch.cat([neg_u, neg_m], dim=-1)
        
        pos_scores = self.mlp(pos_cat).squeeze(-1)
        neg_scores = self.mlp(neg_cat).squeeze(-1)
        
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        return bpr_loss

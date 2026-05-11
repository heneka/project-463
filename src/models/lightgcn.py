import torch
import torch.nn as nn
from torch_geometric.nn import LightGCN

class LightGCNModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim=64, num_layers=3):
        super().__init__()
        self.encoder = LightGCN(num_nodes=num_nodes, embedding_dim=embedding_dim, num_layers=num_layers)
        
    def forward(self, edge_index):
        return self.encoder.get_embedding(edge_index)
    
    def decode(self, user_emb, movie_emb):
        return (user_emb * movie_emb).sum(dim=-1)
    
    def predict_link(self, edge_index, user_indices, movie_indices):
        out = self.encoder.get_embedding(edge_index)
        user_emb = out[user_indices]
        movie_emb = out[movie_indices]
        return self.decode(user_emb, movie_emb)
    
    def get_bpr_loss(self, edge_index, pos_edge_index, neg_edge_index):
        out = self.encoder.get_embedding(edge_index)
        
        pos_user_emb = out[pos_edge_index[0]]
        pos_movie_emb = out[pos_edge_index[1]]
        
        neg_user_emb = out[neg_edge_index[0]]
        neg_movie_emb = out[neg_edge_index[1]]
        
        pos_scores = self.decode(pos_user_emb, pos_movie_emb)
        neg_scores = self.decode(neg_user_emb, neg_movie_emb)
        
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        
        return loss

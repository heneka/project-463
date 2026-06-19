import torch

import torch.nn as nn

import torch.nn.functional as F

from torch_geometric.nn import LightGCN


class LightGCNModel(nn.Module):


    def __init__(self, num_nodes, embedding_dim=64, num_layers=3, decoder='dot', mlp_hidden_dims=(64, 32)):

        super().__init__()

        self.decoder_type = decoder

        self.encoder = LightGCN(num_nodes=num_nodes, embedding_dim=embedding_dim, num_layers=num_layers)

        if decoder == 'mlp':

            layers = []

            input_dim = embedding_dim * 2

            for hidden_dim in mlp_hidden_dims:

                layers.append(nn.Linear(input_dim, hidden_dim))

                layers.append(nn.ReLU())

                input_dim = hidden_dim

            layers.append(nn.Linear(input_dim, 1))

            self.interaction = nn.Sequential(*layers)


    def forward(self, edge_index):

        return self.encoder.get_embedding(edge_index)


    def decode(self, user_emb, movie_emb):

        if self.decoder_type == 'cosine':

            user_emb = F.normalize(user_emb, dim=-1)

            movie_emb = F.normalize(movie_emb, dim=-1)

            return (user_emb * movie_emb).sum(dim=-1)

        if self.decoder_type == 'mlp':

            concat = torch.cat([user_emb, movie_emb], dim=-1)

            return self.interaction(concat).squeeze(-1)

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

        return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-08).mean()

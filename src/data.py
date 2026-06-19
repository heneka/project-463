import os

import pandas as pd

import torch

from torch_geometric.data import Data

from torch_geometric.datasets import MovieLens1M

from torch_geometric.transforms import RandomLinkSplit, ToUndirected


def get_movielens_dataset(root_dir='./data/MovieLens1M'):

    os.makedirs(root_dir, exist_ok=True)

    dataset = MovieLens1M(root=root_dir)

    return dataset


def filter_high_rating_edges(edge_index, ratings, min_rating=4):

    df = pd.DataFrame({'user_idx': edge_index[0].cpu().numpy(), 'movie_idx': edge_index[1].cpu().numpy(), 'rating': ratings.cpu().numpy()})

    df = df[df['rating'] >= min_rating].reset_index(drop=True)

    filtered_edge_index = torch.tensor(df[['user_idx', 'movie_idx']].values.T, dtype=torch.long)

    filtered_ratings = torch.tensor(df['rating'].values, dtype=torch.float)

    return (filtered_edge_index, filtered_ratings)


def prepare_data_splits(dataset, val_ratio=0.1, test_ratio=0.1, min_rating=4):

    data = dataset[0]

    num_users = data['user'].num_nodes

    num_movies = data['movie'].num_nodes

    edge_index = data['user', 'rates', 'movie'].edge_index

    rates_tensor = data['user', 'rates', 'movie'].edge_label if 'edge_label' in data['user', 'rates', 'movie'] else data['user', 'rates', 'movie'].rating

    (edge_index, rates_tensor) = filter_high_rating_edges(edge_index, rates_tensor, min_rating=min_rating)

    movie_offset = num_users

    edge_index_homo = edge_index.clone()

    edge_index_homo[1] += movie_offset

    homo_data = Data(edge_index=edge_index_homo, num_nodes=num_users + num_movies, y=rates_tensor)

    homo_data = ToUndirected()(homo_data)

    transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=False, neg_sampling_ratio=1.0, num_val=val_ratio, num_test=test_ratio)

    (train_data, val_data, test_data) = transform(homo_data)

    return (train_data, val_data, test_data, num_users, num_movies)

if __name__ == '__main__':

    dataset = get_movielens_dataset()

    print('Dataset:', dataset)

    (train_data, val_data, test_data, num_u, num_m) = prepare_data_splits(dataset)

    print('Train:', train_data)

    print('Val:', val_data)

    print('Test:', test_data)

    print(f'Num Users: {num_u}, Num Movies: {num_m}')

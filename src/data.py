import os
import torch
from torch_geometric.datasets import MovieLens1M
from torch_geometric.transforms import RandomLinkSplit, ToUndirected

def get_movielens_dataset(root_dir='./data/MovieLens1M'):
    """
    Downloads and prepares the MovieLens 1M dataset.
    Returns the raw dataset.
    """
    os.makedirs(root_dir, exist_ok=True)
    dataset = MovieLens1M(root=root_dir)
    return dataset

def prepare_data_splits(dataset, val_ratio=0.1, test_ratio=0.1):
    """
    Given the MovieLens 1M dataset, converts the heterogeneous graph into
    a homogeneous bipartite graph format suitable for LightGCN, and splits
    it into train, validation, and test edge sets.
    """
    data = dataset[0]
    
    # In MovieLens1M, we have nodes 'user' and 'movie' and edge type ('user', 'rates', 'movie')
    # For LightGCN, we just need the bipartite interactions.
    # Convert HeteroData to standard Data by focusing on the bipartite edges.
    
    num_users = data['user'].num_nodes
    num_movies = data['movie'].num_nodes
    
    # The edges from users to movies
    edge_index = data['user', 'rates', 'movie'].edge_index
    rates_tensor = data['user', 'rates', 'movie'].edge_label if 'edge_label' in data['user', 'rates', 'movie'] else data['user', 'rates', 'movie'].rating
    
    # We want to perform Link Prediction. Let's use PyG's RandomLinkSplit
    # RandomLinkSplit works best on homogeneous graphs or we can use the hetero version.
    # Let's map it to a homogeneous graph with (num_users + num_movies) nodes
    
    # Map movie indices to ensure they do not overlap with user indices
    movie_offset = num_users
    edge_index_homo = edge_index.clone()
    edge_index_homo[1] += movie_offset
    
    from torch_geometric.data import Data
    homo_data = Data(
        edge_index=edge_index_homo,
        num_nodes=num_users + num_movies,
        y=rates_tensor # We'll just preserve the ratings if needed
    )
    
    # Make undirected (since interactions are mutual in bipartite context for message passing)
    homo_data = ToUndirected()(homo_data)
    
    # Split edges
    transform = RandomLinkSplit(
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
        num_val=val_ratio,
        num_test=test_ratio
    )
    
    train_data, val_data, test_data = transform(homo_data)
    
    return train_data, val_data, test_data, num_users, num_movies

if __name__ == "__main__":
    dataset = get_movielens_dataset()
    print("Dataset:", dataset)
    train_data, val_data, test_data, num_u, num_m = prepare_data_splits(dataset)
    print("Train:", train_data)
    print("Val:", val_data)
    print("Test:", test_data)
    print(f"Num Users: {num_u}, Num Movies: {num_m}")

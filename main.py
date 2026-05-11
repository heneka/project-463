import sys
import os
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data import get_movielens_dataset, prepare_data_splits
from models.lightgcn import LightGCNModel
from models.baselines import GATModel, GraphSAGEModel, NCFModel
from train import train
import pandas as pd

def main():
    print("1. Preparing Data (MovieLens 1M)...")
    dataset = get_movielens_dataset()
    train_data, val_data, test_data, num_u, num_m = prepare_data_splits(dataset)
    num_nodes = num_u + num_m
    print(f"Data splits generated. Total Nodes: {num_nodes}. Train edges: {train_data.edge_index.size(1)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    models = {
        "LightGCN": LightGCNModel(num_nodes=num_nodes, embedding_dim=64, num_layers=3),
        "GAT": GATModel(num_nodes=num_nodes, embedding_dim=64, num_layers=2),
        "GraphSAGE": GraphSAGEModel(num_nodes=num_nodes, embedding_dim=64, num_layers=2),
        "NCF": NCFModel(num_nodes=num_nodes, embedding_dim=64)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\n{'='*50}\nStarting Evaluation: {name}\n{'='*50}")
        history, test_metrics = train(
            model=model,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            num_nodes=num_nodes,
            epochs=20,
            lr=0.005,
            device=device
        )
        
        results.append({
            "Model": name,
            "Test NDCG@20": test_metrics["NDCG@K"],
            "Test Recall@20": test_metrics["Recall@K"],
            "Test F1": test_metrics["F1"],
            "Avg Epoch Time (s)": np.mean(history["epoch_times"]),
            "Peak VRAM (MB)": np.max(history["peak_vram"])
        })
        
    print("\n\nFINAL RESULTS:")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    df.to_csv("results.csv", index=False)
    print("Results saved to results.csv")

if __name__ == "__main__":
    main()

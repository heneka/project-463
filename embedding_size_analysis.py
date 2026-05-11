import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data import get_movielens_dataset, prepare_data_splits
from models.lightgcn import LightGCNModel
from models.baselines import GATModel, GraphSAGEModel, NCFModel
from train import train

def main():
    print("Loading dataset...")
    dataset = get_movielens_dataset()
    train_data, val_data, test_data, num_u, num_m = prepare_data_splits(dataset)
    num_nodes = num_u + num_m
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    embedding_sizes = [16, 32, 64, 128]
    
    results = []
    
    model_constructors = {
        "LightGCN": lambda num_nodes, emb_size: LightGCNModel(num_nodes=num_nodes, embedding_dim=emb_size, num_layers=3),
        "GAT": lambda num_nodes, emb_size: GATModel(num_nodes=num_nodes, embedding_dim=emb_size, num_layers=2),
        "GraphSAGE": lambda num_nodes, emb_size: GraphSAGEModel(num_nodes=num_nodes, embedding_dim=emb_size, num_layers=2),
        "NCF": lambda num_nodes, emb_size: NCFModel(num_nodes=num_nodes, embedding_dim=emb_size),
    }
    
    for model_name, model_constructor in model_constructors.items():
        print(f"\n{'='*60}\nEvaluating: {model_name}\n{'='*60}")
        
        for emb_size in embedding_sizes:
            print(f"\n  Embedding size: {emb_size}")
            
            model = model_constructor(num_nodes, emb_size)
            history, test_metrics = train(
                model=model,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                num_nodes=num_nodes,
                epochs=15,
                lr=0.005,
                device=device,
            )
            
            recall = test_metrics['Recall@K']
            print(f"    Test Recall@20: {recall:.4f}")
            
            results.append({
                "Model": model_name,
                "Embedding Size": emb_size,
                "Test Recall@20": recall,
            })
    
    df = pd.DataFrame(results)
    csv_path = "embedding_size_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df)
    
    print("\nGenerating plot...")
    plt.figure(figsize=(12, 7))
    
    for model_name in model_constructors.keys():
        model_data = df[df["Model"] == model_name]
        plt.plot(
            model_data["Embedding Size"],
            model_data["Test Recall@20"],
            marker='o',
            linewidth=2.5,
            markersize=8,
            label=model_name,
        )
    
    plt.xlabel("Embedding Size (dimensions)", fontsize=12, fontweight='bold')
    plt.ylabel("Test Recall@20", fontsize=12, fontweight='bold')
    plt.title("Recommendation Recall vs Embedding Size (MovieLens 1M)", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join("plots", "recall_vs_embedding_size.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()

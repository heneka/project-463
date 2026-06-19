import os

import sys

import matplotlib.pyplot as plt

import pandas as pd

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data import get_movielens_dataset, prepare_data_splits

from models.lightgcn import LightGCNModel

from models.baselines import GATModel, GraphSAGEModel, NCFModel

from train import train

MODEL_COLORS = {'LightGCN': '#4C72B0', 'GAT': '#55A868', 'GraphSAGE': '#C44E52', 'NCF': '#8172B3'}


def plot_recall_vs_embedding_size(df, output_dir='plots'):

    os.makedirs(output_dir, exist_ok=True)

    (fig, ax) = plt.subplots(figsize=(12, 7))

    for model_name in df['Model'].unique():

        model_data = df[df['Model'] == model_name]

        ax.plot(model_data['Embedding Size'], model_data['Test Recall@20'], marker='o', linewidth=2.5, markersize=8, label=model_name, color=MODEL_COLORS.get(model_name))

    ax.set_xlabel('Embedding Size (dimensions)')

    ax.set_ylabel('Test Recall@20')

    ax.set_title('Recall@20 vs Embedding Size (Full-Catalog Evaluation, 4--5 Star Items)')

    ax.legend(loc='best')

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'recall_vs_embedding_size.png'), dpi=300)

    plt.close(fig)


def plot_embedding_size_heatmap(df, output_dir='plots'):

    os.makedirs(output_dir, exist_ok=True)

    pivot_df = df.pivot(index='Model', columns='Embedding Size', values='Test Recall@20')

    model_order = [m for m in MODEL_COLORS if m in pivot_df.index]

    pivot_df = pivot_df.reindex(model_order)

    (fig, ax) = plt.subplots(figsize=(8, 6))

    im = ax.imshow(pivot_df.values, cmap='YlGnBu', aspect='auto')

    ax.set_xticks(range(len(pivot_df.columns)))

    ax.set_xticklabels(pivot_df.columns)

    ax.set_yticks(range(len(pivot_df.index)))

    ax.set_yticklabels(pivot_df.index)

    ax.set_xlabel('Embedding Size')

    ax.set_ylabel('Model')

    ax.set_title('Test Recall@20 by Model and Embedding Size')

    for row in range(pivot_df.shape[0]):

        for col in range(pivot_df.shape[1]):

            ax.text(col, row, f'{pivot_df.values[row, col]:.4f}', ha='center', va='center', color='black', fontsize=9)

    fig.colorbar(im, ax=ax, label='Test Recall@20')

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'embedding_size_heatmap.png'), dpi=300)

    plt.close(fig)


def main():

    print('Loading dataset...')

    dataset = get_movielens_dataset()

    (train_data, val_data, test_data, num_u, num_m) = prepare_data_splits(dataset)

    num_nodes = num_u + num_m

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')

    embedding_sizes = [16, 32, 64, 128]

    model_constructors = {'LightGCN': lambda n, e: LightGCNModel(num_nodes=n, embedding_dim=e, num_layers=3), 'GAT': lambda n, e: GATModel(num_nodes=n, embedding_dim=e, num_layers=2), 'GraphSAGE': lambda n, e: GraphSAGEModel(num_nodes=n, embedding_dim=e, num_layers=2), 'NCF': lambda n, e: NCFModel(num_nodes=n, embedding_dim=e)}

    results = []

    for (model_name, constructor) in model_constructors.items():

        print(f"\n{'=' * 60}\nEvaluating: {model_name}\n{'=' * 60}")

        for emb_size in embedding_sizes:

            print(f'  Embedding size: {emb_size}')

            model = constructor(num_nodes, emb_size)

            (_, test_metrics) = train(model=model, train_data=train_data, val_data=val_data, test_data=test_data, num_nodes=num_nodes, num_users=num_u, num_movies=num_m, epochs=15, lr=0.005, device=device)

            recall = test_metrics['Recall@K']

            print(f'    Test Recall@20: {recall:.4f}')

            results.append({'Model': model_name, 'Embedding Size': emb_size, 'Test Recall@20': recall})

    df = pd.DataFrame(results)

    df.to_csv('embedding_size_results.csv', index=False)

    print(f'\nResults saved to embedding_size_results.csv')

    plot_recall_vs_embedding_size(df)

    plot_embedding_size_heatmap(df)

    print('Plots saved to plots/')

if __name__ == '__main__':

    main()

import os

import sys

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data import get_movielens_dataset, prepare_data_splits

from models.lightgcn import LightGCNModel

from models.baselines import GATModel, GraphSAGEModel, NCFModel

from train import train

PLOTS_DIR = 'plots'

CACHE_DIR = 'cache'

EMBEDDING_DIM = 64

EPOCHS = 20

MODEL_COLORS = {'LightGCN': '#4C72B0', 'GAT': '#55A868', 'GraphSAGE': '#C44E52', 'NCF': '#8172B3'}


def _gaussian_kde_curve(samples, grid_size=400):

    samples = np.asarray(samples, dtype=float)

    x_grid = np.linspace(samples.min(), samples.max(), grid_size)

    bandwidth = 1.06 * samples.std(ddof=1) * len(samples) ** (-1 / 5)

    bandwidth = max(bandwidth, 1e-06)

    diffs = (x_grid[:, None] - samples[None, :]) / bandwidth

    density = np.exp(-0.5 * diffs ** 2).sum(axis=1)

    density /= len(samples) * bandwidth * np.sqrt(2 * np.pi)

    return (x_grid, density)


def _model_factory(name, num_nodes):

    factories = {'LightGCN': lambda : LightGCNModel(num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM, num_layers=3), 'GAT': lambda : GATModel(num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM, num_layers=2), 'GraphSAGE': lambda : GraphSAGEModel(num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM, num_layers=2), 'NCF': lambda : NCFModel(num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM)}

    return factories[name]()


def _compute_movie_similarities(model, train_data, num_u, num_m, device, sample_size=5000, seed=42):

    model = model.to(device)

    train_data = train_data.to(device)

    model.eval()

    with torch.no_grad():

        embeddings = model(train_data.edge_index).cpu().numpy()

    movie_emb = embeddings[num_u:num_u + num_m]

    movie_emb /= np.linalg.norm(movie_emb, axis=1, keepdims=True) + 1e-08

    rng = np.random.default_rng(seed)

    idx_a = rng.integers(0, movie_emb.shape[0], size=sample_size)

    idx_b = rng.integers(0, movie_emb.shape[0], size=sample_size)

    return np.sum(movie_emb[idx_a] * movie_emb[idx_b], axis=1)


def load_or_build_similarities(model_name, train_data, val_data, test_data, num_nodes, num_u, num_m, device):

    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_path = os.path.join(CACHE_DIR, f'embedding_similarities_{model_name}.npy')

    if os.path.exists(cache_path):

        print(f'  Loading cache: {cache_path}')

        return np.load(cache_path)

    print(f'  Training {model_name} ({EPOCHS} epochs) to build embedding cache...')

    model = _model_factory(model_name, num_nodes)

    train(model=model, train_data=train_data, val_data=val_data, test_data=test_data, num_nodes=num_nodes, num_users=num_u, num_movies=num_m, epochs=EPOCHS, lr=0.005, device=device)

    similarities = _compute_movie_similarities(model, train_data, num_u, num_m, device)

    np.save(cache_path, similarities)

    return similarities


def plot_baseline_bars(results_df, output_dir):

    metrics = ['Test NDCG@20', 'Test Recall@20', 'Test Hit Rate@20', 'Test MRR']

    models = results_df['Model'].tolist()

    x = np.arange(len(models))

    width = 0.8 / len(metrics)

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']

    (fig, ax) = plt.subplots(figsize=(12, 6))

    for (idx, metric) in enumerate(metrics):

        offsets = x + (idx - (len(metrics) - 1) / 2) * width

        bars = ax.bar(offsets, results_df[metric], width=width, label=metric.replace('Test ', ''), color=colors[idx])

        for bar in bars:

            ax.annotate(f'{bar.get_height():.3f}', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='bottom', fontsize=7, xytext=(0, 3), textcoords='offset points')

    ax.set_xticks(x)

    ax.set_xticklabels(models)

    ax.set_ylabel('Score')

    ax.set_title('Full-Catalog Ranking Metrics (4--5 Star Relevant Items)')

    ax.legend()

    ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'ranking_accuracy.png'), dpi=300)

    plt.close(fig)

    prf = ['Test Precision@20', 'Test Recall@20', 'Test F1@20']

    (fig, ax) = plt.subplots(figsize=(10, 6))

    for (idx, metric) in enumerate(prf):

        offsets = x + (idx - 1) * width

        ax.bar(offsets, results_df[metric], width=width, label=metric.replace('Test ', ''), color=colors[idx])

    ax.set_xticks(x)

    ax.set_xticklabels(models)

    ax.set_ylabel('Score')

    ax.set_title('Top-20 Cutoff: Precision, Recall, and F1@20')

    ax.legend()

    ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'precision_recall_f1.png'), dpi=300)

    plt.close(fig)


def plot_hit_rate_vs_mrr(results_df, output_dir):

    (fig, ax) = plt.subplots(figsize=(8, 6))

    for (_, row) in results_df.iterrows():

        color = MODEL_COLORS.get(row['Model'], '#4C72B0')

        ax.scatter(row['Test Hit Rate@20'], row['Test MRR'], s=120, c=color, edgecolors='black', linewidths=0.6)

        ax.annotate(row['Model'], (row['Test Hit Rate@20'], row['Test MRR']), textcoords='offset points', xytext=(6, 4), fontsize=9)

    ax.set_xlabel('Hit Rate@20')

    ax.set_ylabel('MRR')

    ax.set_title('Hit Rate@20 vs MRR Across Models')

    ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'hit_rate_vs_mrr.png'), dpi=300)

    plt.close(fig)


def plot_layer_ablation(layer_df, output_dir):

    (fig, ax1) = plt.subplots(figsize=(10, 6))

    hops = layer_df['Propagation Hops'].tolist()

    x = np.arange(len(hops))

    width = 0.35

    bars1 = ax1.bar(x - width / 2, layer_df['Hit Rate@20'], width, label='Hit Rate@20', color='#4C72B0')

    bars2 = ax1.bar(x + width / 2, layer_df['MRR'], width, label='MRR', color='#55A868')

    ax1.set_xlabel('Number of Propagation Hops (LightGCN Layers)')

    ax1.set_ylabel('Ranking Score')

    ax1.set_xticks(x)

    ax1.set_xticklabels(hops)

    ax1.set_title('LightGCN: Propagation Depth vs Hit Rate and MRR')

    ax1.legend(loc='upper left')

    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    for bar in list(bars1) + list(bars2):

        ax1.annotate(f'{bar.get_height():.3f}', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='bottom', fontsize=8, xytext=(0, 3), textcoords='offset points')

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'layer_ablation.png'), dpi=300)

    plt.close(fig)


def plot_hardware(results_df, output_dir):

    colors = [MODEL_COLORS.get(m, '#4C72B0') for m in results_df['Model']]

    (fig, ax) = plt.subplots(figsize=(8, 6))

    bars = ax.bar(results_df['Model'], results_df['Peak VRAM (MB)'], color=colors)

    ax.set_title('Memory Efficiency: Peak GPU VRAM Consumption')

    ax.set_ylabel('MegaBytes (MB)')

    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for bar in bars:

        ax.annotate(f'{int(bar.get_height())} MB', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='bottom', xytext=(0, 4), textcoords='offset points')

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'hardware_vram.png'), dpi=300)

    plt.close(fig)

    (fig, ax) = plt.subplots(figsize=(8, 6))

    bars = ax.bar(results_df['Model'], results_df['Avg Epoch Time (s)'], color=colors)

    ax.set_title('Computational Efficiency: Average Epoch Training Time')

    ax.set_ylabel('Seconds (s)')

    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for bar in bars:

        ax.annotate(f'{bar.get_height():.3f}s', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='bottom', xytext=(0, 4), textcoords='offset points')

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'hardware_epoch_time.png'), dpi=300)

    plt.close(fig)


def plot_all_models_embedding_kde(similarities_by_model, output_dir):

    (fig, ax) = plt.subplots(figsize=(9, 6))

    for (model_name, similarities) in similarities_by_model.items():

        (x_grid, density) = _gaussian_kde_curve(similarities)

        color = MODEL_COLORS.get(model_name, '#4C72B0')

        ax.plot(x_grid, density, color=color, linewidth=2.2, label=model_name)

        ax.fill_between(x_grid, density, alpha=0.12, color=color)

    ax.set_xlabel('Pairwise Cosine Similarity')

    ax.set_ylabel('Probability Density')

    ax.set_title('Pairwise Movie Embedding Cosine Similarities (All Models, dim=64)')

    ax.legend()

    ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'embedding_cosine_similarity_density.png'), dpi=300)

    plt.close(fig)


def plot_recall_vs_embedding_size(emb_df, output_dir):

    (fig, ax) = plt.subplots(figsize=(12, 7))

    for model_name in emb_df['Model'].unique():

        model_data = emb_df[emb_df['Model'] == model_name]

        ax.plot(model_data['Embedding Size'], model_data['Test Recall@20'], marker='o', linewidth=2.5, markersize=8, label=model_name, color=MODEL_COLORS.get(model_name, None))

    ax.set_xlabel('Embedding Size (dimensions)')

    ax.set_ylabel('Test Recall@20')

    ax.set_title('Recall@20 vs Embedding Size (Full-Catalog Evaluation, 4--5 Star Items)')

    ax.legend(loc='best')

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'recall_vs_embedding_size.png'), dpi=300)

    plt.close(fig)


def plot_embedding_size_heatmap(emb_df, output_dir):

    pivot_df = emb_df.pivot(index='Model', columns='Embedding Size', values='Test Recall@20')

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

    os.makedirs(PLOTS_DIR, exist_ok=True)

    results_df = pd.read_csv('results.csv')

    layer_df = pd.read_csv('layer_ablation_results.csv')

    plot_baseline_bars(results_df, PLOTS_DIR)

    plot_hit_rate_vs_mrr(results_df, PLOTS_DIR)

    plot_layer_ablation(layer_df, PLOTS_DIR)

    plot_hardware(results_df, PLOTS_DIR)

    results_df[['Model', 'Test Hit Rate@20', 'Test MRR', 'Test Recall@20', 'Test NDCG@20']].to_csv('hit_rate_mrr_results.csv', index=False)

    if os.path.exists('embedding_size_results.csv'):

        emb_df = pd.read_csv('embedding_size_results.csv')

        plot_recall_vs_embedding_size(emb_df, PLOTS_DIR)

        plot_embedding_size_heatmap(emb_df, PLOTS_DIR)

        print('Updated embedding-size plots from embedding_size_results.csv')

    else:

        print('Warning: embedding_size_results.csv not found; skipping embedding-size plots')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = get_movielens_dataset()

    (train_data, val_data, test_data, num_u, num_m) = prepare_data_splits(dataset)

    num_nodes = num_u + num_m

    similarities_by_model = {}

    for model_name in MODEL_COLORS:

        print(f'Building embedding similarities for {model_name}...')

        similarities_by_model[model_name] = load_or_build_similarities(model_name, train_data, val_data, test_data, num_nodes, num_u, num_m, device)

    plot_all_models_embedding_kde(similarities_by_model, PLOTS_DIR)

    print('Plots regenerated in plots/')

if __name__ == '__main__':

    main()

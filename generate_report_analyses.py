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

EPOCHS = 20

EMBEDDING_DIM = 64

LAYER_COUNTS = [1, 2, 3, 4]


def run_baseline_experiments(device):

    dataset = get_movielens_dataset()

    (train_data, val_data, test_data, num_u, num_m) = prepare_data_splits(dataset)

    num_nodes = num_u + num_m

    models = {'LightGCN': LightGCNModel(num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM, num_layers=3), 'GAT': GATModel(num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM, num_layers=2), 'GraphSAGE': GraphSAGEModel(num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM, num_layers=2), 'NCF': NCFModel(num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM)}

    results = []

    for (name, model) in models.items():

        print(f'\n=== Baseline: {name} ===')

        (history, test_metrics) = train(model=model, train_data=train_data, val_data=val_data, test_data=test_data, num_nodes=num_nodes, num_users=num_u, num_movies=num_m, epochs=EPOCHS, lr=0.005, device=device)

        results.append({'Model': name, 'Test NDCG@20': test_metrics['NDCG@K'], 'Test Precision@20': test_metrics['Precision@K'], 'Test Recall@20': test_metrics['Recall@K'], 'Test F1@20': test_metrics['F1@K'], 'Test Hit Rate@20': test_metrics['HitRate@K'], 'Test MRR': test_metrics['MRR'], 'Avg Epoch Time (s)': np.mean(history['epoch_times']), 'Peak VRAM (MB)': np.max(history['peak_vram'])})

    df = pd.DataFrame(results)

    df.to_csv('results.csv', index=False)

    return (df, train_data, val_data, test_data, num_u, num_m, num_nodes)


def run_layer_ablation(device, train_data, val_data, test_data, num_u, num_m, num_nodes):

    rows = []

    trained_models = {}

    for num_layers in LAYER_COUNTS:

        print(f'\n=== LightGCN layer ablation: {num_layers} hop(s) ===')

        model = LightGCNModel(num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM, num_layers=num_layers)

        (history, test_metrics) = train(model=model, train_data=train_data, val_data=val_data, test_data=test_data, num_nodes=num_nodes, num_users=num_u, num_movies=num_m, epochs=EPOCHS, lr=0.005, device=device)

        rows.append({'Propagation Hops': num_layers, 'Hit Rate@20': test_metrics['HitRate@K'], 'MRR': test_metrics['MRR'], 'Recall@20': test_metrics['Recall@K'], 'NDCG@20': test_metrics['NDCG@K'], 'Avg Epoch Time (s)': np.mean(history['epoch_times']), 'Peak VRAM (MB)': np.max(history['peak_vram'])})

        trained_models[num_layers] = model.cpu()

    df = pd.DataFrame(rows)

    df.to_csv('layer_ablation_results.csv', index=False)

    return (df, trained_models)


def plot_hit_rate_vs_mrr(results_df, output_dir):

    (fig, ax) = plt.subplots(figsize=(8, 6))

    ax.scatter(results_df['Test Hit Rate@20'], results_df['Test MRR'], s=120, c='#4C72B0', edgecolors='black', linewidths=0.6)

    for (_, row) in results_df.iterrows():

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


def _gaussian_kde_curve(samples, grid_size=400):

    samples = np.asarray(samples, dtype=float)

    x_grid = np.linspace(samples.min(), samples.max(), grid_size)

    bandwidth = 1.06 * samples.std(ddof=1) * len(samples) ** (-1 / 5)

    bandwidth = max(bandwidth, 1e-06)

    diffs = (x_grid[:, None] - samples[None, :]) / bandwidth

    density = np.exp(-0.5 * diffs ** 2).sum(axis=1)

    density /= len(samples) * bandwidth * np.sqrt(2 * np.pi)

    return (x_grid, density)


def plot_embedding_similarity_kde(similarities, output_dir):

    (x_grid, density) = _gaussian_kde_curve(similarities)

    (fig, ax) = plt.subplots(figsize=(8, 6))

    ax.plot(x_grid, density, color='#4C72B0', linewidth=2.2)

    ax.fill_between(x_grid, density, alpha=0.25, color='#4C72B0')

    ax.set_xlabel('Pairwise Cosine Similarity')

    ax.set_ylabel('Probability Density')

    ax.set_title('Distribution of Pairwise Movie Embedding Cosine Similarities (LightGCN)')

    ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'embedding_cosine_similarity_density.png'), dpi=300)

    plt.close(fig)


def plot_embedding_similarity_density(model, train_data, num_u, num_m, output_dir, device, sample_size=5000, seed=42):

    model = model.to(device)

    train_data = train_data.to(device)

    model.eval()

    with torch.no_grad():

        embeddings = model(train_data.edge_index).cpu().numpy()

    movie_emb = embeddings[num_u:num_u + num_m]

    movie_emb = movie_emb / (np.linalg.norm(movie_emb, axis=1, keepdims=True) + 1e-08)

    rng = np.random.default_rng(seed)

    num_movies = movie_emb.shape[0]

    idx_a = rng.integers(0, num_movies, size=sample_size)

    idx_b = rng.integers(0, num_movies, size=sample_size)

    similarities = np.sum(movie_emb[idx_a] * movie_emb[idx_b], axis=1)

    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')

    os.makedirs(cache_dir, exist_ok=True)

    np.save(os.path.join(cache_dir, 'embedding_similarities.npy'), similarities)

    plot_embedding_similarity_kde(similarities, output_dir)


def plot_baseline_bars(results_df, output_dir):

    metrics = ['Test NDCG@20', 'Test Recall@20', 'Test Hit Rate@20', 'Test MRR']

    models = results_df['Model'].tolist()

    x = np.arange(len(models))

    width = 0.8 / len(metrics)

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']

    (fig, ax) = plt.subplots(figsize=(12, 6))

    for (idx, metric) in enumerate(metrics):

        offsets = x + (idx - (len(metrics) - 1) / 2) * width

        values = results_df[metric].tolist()

        bars = ax.bar(offsets, values, width=width, label=metric.replace('Test ', ''), color=colors[idx])

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

        values = results_df[metric].tolist()

        ax.bar(offsets, values, width=width, label=metric.replace('Test ', ''), color=colors[idx])

    ax.set_xticks(x)

    ax.set_xticklabels(models)

    ax.set_ylabel('Score')

    ax.set_title('Top-20 Cutoff: Precision, Recall, and F1@20')

    ax.legend()

    ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'precision_recall_f1.png'), dpi=300)

    plt.close(fig)


def main():

    os.makedirs(PLOTS_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')

    (results_df, train_data, val_data, test_data, num_u, num_m, num_nodes) = run_baseline_experiments(device)

    (layer_df, trained_models) = run_layer_ablation(device, train_data, val_data, test_data, num_u, num_m, num_nodes)

    plot_baseline_bars(results_df, PLOTS_DIR)

    plot_hit_rate_vs_mrr(results_df, PLOTS_DIR)

    plot_layer_ablation(layer_df, PLOTS_DIR)

    best_layer = int(layer_df.loc[layer_df['MRR'].idxmax(), 'Propagation Hops'])

    plot_embedding_similarity_density(trained_models[best_layer], train_data, num_u, num_m, PLOTS_DIR, device)

    hit_mrr_df = results_df[['Model', 'Test Hit Rate@20', 'Test MRR', 'Test Recall@20', 'Test NDCG@20']]

    hit_mrr_df.to_csv('hit_rate_mrr_results.csv', index=False)

    print('\n=== Baseline Results ===')

    print(results_df.to_string(index=False))

    print('\n=== Layer Ablation ===')

    print(layer_df.to_string(index=False))

    print('\nArtifacts written to results.csv, layer_ablation_results.csv, hit_rate_mrr_results.csv, plots/')

if __name__ == '__main__':

    main()

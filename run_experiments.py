import os

import sys

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data import get_movielens_dataset, prepare_data_splits

from models.baselines import GATModel, GraphSAGEModel, MFModel, NCFModel

from models.lightgcn import LightGCNModel

from metrics import evaluate_metrics

from train import train

EMBEDDING_DIM = 64

EPOCHS = 20

LR = 0.005

LIGHTGCN_LAYERS = 3

LIGHTWEIGHT_LAYERS = 1

PLOTS_DIR = 'plots'

CACHE_DIR = 'cache'

EXP1_CSV = 'results_exp1.csv'

EXP2_CSV = 'results_exp2.csv'

EXP3_LAYERS_CSV = 'results_exp3_layers.csv'

EXP3_DIMS_CSV = 'results_exp3_dims.csv'

EXP4_GEOMETRY_CSV = 'results_exp4_geometry.csv'

EXP4_DECODER_CSV = 'results_exp4_decoder_swap.csv'

EXP5_CSV = 'results_exp5_training.csv'

RANKING_METRICS = ['Test NDCG@20', 'Test Recall@20', 'Test Hit Rate@20', 'Test MRR']


def metrics_row(name, encoder, decoder, history, test_metrics, extra=None):

    row = {'Model': name, 'Encoder': encoder, 'Decoder': decoder, 'Test NDCG@20': test_metrics['NDCG@K'], 'Test Precision@20': test_metrics['Precision@K'], 'Test Recall@20': test_metrics['Recall@K'], 'Test F1@20': test_metrics['F1@K'], 'Test Hit Rate@20': test_metrics['HitRate@K'], 'Test MRR': test_metrics['MRR'], 'Avg Epoch Time (s)': float(np.mean(history['epoch_times'])), 'Peak VRAM (MB)': float(np.max(history['peak_vram']))}

    if extra:

        row.update(extra)

    return row


def ranking_row(name, history, test_metrics, extra=None):

    row = {'Model': name, 'Test NDCG@20': test_metrics['NDCG@K'], 'Test Precision@20': test_metrics['Precision@K'], 'Test Recall@20': test_metrics['Recall@K'], 'Test F1@20': test_metrics['F1@K'], 'Test Hit Rate@20': test_metrics['HitRate@K'], 'Test MRR': test_metrics['MRR'], 'Avg Epoch Time (s)': float(np.mean(history['epoch_times'])), 'Peak VRAM (MB)': float(np.max(history['peak_vram']))}

    if extra:

        row.update(extra)

    return row


def run_model(model, name, train_data, val_data, test_data, num_nodes, num_u, num_m, device, num_negatives=1, l2_reg=0.0, return_model=False):

    print(f"\n{'=' * 60}\nTraining: {name}\n{'=' * 60}")

    result = train(model=model, train_data=train_data, val_data=val_data, test_data=test_data, num_nodes=num_nodes, num_users=num_u, num_movies=num_m, epochs=EPOCHS, lr=LR, device=device, num_negatives=num_negatives, l2_reg=l2_reg, return_model=return_model)

    if return_model:

        (history, test_metrics, model) = result

        return (history, test_metrics, model)

    (history, test_metrics) = result

    return (history, test_metrics)


def experiment_1(train_data, val_data, test_data, num_nodes, num_u, num_m, device):

    configs = [('MF', MFModel(num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM), 'ID embeddings', 'dot'), ('LightGCN', LightGCNModel(num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM, num_layers=LIGHTGCN_LAYERS, decoder='dot'), 'LightGCN', 'dot'), ('GraphSAGE', GraphSAGEModel(num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM, num_layers=2), 'GraphSAGE', 'dot'), ('GAT', GATModel(num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM, num_layers=2), 'GAT', 'dot'), ('NCF', NCFModel(num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM), 'ID embeddings', 'mlp')]

    rows = []

    for (name, model, encoder, decoder) in configs:

        (history, test_metrics) = run_model(model, name, train_data, val_data, test_data, num_nodes, num_u, num_m, device)

        rows.append(metrics_row(name, encoder, decoder, history, test_metrics))

    df = pd.DataFrame(rows)

    df.to_csv(EXP1_CSV, index=False)

    print(f'\nExperiment 1 saved to {EXP1_CSV}')

    return df


def experiment_2(train_data, val_data, test_data, num_nodes, num_u, num_m, device):

    decoders = [('LightGCN + Dot', 'dot'), ('LightGCN + Cosine', 'cosine'), ('LightGCN + MLP', 'mlp')]

    rows = []

    for (name, decoder) in decoders:

        model = LightGCNModel(num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM, num_layers=LIGHTGCN_LAYERS, decoder=decoder)

        (history, test_metrics) = run_model(model, name, train_data, val_data, test_data, num_nodes, num_u, num_m, device)

        rows.append(metrics_row(name, 'LightGCN', decoder, history, test_metrics))

    df = pd.DataFrame(rows)

    df.to_csv(EXP2_CSV, index=False)

    print(f'\nExperiment 2 saved to {EXP2_CSV}')

    return df


def experiment_3(train_data, val_data, test_data, num_nodes, num_u, num_m, device):

    layer_rows = []

    for num_layers in [1, 2, 3, 4]:

        name = f'LightGCN ({num_layers}-hop)'

        model = LightGCNModel(num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM, num_layers=num_layers, decoder='dot')

        (history, test_metrics) = run_model(model, name, train_data, val_data, test_data, num_nodes, num_u, num_m, device)

        layer_rows.append(ranking_row(name, history, test_metrics, extra={'Propagation Hops': num_layers, 'Embedding Dim': EMBEDDING_DIM}))

    layers_df = pd.DataFrame(layer_rows)

    layers_df.to_csv(EXP3_LAYERS_CSV, index=False)

    print(f'\nExperiment 3 (layers) saved to {EXP3_LAYERS_CSV}')

    dim_rows = []

    for emb_dim in [16, 32, 64, 128]:

        name = f'LightGCN (dim={emb_dim})'

        model = LightGCNModel(num_nodes=num_nodes, embedding_dim=emb_dim, num_layers=LIGHTWEIGHT_LAYERS, decoder='dot')

        (history, test_metrics) = run_model(model, name, train_data, val_data, test_data, num_nodes, num_u, num_m, device)

        dim_rows.append(ranking_row(name, history, test_metrics, extra={'Propagation Hops': LIGHTWEIGHT_LAYERS, 'Embedding Dim': emb_dim}))

    dims_df = pd.DataFrame(dim_rows)

    dims_df.to_csv(EXP3_DIMS_CSV, index=False)

    print(f'Experiment 3 (dims) saved to {EXP3_DIMS_CSV}')

    return (layers_df, dims_df)


def _gaussian_kde_curve(samples, grid_size=400):

    samples = np.asarray(samples, dtype=float)

    x_grid = np.linspace(samples.min(), samples.max(), grid_size)

    bandwidth = 1.06 * samples.std(ddof=1) * len(samples) ** (-1 / 5)

    bandwidth = max(bandwidth, 1e-06)

    diffs = (x_grid[:, None] - samples[None, :]) / bandwidth

    density = np.exp(-0.5 * diffs ** 2).sum(axis=1)

    density /= len(samples) * bandwidth * np.sqrt(2 * np.pi)

    return (x_grid, density)


def _compute_geometry_stats(embeddings, num_u, num_m, sample_size=5000, seed=42):

    user_emb = embeddings[:num_u]

    movie_emb = embeddings[num_u:num_u + num_m]

    user_norms = np.linalg.norm(user_emb, axis=1)

    movie_norms = np.linalg.norm(movie_emb, axis=1)

    movie_unit = movie_emb / (movie_norms[:, None] + 1e-08)

    rng = np.random.default_rng(seed)

    idx_a = rng.integers(0, num_m, size=sample_size)

    idx_b = rng.integers(0, num_m, size=sample_size)

    cosine_sims = np.sum(movie_unit[idx_a] * movie_unit[idx_b], axis=1)

    return {'user_norm_mean': float(user_norms.mean()), 'user_norm_std': float(user_norms.std()), 'movie_norm_mean': float(movie_norms.mean()), 'movie_norm_std': float(movie_norms.std()), 'cosine_sim_mean': float(cosine_sims.mean()), 'cosine_sim_std': float(cosine_sims.std()), 'cosine_sims': cosine_sims, 'user_norms': user_norms, 'movie_norms': movie_norms}


def experiment_4(train_data, val_data, test_data, num_nodes, num_u, num_m, device):

    name = f'LightGCN ({LIGHTWEIGHT_LAYERS}-hop, dim={EMBEDDING_DIM})'

    model = LightGCNModel(num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM, num_layers=LIGHTWEIGHT_LAYERS, decoder='dot')

    (history, test_metrics, model) = run_model(model, name, train_data, val_data, test_data, num_nodes, num_u, num_m, device, return_model=True)

    model.eval()

    with torch.no_grad():

        embeddings = model(train_data.edge_index).cpu().numpy()

    stats = _compute_geometry_stats(embeddings, num_u, num_m)

    os.makedirs(CACHE_DIR, exist_ok=True)

    np.save(os.path.join(CACHE_DIR, 'exp4_cosine_sims.npy'), stats['cosine_sims'])

    np.save(os.path.join(CACHE_DIR, 'exp4_user_norms.npy'), stats['user_norms'])

    np.save(os.path.join(CACHE_DIR, 'exp4_movie_norms.npy'), stats['movie_norms'])

    geometry_row = {'Model': name, 'User Norm Mean': stats['user_norm_mean'], 'User Norm Std': stats['user_norm_std'], 'Movie Norm Mean': stats['movie_norm_mean'], 'Movie Norm Std': stats['movie_norm_std'], 'Cosine Sim Mean': stats['cosine_sim_mean'], 'Cosine Sim Std': stats['cosine_sim_std'], 'Test Hit Rate@20': test_metrics['HitRate@K'], 'Test MRR': test_metrics['MRR'], 'Test NDCG@20': test_metrics['NDCG@K']}

    geometry_df = pd.DataFrame([geometry_row])

    geometry_df.to_csv(EXP4_GEOMETRY_CSV, index=False)

    print(f'\nExperiment 4 (geometry stats) saved to {EXP4_GEOMETRY_CSV}')

    decoder_rows = []

    for decoder in ['dot', 'cosine']:

        swapped = evaluate_metrics(model=model, data=test_data, edge_label_index=test_data.edge_label_index, edge_label=test_data.edge_label, num_users=num_u, num_movies=num_m, train_edge_index=train_data.edge_index, decoder_override=decoder)

        decoder_rows.append({'Decoder (frozen embeddings)': decoder, 'Test NDCG@20': swapped['NDCG@K'], 'Test Recall@20': swapped['Recall@K'], 'Test Hit Rate@20': swapped['HitRate@K'], 'Test MRR': swapped['MRR']})

    decoder_df = pd.DataFrame(decoder_rows)

    decoder_df.to_csv(EXP4_DECODER_CSV, index=False)

    print(f'Experiment 4 (decoder swap) saved to {EXP4_DECODER_CSV}')

    return (geometry_df, decoder_df, stats)


def experiment_5(train_data, val_data, test_data, num_nodes, num_u, num_m, device):

    configs = [('BPR baseline (1 neg)', 1, 0.0), ('L2 reg (1e-4)', 1, 0.0001), ('4 negatives', 4, 0.0), ('L2 + 4 negatives', 4, 0.0001)]

    rows = []

    for (name, num_negatives, l2_reg) in configs:

        model = LightGCNModel(num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM, num_layers=LIGHTWEIGHT_LAYERS, decoder='dot')

        (history, test_metrics) = run_model(model, name, train_data, val_data, test_data, num_nodes, num_u, num_m, device, num_negatives=num_negatives, l2_reg=l2_reg)

        rows.append(ranking_row(name, history, test_metrics, extra={'Num Negatives': num_negatives, 'L2 Reg': l2_reg, 'Propagation Hops': LIGHTWEIGHT_LAYERS, 'Embedding Dim': EMBEDDING_DIM}))

    df = pd.DataFrame(rows)

    df.to_csv(EXP5_CSV, index=False)

    print(f'\nExperiment 5 saved to {EXP5_CSV}')

    return df


def _grouped_barplot(df, metrics, title, ylabel, output_path, model_col='Model'):

    models = df[model_col].tolist()

    x = np.arange(len(models))

    width = 0.8 / len(metrics)

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']

    (fig, ax) = plt.subplots(figsize=(12, 6))

    for (idx, metric) in enumerate(metrics):

        offsets = x + (idx - (len(metrics) - 1) / 2) * width

        bars = ax.bar(offsets, df[metric], width=width, label=metric.replace('Test ', ''), color=colors[idx % len(colors)])

        for bar in bars:

            ax.annotate(f'{bar.get_height():.3f}', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='bottom', fontsize=7, xytext=(0, 3), textcoords='offset points')

    ax.set_xticks(x)

    ax.set_xticklabels(models, rotation=15, ha='right')

    ax.set_ylabel(ylabel)

    ax.set_title(title)

    ax.legend()

    ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()

    fig.savefig(output_path, dpi=300)

    plt.close(fig)


def plot_exp1(df, output_dir):

    _grouped_barplot(df, RANKING_METRICS, 'Exp 1: Graph vs No-Graph (Encoder Ablation, Full-Catalog @20)', 'Score', os.path.join(output_dir, 'exp1_encoder_ablation.png'))

    (fig, ax) = plt.subplots(figsize=(8, 6))

    dot_models = df[df['Decoder'] == 'dot']

    colors = ['#8172B3' if enc == 'ID embeddings' else '#4C72B0' for enc in dot_models['Encoder']]

    bars = ax.bar(dot_models['Model'], dot_models['Test MRR'], color=colors)

    ax.set_ylabel('Test MRR')

    ax.set_title('Exp 1: MRR for Dot-Product Models (MF vs Graph Encoders)')

    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for bar in bars:

        ax.annotate(f'{bar.get_height():.3f}', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='bottom', fontsize=8, xytext=(0, 3), textcoords='offset points')

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'exp1_dot_product_mrr.png'), dpi=300)

    plt.close(fig)

    (fig, ax) = plt.subplots(figsize=(8, 6))

    for (_, row) in df.iterrows():

        ax.scatter(row['Avg Epoch Time (s)'], row['Test MRR'], s=100, edgecolors='black', linewidths=0.5)

        ax.annotate(row['Model'], (row['Avg Epoch Time (s)'], row['Test MRR']), textcoords='offset points', xytext=(5, 4), fontsize=8)

    ax.set_xlabel('Avg Epoch Time (s)')

    ax.set_ylabel('Test MRR')

    ax.set_title('Exp 1: Accuracy vs Training Cost')

    ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'exp1_pareto_time_mrr.png'), dpi=300)

    plt.close(fig)


def plot_exp2(df, output_dir):

    _grouped_barplot(df, RANKING_METRICS, 'Exp 2: Decoder Ablation on LightGCN (Full-Catalog @20)', 'Score', os.path.join(output_dir, 'exp2_decoder_ablation.png'), model_col='Model')

    (fig, ax) = plt.subplots(figsize=(7, 5))

    decoders = df['Decoder'].tolist()

    x = np.arange(len(decoders))

    width = 0.25

    ax.bar(x - width, df['Test NDCG@20'], width, label='NDCG@20', color='#8172B3')

    ax.bar(x, df['Test Hit Rate@20'], width, label='Hit Rate@20', color='#4C72B0')

    ax.bar(x + width, df['Test MRR'], width, label='MRR', color='#55A868')

    ax.set_xticks(x)

    ax.set_xticklabels(decoders)

    ax.set_xlabel('Decoder')

    ax.set_ylabel('Score')

    ax.set_title('Exp 2: NDCG, Hit Rate, and MRR by Decoder')

    ax.legend()

    ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'exp2_hit_rate_vs_mrr.png'), dpi=300)

    plt.close(fig)


def plot_exp3(layers_df, dims_df, output_dir):

    (fig, ax) = plt.subplots(figsize=(10, 6))

    hops = layers_df['Propagation Hops'].tolist()

    x = np.arange(len(hops))

    width = 0.25

    ax.bar(x - width, layers_df['Test NDCG@20'], width, label='NDCG@20', color='#8172B3')

    ax.bar(x, layers_df['Test Hit Rate@20'], width, label='Hit Rate@20', color='#4C72B0')

    ax.bar(x + width, layers_df['Test MRR'], width, label='MRR', color='#55A868')

    ax.set_xticks(x)

    ax.set_xticklabels(hops)

    ax.set_xlabel('Propagation Hops')

    ax.set_ylabel('Score')

    ax.set_title('Exp 3: LightGCN Depth vs Ranking Metrics (dim=64)')

    ax.legend()

    ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'exp3_layer_ablation.png'), dpi=300)

    plt.close(fig)

    (fig, ax) = plt.subplots(figsize=(10, 6))

    ax.plot(dims_df['Embedding Dim'], dims_df['Test Recall@20'], marker='o', linewidth=2.2, label='Recall@20', color='#4C72B0')

    ax.plot(dims_df['Embedding Dim'], dims_df['Test MRR'], marker='s', linewidth=2.2, label='MRR', color='#55A868')

    ax.set_xlabel('Embedding Dimension')

    ax.set_ylabel('Score')

    ax.set_title(f'Exp 3: Embedding Size vs Ranking ({LIGHTWEIGHT_LAYERS}-hop LightGCN)')

    ax.legend()

    ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'exp3_embedding_dim.png'), dpi=300)

    plt.close(fig)

    pareto_df = pd.concat([layers_df.assign(Config=layers_df['Propagation Hops'].astype(str) + '-hop'), dims_df.assign(Config='dim=' + dims_df['Embedding Dim'].astype(str))], ignore_index=True)

    (fig, axes) = plt.subplots(1, 2, figsize=(14, 6))

    for (ax, x_col, xlabel) in zip(axes, ['Avg Epoch Time (s)', 'Peak VRAM (MB)'], ['Avg Epoch Time (s)', 'Peak VRAM (MB)']):

        for (_, row) in pareto_df.iterrows():

            ax.scatter(row[x_col], row['Test MRR'], s=90, edgecolors='black', linewidths=0.5)

            ax.annotate(row['Config'], (row[x_col], row['Test MRR']), textcoords='offset points', xytext=(4, 3), fontsize=8)

        ax.set_xlabel(xlabel)

        ax.set_ylabel('Test MRR')

        ax.grid(True, linestyle='--', alpha=0.4)

    axes[0].set_title('Exp 3: MRR vs Training Time')

    axes[1].set_title('Exp 3: MRR vs Peak VRAM')

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'exp3_pareto.png'), dpi=300)

    plt.close(fig)


def plot_exp4(geometry_df, decoder_df, stats, output_dir):

    (x_grid, density) = _gaussian_kde_curve(stats['cosine_sims'])

    (fig, ax) = plt.subplots(figsize=(8, 6))

    ax.plot(x_grid, density, color='#4C72B0', linewidth=2.2)

    ax.fill_between(x_grid, density, alpha=0.2, color='#4C72B0')

    ax.set_xlabel('Pairwise Movie Cosine Similarity')

    ax.set_ylabel('Density')

    ax.set_title('Exp 4: Movie Embedding Cosine Similarity Distribution')

    ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'exp4_cosine_kde.png'), dpi=300)

    plt.close(fig)

    (fig, ax) = plt.subplots(figsize=(8, 6))

    ax.hist(stats['user_norms'], bins=50, alpha=0.65, label='User', color='#4C72B0', density=True)

    ax.hist(stats['movie_norms'], bins=50, alpha=0.65, label='Movie', color='#55A868', density=True)

    ax.set_xlabel('L2 Norm')

    ax.set_ylabel('Density')

    ax.set_title('Exp 4: User vs Movie Embedding Norms')

    ax.legend()

    ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'exp4_embedding_norms.png'), dpi=300)

    plt.close(fig)

    (fig, ax) = plt.subplots(figsize=(7, 6))

    ax.scatter(geometry_df['Test Hit Rate@20'], geometry_df['Test MRR'], s=120, c='#4C72B0', edgecolors='black')

    ax.set_xlabel('Hit Rate@20')

    ax.set_ylabel('MRR')

    ax.set_title('Exp 4: Hit Rate vs MRR (LightGCN)')

    ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'exp4_hit_rate_vs_mrr.png'), dpi=300)

    plt.close(fig)

    decoders = decoder_df['Decoder (frozen embeddings)'].tolist()

    x = np.arange(len(decoders))

    width = 0.35

    (fig, ax) = plt.subplots(figsize=(7, 5))

    ax.bar(x - width / 2, decoder_df['Test Hit Rate@20'], width, label='Hit Rate@20', color='#4C72B0')

    ax.bar(x + width / 2, decoder_df['Test MRR'], width, label='MRR', color='#55A868')

    ax.set_xticks(x)

    ax.set_xticklabels(decoders)

    ax.set_xlabel('Scorer (same frozen embeddings)')

    ax.set_ylabel('Score')

    ax.set_title('Exp 4: Dot vs Cosine on Frozen LightGCN Embeddings')

    ax.legend()

    ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'exp4_decoder_swap.png'), dpi=300)

    plt.close(fig)


def plot_exp5(df, output_dir):

    _grouped_barplot(df, RANKING_METRICS, 'Exp 5: Training Objective Ablation (LightGCN, dot product)', 'Score', os.path.join(output_dir, 'exp5_training_ablation.png'))

    (fig, ax) = plt.subplots(figsize=(8, 6))

    for (_, row) in df.iterrows():

        ax.scatter(row['Avg Epoch Time (s)'], row['Test MRR'], s=100, edgecolors='black', linewidths=0.5)

        ax.annotate(row['Model'], (row['Avg Epoch Time (s)'], row['Test MRR']), textcoords='offset points', xytext=(4, 3), fontsize=8)

    ax.set_xlabel('Avg Epoch Time (s)')

    ax.set_ylabel('Test MRR')

    ax.set_title('Exp 5: MRR vs Training Cost')

    ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'exp5_pareto_time_mrr.png'), dpi=300)

    plt.close(fig)


def summarize_exp345(layers_df, dims_df, geometry_df, decoder_df, exp5_df):

    print('\n' + '=' * 60)

    print('EXPERIMENTS 3–5 SUMMARY')

    print('=' * 60)

    best_layer = layers_df.loc[layers_df['Test MRR'].idxmax()]

    best_dim = dims_df.loc[dims_df['Test MRR'].idxmax()]

    print('\nExperiment 3 — Lightweight encoder:')

    print(f"  Best depth: {int(best_layer['Propagation Hops'])}-hop (MRR={best_layer['Test MRR']:.4f}, NDCG@20={best_layer['Test NDCG@20']:.4f})")

    print(f"  Best dim: {int(best_dim['Embedding Dim'])} (MRR={best_dim['Test MRR']:.4f}, Recall@20={best_dim['Test Recall@20']:.4f})")

    dot_row = decoder_df[decoder_df['Decoder (frozen embeddings)'] == 'dot'].iloc[0]

    cos_row = decoder_df[decoder_df['Decoder (frozen embeddings)'] == 'cosine'].iloc[0]

    print('\nExperiment 4 — Embedding geometry:')

    print(f"  User norm: {geometry_df['User Norm Mean'].iloc[0]:.3f} ± {geometry_df['User Norm Std'].iloc[0]:.3f} | Movie norm: {geometry_df['Movie Norm Mean'].iloc[0]:.3f} ± {geometry_df['Movie Norm Std'].iloc[0]:.3f}")

    print(f"  Frozen dot MRR={dot_row['Test MRR']:.4f} vs cosine MRR={cos_row['Test MRR']:.4f}")

    best_train = exp5_df.loc[exp5_df['Test MRR'].idxmax()]

    baseline = exp5_df.iloc[0]

    print('\nExperiment 5 — Training objective:')

    print(f"  Best config: {best_train['Model']} (MRR={best_train['Test MRR']:.4f}, NDCG@20={best_train['Test NDCG@20']:.4f})")

    delta = best_train['Test MRR'] - baseline['Test MRR']

    print(f'  Gain over BPR baseline: {delta:+.4f} MRR')


def summarize_findings(exp1_df, exp2_df):

    print('\n' + '=' * 60)

    print('FINDINGS SUMMARY')

    print('=' * 60)

    dot_only = exp1_df[exp1_df['Decoder'] == 'dot'].sort_values('Test MRR', ascending=False)

    best_dot = dot_only.iloc[0]

    mf_row = exp1_df[exp1_df['Model'] == 'MF'].iloc[0]

    ncf_row = exp1_df[exp1_df['Model'] == 'NCF'].iloc[0]

    lgc_row = exp1_df[exp1_df['Model'] == 'LightGCN'].iloc[0]

    print('\nExperiment 1 — Graph vs no graph (dot product):')

    print(f"  Best dot-product model: {best_dot['Model']} (NDCG@20={best_dot['Test NDCG@20']:.4f}, MRR={best_dot['Test MRR']:.4f})")

    print(f"  MF: NDCG@20={mf_row['Test NDCG@20']:.4f}, MRR={mf_row['Test MRR']:.4f} | LightGCN: NDCG@20={lgc_row['Test NDCG@20']:.4f}, MRR={lgc_row['Test MRR']:.4f}")

    graph_helps = lgc_row['Test MRR'] > mf_row['Test MRR']

    print(f"  Graph helps dot product: {('YES' if graph_helps else 'NO')}")

    print(f"  LightGCN vs NCF: NDCG@20={lgc_row['Test NDCG@20']:.4f} vs {ncf_row['Test NDCG@20']:.4f}, MRR={lgc_row['Test MRR']:.4f} vs {ncf_row['Test MRR']:.4f}")

    exp2_sorted = exp2_df.sort_values('Test MRR', ascending=False)

    best_decoder = exp2_sorted.iloc[0]

    dot_decoder = exp2_df[exp2_df['Decoder'] == 'dot'].iloc[0]

    mlp_decoder = exp2_df[exp2_df['Decoder'] == 'mlp'].iloc[0]

    print('\nExperiment 2 — Decoder ablation (same LightGCN encoder):')

    print(f"  Best decoder: {best_decoder['Decoder']} (NDCG@20={best_decoder['Test NDCG@20']:.4f}, MRR={best_decoder['Test MRR']:.4f})")

    dot_enough = dot_decoder['Test MRR'] >= mlp_decoder['Test MRR']

    print(f"  Dot product sufficient vs MLP: {('YES' if dot_enough else 'NO')}")

    cosine_row = exp2_df[exp2_df['Decoder'] == 'cosine'].iloc[0]

    print(f"  Dot:   NDCG@20={dot_decoder['Test NDCG@20']:.4f}, MRR={dot_decoder['Test MRR']:.4f}\n  Cosine: NDCG@20={cosine_row['Test NDCG@20']:.4f}, MRR={cosine_row['Test MRR']:.4f}\n  MLP:   NDCG@20={mlp_decoder['Test NDCG@20']:.4f}, MRR={mlp_decoder['Test MRR']:.4f}")

    print('\nSuggested next steps (Experiments 3–5):')

    if graph_helps and dot_enough:

        print('  → Run Exp 3 (depth/dim Pareto) and Exp 4 (embedding geometry)')

        print('  → Consider Exp 5 (L2 reg, multi-negative BPR) if dot wins')

    elif not graph_helps:

        print('  → Investigate graph construction / propagation depth before Exp 3')

    else:

        print('  → MLP beats dot: analyze when/why; still run Exp 3 Pareto for cost tradeoff')


def main():

    import argparse

    parser = argparse.ArgumentParser(description='Run thesis experiments 1–5')

    parser.add_argument('--experiments', type=int, nargs='+', choices=[1, 2, 3, 4, 5], help='Which experiments to run (default: all)')

    parser.add_argument('--plots-only', action='store_true', help='Regenerate plots and summaries from saved CSVs without retraining')

    args = parser.parse_args()

    experiments = args.experiments or [1, 2, 3, 4, 5]

    os.makedirs(PLOTS_DIR, exist_ok=True)

    if args.plots_only:

        if 1 in experiments and os.path.exists(EXP1_CSV):

            plot_exp1(pd.read_csv(EXP1_CSV), PLOTS_DIR)

        if 2 in experiments and os.path.exists(EXP2_CSV):

            plot_exp2(pd.read_csv(EXP2_CSV), PLOTS_DIR)

        if 1 in experiments and 2 in experiments and os.path.exists(EXP1_CSV) and os.path.exists(EXP2_CSV):

            summarize_findings(pd.read_csv(EXP1_CSV), pd.read_csv(EXP2_CSV))

        if 3 in experiments and os.path.exists(EXP3_LAYERS_CSV) and os.path.exists(EXP3_DIMS_CSV):

            plot_exp3(pd.read_csv(EXP3_LAYERS_CSV), pd.read_csv(EXP3_DIMS_CSV), PLOTS_DIR)

        if 4 in experiments and os.path.exists(EXP4_GEOMETRY_CSV) and os.path.exists(EXP4_DECODER_CSV):

            stats = {'cosine_sims': np.load(os.path.join(CACHE_DIR, 'exp4_cosine_sims.npy')), 'user_norms': np.load(os.path.join(CACHE_DIR, 'exp4_user_norms.npy')), 'movie_norms': np.load(os.path.join(CACHE_DIR, 'exp4_movie_norms.npy'))}

            plot_exp4(pd.read_csv(EXP4_GEOMETRY_CSV), pd.read_csv(EXP4_DECODER_CSV), stats, PLOTS_DIR)

        if 5 in experiments and os.path.exists(EXP5_CSV):

            plot_exp5(pd.read_csv(EXP5_CSV), PLOTS_DIR)

        if 3 in experiments and 4 in experiments and (5 in experiments):

            if all((os.path.exists(p) for p in [EXP3_LAYERS_CSV, EXP3_DIMS_CSV, EXP4_GEOMETRY_CSV, EXP4_DECODER_CSV, EXP5_CSV])):

                summarize_exp345(pd.read_csv(EXP3_LAYERS_CSV), pd.read_csv(EXP3_DIMS_CSV), pd.read_csv(EXP4_GEOMETRY_CSV), pd.read_csv(EXP4_DECODER_CSV), pd.read_csv(EXP5_CSV))

        print('\nPlots regenerated from saved CSVs.')

        return

    print('Loading MovieLens 1M...')

    dataset = get_movielens_dataset()

    (train_data, val_data, test_data, num_u, num_m) = prepare_data_splits(dataset)

    num_nodes = num_u + num_m

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Nodes={num_nodes}, device={device}')

    exp1_df = exp2_df = None

    layers_df = dims_df = geometry_df = decoder_df = exp5_df = None

    exp4_stats = None

    if 1 in experiments:

        exp1_df = experiment_1(train_data, val_data, test_data, num_nodes, num_u, num_m, device)

        plot_exp1(exp1_df, PLOTS_DIR)

    if 2 in experiments:

        exp2_df = experiment_2(train_data, val_data, test_data, num_nodes, num_u, num_m, device)

        plot_exp2(exp2_df, PLOTS_DIR)

    if 1 in experiments and 2 in experiments:

        summarize_findings(exp1_df, exp2_df)

    if 3 in experiments:

        (layers_df, dims_df) = experiment_3(train_data, val_data, test_data, num_nodes, num_u, num_m, device)

        plot_exp3(layers_df, dims_df, PLOTS_DIR)

    if 4 in experiments:

        (geometry_df, decoder_df, exp4_stats) = experiment_4(train_data, val_data, test_data, num_nodes, num_u, num_m, device)

        plot_exp4(geometry_df, decoder_df, exp4_stats, PLOTS_DIR)

    if 5 in experiments:

        exp5_df = experiment_5(train_data, val_data, test_data, num_nodes, num_u, num_m, device)

        plot_exp5(exp5_df, PLOTS_DIR)

    if 3 in experiments and 4 in experiments and (5 in experiments):

        summarize_exp345(layers_df, dims_df, geometry_df, decoder_df, exp5_df)

    print(f'\nDone. Plots saved to {PLOTS_DIR}/')

if __name__ == '__main__':

    main()

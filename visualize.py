import pandas as pd

import matplotlib.pyplot as plt

import os

RANKING_METRICS = ['Test NDCG@20', 'Test Recall@20', 'Test Hit Rate@20', 'Test MRR']

PRECISION_RECALL_F1 = ['Test Precision@20', 'Test Recall@20', 'Test F1@20']


def _available_columns(df, columns):

    return [column for column in columns if column in df.columns]


def _grouped_barplot(df, columns, title, ylabel, output_path, ylim=None):

    models = df['Model'].tolist()

    x = range(len(models))

    width = 0.8 / len(columns)

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974']

    (fig, ax) = plt.subplots(figsize=(12, 6))

    for (idx, column) in enumerate(columns):

        offsets = [pos + (idx - (len(columns) - 1) / 2) * width for pos in x]

        values = df[column].tolist()

        bars = ax.bar(offsets, values, width=width, label=column, color=colors[idx % len(colors)])

        for bar in bars:

            height = bar.get_height()

            if pd.isna(height):

                continue

            ax.annotate(f'{height:.3f}', (bar.get_x() + bar.get_width() / 2.0, height), ha='center', va='bottom', xytext=(0, 4), textcoords='offset points', fontsize=8)

    ax.set_xticks(list(x))

    ax.set_xticklabels(models)

    ax.set_title(title, fontsize=14)

    ax.set_ylabel(ylabel)

    ax.legend()

    ax.grid(axis='y', linestyle='--', alpha=0.4)

    if ylim is not None:

        ax.set_ylim(*ylim)

    fig.tight_layout()

    fig.savefig(output_path, dpi=300)

    plt.close(fig)


def create_visualizations(csv_file='results.csv', output_dir='plots'):

    if not os.path.exists(csv_file):

        print(f'Error: {csv_file} not found. Please run main.py first.')

        return

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_file)

    ranking_cols = _available_columns(df, RANKING_METRICS)

    if ranking_cols:

        _grouped_barplot(df, ranking_cols, 'Recommendation Accuracy: NDCG, Recall, Hit Rate, and MRR', 'Score', os.path.join(output_dir, 'ranking_accuracy.png'))

    prf_cols = _available_columns(df, PRECISION_RECALL_F1)

    if prf_cols:

        _grouped_barplot(df, prf_cols, 'Top-20 Cutoff: Precision, Recall, and F1@20', 'Score', os.path.join(output_dir, 'precision_recall_f1.png'))

    (fig, ax) = plt.subplots(figsize=(8, 6))

    bars = ax.bar(df['Model'], df['Peak VRAM (MB)'], color='#C44E52')

    ax.set_title('Memory Efficiency: Peak GPU VRAM Consumption', fontsize=14)

    ax.set_ylabel('MegaBytes (MB)')

    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for bar in bars:

        ax.annotate(f'{int(bar.get_height())} MB', (bar.get_x() + bar.get_width() / 2.0, bar.get_height()), ha='center', va='bottom', xytext=(0, 4), textcoords='offset points')

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'hardware_vram.png'), dpi=300)

    plt.close(fig)

    (fig, ax) = plt.subplots(figsize=(8, 6))

    bars = ax.bar(df['Model'], df['Avg Epoch Time (s)'], color='#4C72B0')

    ax.set_title('Computational Efficiency: Average Epoch Training Time', fontsize=14)

    ax.set_ylabel('Seconds (s)')

    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for bar in bars:

        ax.annotate(f'{bar.get_height():.3f}s', (bar.get_x() + bar.get_width() / 2.0, bar.get_height()), ha='center', va='bottom', xytext=(0, 4), textcoords='offset points')

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'hardware_epoch_time.png'), dpi=300)

    plt.close(fig)

    print(f"Visualizations successfully generated in the '{output_dir}' directory.")


def plot_embedding_heatmap(csv_file='embedding_size_results.csv', output_dir='plots'):

    if not os.path.exists(csv_file):

        print(f'Error: {csv_file} not found.')

        return

    os.makedirs(output_dir, exist_ok=True)

    df_emb = pd.read_csv(csv_file)

    pivot_df = df_emb.pivot(index='Model', columns='Embedding Size', values='Test Recall@20')

    (fig, ax) = plt.subplots(figsize=(8, 6))

    im = ax.imshow(pivot_df.values, cmap='YlGnBu', aspect='auto')

    ax.set_xticks(range(len(pivot_df.columns)))

    ax.set_xticklabels(pivot_df.columns)

    ax.set_yticks(range(len(pivot_df.index)))

    ax.set_yticklabels(pivot_df.index)

    ax.set_title('Heat Map: Test Recall@20 by Model and Embedding Size')

    for row in range(pivot_df.shape[0]):

        for col in range(pivot_df.shape[1]):

            ax.text(col, row, f'{pivot_df.values[row, col]:.4f}', ha='center', va='center', color='black', fontsize=9)

    fig.colorbar(im, ax=ax, label='Test Recall@20')

    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, 'embedding_size_heatmap.png'), dpi=300)

    plt.close(fig)

    print(f"Heatmap successfully generated in the '{output_dir}' directory.")

if __name__ == '__main__':

    create_visualizations()

    plot_embedding_heatmap()

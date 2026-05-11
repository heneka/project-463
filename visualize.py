import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_visualizations(csv_file="results.csv", output_dir="plots"):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please run main.py first.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file)

    sns.set_theme(style="whitegrid", palette="muted")

    plt.figure(figsize=(10, 6))
    metrics_melted = df.melt(
        id_vars=["Model"], 
        value_vars=["Test NDCG@20", "Test Recall@20"], 
        var_name="Metric", 
        value_name="Score"
    )
    ax1 = sns.barplot(data=metrics_melted, x="Model", y="Score", hue="Metric")
    plt.title("Recommendation Accuracy: NDCG vs Recall", fontsize=14)
    plt.ylabel("Score")
    plt.ylim(0.8, 1.0) 
    
    for p in ax1.patches:
        ax1.annotate(format(p.get_height(), '.3f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ranking_accuracy.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    ax2 = sns.barplot(data=df, x="Model", y="Peak VRAM (MB)", hue="Model", palette="Reds_d", legend=False)
    plt.title("Memory Efficiency: Peak GPU VRAM Consumption", fontsize=14)
    plt.ylabel("MegaBytes (MB)")
    
    for p in ax2.patches:
        ax2.annotate(f"{int(p.get_height())} MB", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hardware_vram.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    ax3 = sns.barplot(data=df, x="Model", y="Avg Epoch Time (s)", hue="Model", palette="Blues_d", legend=False)
    plt.title("Computational Efficiency: Average Epoch Training Time", fontsize=14)
    plt.ylabel("Seconds (s)")
    
    for p in ax3.patches:
        ax3.annotate(f"{p.get_height():.3f}s", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hardware_epoch_time.png"), dpi=300)
    plt.close()

    print(f"Visualizations successfully generated in the '{output_dir}' directory.")

def plot_embedding_heatmap(csv_file="embedding_size_results.csv", output_dir="plots"):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df_emb = pd.read_csv(csv_file)
    
    pivot_df = df_emb.pivot(index="Model", columns="Embedding Size", values="Test Recall@20")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Test Recall@20'})
    plt.title("Heat Map: Test Recall@20 by Model and Embedding Size")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "embedding_size_heatmap.png"), dpi=300)
    plt.close()
    
    print(f"Heatmap successfully generated in the '{output_dir}' directory.")

if __name__ == "__main__":
    create_visualizations()
    plot_embedding_heatmap()

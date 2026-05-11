import torch
import time
from tqdm import tqdm
from torch_geometric.utils import structured_negative_sampling
from metrics import evaluate_metrics


def get_node_embeddings(model, edge_index):
    if hasattr(model, 'encoder'):
        return model.encoder.get_embedding(edge_index)
    return model(edge_index)


def preview_embedding(vector, dims=8):
    return [round(value, 4) for value in vector[:dims].detach().cpu().tolist()]


def recommend_movies_for_user(model, edge_index, user_id, num_users, num_movies, top_k=5, train_edge_index=None):
    model.eval()
    with torch.no_grad():
        embeddings = get_node_embeddings(model, edge_index)
        user_embedding = embeddings[user_id]
        movie_embeddings = embeddings[num_users:num_users + num_movies]
        scores = torch.matmul(movie_embeddings, user_embedding)

        if train_edge_index is not None:
            seen_mask = train_edge_index[0] == user_id
            seen_movies = train_edge_index[1, seen_mask]
            seen_movies = seen_movies[seen_movies >= num_users] - num_users
            if seen_movies.numel() > 0:
                scores[seen_movies] = float('-inf')

        top_scores, top_indices = torch.topk(scores, k=min(top_k, num_movies))
        recommendations = []
        for movie_index, score in zip(top_indices.tolist(), top_scores.tolist()):
            recommendations.append({
                'movie_node_id': num_users + movie_index,
                'movie_index': movie_index,
                'score': round(score, 4),
            })
        return user_embedding, recommendations


def get_positive_matches(edge_label_index, edge_label, user_id, num_users):
    user_mask = edge_label_index[0] == user_id
    positive_mask = edge_label == 1
    matched_movies = edge_label_index[1, user_mask & positive_mask]
    return [int(movie_node_id - num_users) for movie_node_id in matched_movies.tolist()]

def sample_negative_edges(edge_index, num_nodes):
    """
    Given positive edges, uniformly samples negative edges.
    """
    i, j, k = structured_negative_sampling(edge_index, num_nodes=num_nodes)
    pos_edge_index = torch.stack((i, j), dim=0)
    neg_edge_index = torch.stack((i, k), dim=0)
    return pos_edge_index, neg_edge_index

def train_one_epoch(model, optimizer, train_data, num_nodes):
    model.train()
    optimizer.zero_grad()
    
    pos_edge_index, neg_edge_index = sample_negative_edges(train_data.edge_index, num_nodes)
    
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    loss = model.get_bpr_loss(train_data.edge_index, pos_edge_index, neg_edge_index)
    loss.backward()
    optimizer.step()
    
    end_time = time.time()
    epoch_time = end_time - start_time
    
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) # in MB
    else:
        peak_vram = 0.0
        
    return loss.item(), epoch_time, peak_vram

def train(model, train_data, val_data, test_data, num_nodes, epochs=50, lr=0.001, device='cpu', return_model=False):
    model = model.to(device)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'loss': [],
        'val_recall': [],
        'epoch_times': [],
        'peak_vram': []
    }
    
    print(f"Training {model.__class__.__name__} on {device}")
    
    for epoch in range(1, epochs + 1):
        loss, ep_time, vram = train_one_epoch(model, optimizer, train_data, num_nodes=num_nodes)
        
        if epoch % 10 == 0 or epoch == epochs:
            metrics = evaluate_metrics(model, val_data, val_data.edge_label_index, val_data.edge_label)
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Recall: {metrics['Recall@K']:.4f} | VRAM: {vram:.2f} MB | Time: {ep_time:.3f} s")
            history['val_recall'].append(metrics['Recall@K'])
            
        history['loss'].append(loss)
        history['epoch_times'].append(ep_time)
        history['peak_vram'].append(vram)
        
    print("Evaluating on Test Set...")
    test_metrics = evaluate_metrics(model, test_data, test_data.edge_label_index, test_data.edge_label)
    print(f"Test Metrics: {test_metrics}")
    
    if return_model:
        return history, test_metrics, model
    return history, test_metrics

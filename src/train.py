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

            seen_movies = set()

            for (src, dst) in zip(train_edge_index[0].tolist(), train_edge_index[1].tolist()):

                if src == user_id and dst >= num_users:

                    seen_movies.add(dst - num_users)

                elif dst == user_id and src >= num_users:

                    seen_movies.add(src - num_users)

            if seen_movies:

                seen_indices = torch.tensor(list(seen_movies), device=scores.device, dtype=torch.long)

                scores[seen_indices] = float('-inf')

        (top_scores, top_indices) = torch.topk(scores, k=min(top_k, num_movies))

        recommendations = []

        for (movie_index, score) in zip(top_indices.tolist(), top_scores.tolist()):

            recommendations.append({'movie_node_id': num_users + movie_index, 'movie_index': movie_index, 'score': round(score, 4)})

        return (user_embedding, recommendations)


def get_positive_matches(edge_label_index, edge_label, user_id, num_users):

    user_mask = edge_label_index[0] == user_id

    user_source_mask = edge_label_index[0] < num_users

    positive_mask = edge_label == 1

    matched_movies = edge_label_index[1, user_mask & user_source_mask & positive_mask]

    return [int(movie_node_id - num_users) for movie_node_id in matched_movies.tolist()]


def sample_bipartite_negative_edges(edge_index, num_users, num_movies):

    user_to_movie = edge_index[0] < num_users

    pos_users = edge_index[0][user_to_movie]

    pos_movies = edge_index[1][user_to_movie]

    num_pos = pos_users.size(0)

    neg_movies = torch.randint(num_users, num_users + num_movies, (num_pos,), device=edge_index.device)

    pos_edge_index = torch.stack((pos_users, pos_movies), dim=0)

    neg_edge_index = torch.stack((pos_users, neg_movies), dim=0)

    return (pos_edge_index, neg_edge_index)


def embedding_l2_penalty(model):

    if hasattr(model, 'encoder') and hasattr(model.encoder, 'embedding'):

        return model.encoder.embedding.weight.pow(2).mean()

    if hasattr(model, 'embedding'):

        return model.embedding.weight.pow(2).mean()

    return torch.tensor(0.0, device=next(model.parameters()).device)


def sample_negative_edges(edge_index, num_nodes, num_users=None, num_movies=None, bipartite=False, num_negatives=1):

    pos_edge_index = None

    neg_edge_indices = []

    for _ in range(num_negatives):

        if bipartite and num_users is not None and (num_movies is not None):

            (pos, neg) = sample_bipartite_negative_edges(edge_index, num_users, num_movies)

        else:

            (i, j, k) = structured_negative_sampling(edge_index, num_nodes=num_nodes)

            pos = torch.stack((i, j), dim=0)

            neg = torch.stack((i, k), dim=0)

        if pos_edge_index is None:

            pos_edge_index = pos

        neg_edge_indices.append(neg)

    return (pos_edge_index, neg_edge_indices)


def train_one_epoch(model, optimizer, train_data, num_nodes, num_users=None, num_movies=None, num_negatives=1, l2_reg=0.0):

    model.train()

    optimizer.zero_grad()

    use_bipartite = getattr(model, 'decoder_type', 'dot') == 'dot' and (not hasattr(model, 'encoder'))

    (pos_edge_index, neg_edge_indices) = sample_negative_edges(train_data.edge_index, num_nodes, num_users=num_users, num_movies=num_movies, bipartite=use_bipartite, num_negatives=num_negatives)

    torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    loss = torch.tensor(0.0, device=train_data.edge_index.device)

    for neg_edge_index in neg_edge_indices:

        loss = loss + model.get_bpr_loss(train_data.edge_index, pos_edge_index, neg_edge_index)

    loss = loss / len(neg_edge_indices)

    if l2_reg > 0:

        loss = loss + l2_reg * embedding_l2_penalty(model)

    loss.backward()

    optimizer.step()

    end_time = time.time()

    epoch_time = end_time - start_time

    if torch.cuda.is_available():

        peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 2

    else:

        peak_vram = 0.0

    return (loss.item(), epoch_time, peak_vram)


def train(model, train_data, val_data, test_data, num_nodes, num_users, num_movies, epochs=50, lr=0.001, device='cpu', num_negatives=1, l2_reg=0.0, return_model=False):

    model = model.to(device)

    train_data = train_data.to(device)

    val_data = val_data.to(device)

    test_data = test_data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'loss': [], 'val_recall': [], 'epoch_times': [], 'peak_vram': []}

    print(f'Training {model.__class__.__name__} on {device}')

    for epoch in range(1, epochs + 1):

        (loss, ep_time, vram) = train_one_epoch(model, optimizer, train_data, num_nodes=num_nodes, num_users=num_users, num_movies=num_movies, num_negatives=num_negatives, l2_reg=l2_reg)

        if epoch % 10 == 0 or epoch == epochs:

            metrics = evaluate_metrics(model=model, data=val_data, edge_label_index=val_data.edge_label_index, edge_label=val_data.edge_label, num_users=num_users, num_movies=num_movies, train_edge_index=train_data.edge_index)

            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Recall: {metrics['Recall@K']:.4f} | VRAM: {vram:.2f} MB | Time: {ep_time:.3f} s")

            history['val_recall'].append(metrics['Recall@K'])

        history['loss'].append(loss)

        history['epoch_times'].append(ep_time)

        history['peak_vram'].append(vram)

    print('Evaluating on Test Set...')

    test_metrics = evaluate_metrics(model=model, data=test_data, edge_label_index=test_data.edge_label_index, edge_label=test_data.edge_label, num_users=num_users, num_movies=num_movies, train_edge_index=train_data.edge_index)

    print(f'Test Metrics: {test_metrics}')

    if return_model:

        return (history, test_metrics, model)

    return (history, test_metrics)

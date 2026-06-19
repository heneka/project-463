import torch

import torch.nn.functional as F

import numpy as np


def compute_precision_at_k(preds, targets, k=20):

    precisions = []

    for user in targets.keys():

        if len(targets[user]) == 0:

            continue

        top_k_preds = preds[user][:k]

        hits = len(set(top_k_preds).intersection(set(targets[user])))

        precisions.append(hits / k)

    return np.mean(precisions) if precisions else 0.0


def compute_recall_at_k(preds, targets, k=20):

    recalls = []

    for user in targets.keys():

        if len(targets[user]) == 0:

            continue

        top_k_preds = preds[user][:k]

        hits = len(set(top_k_preds).intersection(set(targets[user])))

        recalls.append(hits / len(targets[user]))

    return np.mean(recalls) if recalls else 0.0


def compute_f1_at_k(precision, recall):

    if precision + recall == 0:

        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_ndcg_at_k(preds, targets, k=20):

    ndcgs = []

    for user in targets.keys():

        if len(targets[user]) == 0:

            continue

        top_k_preds = preds[user][:k]

        dcg = 0.0

        idcg = 0.0

        for (i, pred_item) in enumerate(top_k_preds):

            if pred_item in targets[user]:

                dcg += 1.0 / np.log2(i + 2)

        for i in range(min(k, len(targets[user]))):

            idcg += 1.0 / np.log2(i + 2)

        ndcgs.append(dcg / idcg if idcg > 0 else 0)

    return np.mean(ndcgs) if ndcgs else 0.0


def compute_hit_rate_at_k(preds, targets, k=20):

    hits = []

    for user in targets.keys():

        if len(targets[user]) == 0:

            continue

        top_k_preds = preds[user][:k]

        hit = len(set(top_k_preds).intersection(set(targets[user]))) > 0

        hits.append(float(hit))

    return np.mean(hits) if hits else 0.0


def compute_mrr(preds, targets):

    reciprocal_ranks = []

    for user in targets.keys():

        if len(targets[user]) == 0:

            continue

        target_set = set(targets[user])

        rr = 0.0

        for (rank, pred_item) in enumerate(preds[user], start=1):

            if pred_item in target_set:

                rr = 1.0 / rank

                break

        reciprocal_ranks.append(rr)

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def extract_user_targets(edge_label_index, edge_label, num_users):

    user_mask = edge_label_index[0] < num_users

    positive_mask = edge_label == 1

    eval_mask = user_mask & positive_mask

    users = edge_label_index[0][eval_mask].tolist()

    movies = edge_label_index[1][eval_mask].tolist()

    targets_dict = {}

    for (user_id, movie_id) in zip(users, movies):

        targets_dict.setdefault(user_id, []).append(movie_id)

    return targets_dict


def build_train_seen_movies(train_edge_index, num_users):

    seen = {user_id: set() for user_id in range(num_users)}

    sources = train_edge_index[0].tolist()

    destinations = train_edge_index[1].tolist()

    for (src, dst) in zip(sources, destinations):

        if src < num_users and dst >= num_users:

            seen[src].add(dst)

        elif dst < num_users and src >= num_users:

            seen[dst].add(src)

    return seen


def build_seen_mask(train_edge_index, num_users, num_movies, device):

    seen_mask = torch.zeros(num_users, num_movies, dtype=torch.bool, device=device)

    user_to_movie = (train_edge_index[0] < num_users) & (train_edge_index[1] >= num_users)

    users = train_edge_index[0][user_to_movie]

    movies = train_edge_index[1][user_to_movie] - num_users

    seen_mask[users, movies] = True

    movie_to_user = (train_edge_index[1] < num_users) & (train_edge_index[0] >= num_users)

    users = train_edge_index[1][movie_to_user]

    movies = train_edge_index[0][movie_to_user] - num_users

    seen_mask[users, movies] = True

    return seen_mask


def score_all_user_movie_pairs(model, user_emb, movie_emb, decoder_override=None):

    decoder_type = decoder_override or getattr(model, 'decoder_type', 'dot')

    if decoder_type == 'cosine':

        user_emb = F.normalize(user_emb, dim=-1)

        movie_emb = F.normalize(movie_emb, dim=-1)

        return user_emb @ movie_emb.T

    if decoder_type == 'mlp':

        (num_users, _) = user_emb.shape

        (num_movies, _) = movie_emb.shape

        interaction = model.interaction if hasattr(model, 'interaction') else model.mlp

        scores = torch.empty((num_users, num_movies), device=user_emb.device, dtype=user_emb.dtype)

        for user_id in range(num_users):

            user_vectors = user_emb[user_id].unsqueeze(0).expand(num_movies, -1)

            concat = torch.cat([user_vectors, movie_emb], dim=-1)

            scores[user_id] = interaction(concat).squeeze(-1)

        return scores

    return user_emb @ movie_emb.T


def evaluate_metrics(model, data, edge_label_index, edge_label, num_users, num_movies, train_edge_index, k=20, decoder_override=None):

    model.eval()

    with torch.no_grad():

        embeddings = model(data.edge_index)

        user_emb = embeddings[:num_users]

        movie_emb = embeddings[num_users:num_users + num_movies]

        targets_dict = extract_user_targets(edge_label_index, edge_label, num_users)

        if not targets_dict:

            return {'Precision@K': 0.0, 'Recall@K': 0.0, 'F1@K': 0.0, 'NDCG@K': 0.0, 'HitRate@K': 0.0, 'MRR': 0.0}

        scores = score_all_user_movie_pairs(model, user_emb, movie_emb, decoder_override=decoder_override)

        seen_mask = build_seen_mask(train_edge_index, num_users, num_movies, scores.device)

        scores = scores.masked_fill(seen_mask, float('-inf'))

        preds_dict = {}

        eval_users = sorted(targets_dict.keys())

        for user_id in eval_users:

            user_scores = scores[user_id]

            valid_count = torch.isfinite(user_scores).sum().item()

            top_k = min(k, valid_count)

            if top_k == 0:

                preds_dict[user_id] = []

                continue

            top_indices = torch.topk(user_scores, k=top_k).indices

            preds_dict[user_id] = (top_indices + num_users).tolist()

        precision = compute_precision_at_k(preds_dict, targets_dict, k=k)

        recall = compute_recall_at_k(preds_dict, targets_dict, k=k)

        f1 = compute_f1_at_k(precision, recall)

        ndcg = compute_ndcg_at_k(preds_dict, targets_dict, k=k)

        hit_rate = compute_hit_rate_at_k(preds_dict, targets_dict, k=k)

        mrr = compute_mrr(preds_dict, targets_dict)

    return {'Precision@K': precision, 'Recall@K': recall, 'F1@K': f1, 'NDCG@K': ndcg, 'HitRate@K': hit_rate, 'MRR': mrr}

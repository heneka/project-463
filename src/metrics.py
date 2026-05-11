import torch
import numpy as np
from sklearn.metrics import f1_score

def compute_recall_at_k(preds, targets, k=20):
    """
    Computes Recall@K.
    preds: List or dict grouping predicted items by user
    targets: List or dict grouping target actual items by user
    """
    recalls = []
    for user in targets.keys():
        if len(targets[user]) == 0:
            continue
        top_k_preds = preds[user][:k]
        
        hits = len(set(top_k_preds).intersection(set(targets[user])))
        recalls.append(hits / len(targets[user]))
        
    return np.mean(recalls) if recalls else 0.0

def compute_ndcg_at_k(preds, targets, k=20):
    """
    Computes NDCG@K.
    """
    ndcgs = []
    for user in targets.keys():
        if len(targets[user]) == 0:
            continue
        
        top_k_preds = preds[user][:k]
        
        dcg = 0.0
        idcg = 0.0
        
        for i, pred_item in enumerate(top_k_preds):
            if pred_item in targets[user]:
                dcg += 1.0 / np.log2(i + 2)
                
        for i in range(min(k, len(targets[user]))):
            idcg += 1.0 / np.log2(i + 2)
            
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
        
    return np.mean(ndcgs) if ndcgs else 0.0

def evaluate_metrics(model, data, edge_label_index, edge_label, k=20):
    """
    Predicts edges and calculates NDCG, Recall, and F1.
    For F1-Score, we treat this as a binary classification since we have 
    positive and negative edges provided by the split.
    """
    model.eval()
    with torch.no_grad():
        out = model(data.edge_index)
        
        u_emb = out[edge_label_index[0]]
        m_emb = out[edge_label_index[1]]
        
        if hasattr(model, 'mlp'): # NCF
            concat = torch.cat([u_emb, m_emb], dim=-1)
            scores = model.mlp(concat).squeeze()
        else:
            scores = model.decode(u_emb, m_emb)

        preds_cls = (scores > 0).float().cpu().numpy()
        targets_cls = edge_label.cpu().numpy()
        f1 = f1_score(targets_cls, preds_cls, average='macro', zero_division=0)
        
        
        num_users = len(torch.unique(edge_label_index[0]))
        users = edge_label_index[0].cpu().numpy()
        items = edge_label_index[1].cpu().numpy()
        scores_arr = scores.cpu().numpy()
        labels_arr = edge_label.cpu().numpy()
        
        preds_dict = {}
        targets_dict = {}
        
        for u, i, s, l in zip(users, items, scores_arr, labels_arr):
            if u not in preds_dict:
                preds_dict[u] = []
                targets_dict[u] = []
            preds_dict[u].append((i, s))
            if l == 1:
                targets_dict[u].append(i)
                
        for u in preds_dict:
            preds_dict[u].sort(key=lambda x: x[1], reverse=True)
            preds_dict[u] = [i for i, s in preds_dict[u]]
            
        recall = compute_recall_at_k(preds_dict, targets_dict, k=k)
        ndcg = compute_ndcg_at_k(preds_dict, targets_dict, k=k)

    return {"Recall@K": recall, "NDCG@K": ndcg, "F1": f1}

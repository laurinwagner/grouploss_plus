import torch
import numpy as np
import sklearn.metrics.pairwise

def assign_by_euclidian_at_k(distances, T, k):
    """ 
    X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
    k : for each sample, assign target labels of k nearest points
    """
    # get nearest points
    indices   = np.argsort(distances, axis = 1)[:, 1 : k + 1] 
    return np.array([[T[i] for i in ii] for ii in indices])

def calc_recall_at_k_inshop(distances, query_T, gallery_T, k):
    m = len(distances)
    match_counter = 0

    for i in range(m):
        pos_dist = distances[i][gallery_T == query_T[i]]
        neg_dist = distances[i][gallery_T != query_T[i]]
        thresh = torch.min(pos_dist).item()

        if torch.sum(neg_dist < thresh) < k:
            match_counter += 1
        
    return match_counter / m

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    Y = torch.from_numpy(Y)
    s = sum([1 for t, y in zip(T, Y) if t in y[:k]])
    return s / (1. * len(T))

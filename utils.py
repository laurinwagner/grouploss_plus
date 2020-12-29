
import torch.nn as nn
import torch.nn.functional as F
import evaluation
import torch
import logging
import sys
import numpy as np
import torch
from tqdm import tqdm
import pickle
import os
from sklearn.metrics.pairwise import pairwise_distances
import scipy
from scipy.spatial.distance import cdist

def re_rank(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):

    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)


    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist




def predict_batchwise_inshop(model, dataloader):
    device = "cuda"
    model_is_training = model.training
    model.eval()
    
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    _,J = model(J.cuda())

                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state
    
    return [torch.stack(A[i]) for i in range(len(A))]

def predict_batchwise(model, dataloader, net_type='bn_inception'):
    fc7s, L = [], []
    with torch.no_grad():
        for X, Y in dataloader:
            _, fc7 = model(X.cuda())
            fc7s.append(fc7.cpu())
            L.append(Y)
    fc7, Y = torch.cat(fc7s), torch.cat(L)
    return torch.squeeze(fc7), torch.squeeze(Y)


def evaluate(model, dataloader, nb_classes, rerank=False, net_type='bn_inception', dataroot='CARS'):
    model_is_training = model.training
    model.eval()

    # calculate embeddings with model, also get labels (non-batch-wise)
    X, T = predict_batchwise(model, dataloader, net_type)

    if(rerank):
         distances_q_g = pairwise_distances(X, X)
         distances_g_g= distances_q_g
         distances_q_q=distances_q_g
         distances=re_rank(distances_q_g, distances_q_q, distances_g_g, k1=50, k2=20, lambda_value=.94)
    
    else:
        distances=pairwise_distances(X)
    distances=torch.from_numpy(distances)

    if dataroot != 'Stanford':
        # calculate NMI with kmeans clustering
        nmi = evaluation.calc_normalized_mutual_information(T, evaluation.cluster_by_kmeans(X, nb_classes))
        logging.info("NMI: {:.3f}".format(nmi * 100))
    else:
        nmi = -1

    recall = []
    if dataroot != 'Stanford':
        Y = evaluation.assign_by_euclidian_at_k(distances, T, 8)
        which_nearest_neighbors = [1, 2, 4, 8]
    else:
        Y = evaluation.assign_by_euclidian_at_k(distances, T, 1000)
        which_nearest_neighbors = [1, 10, 100, 1000]
    
    for k in which_nearest_neighbors:
        r_at_k = evaluation.calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))

    model.train(model_is_training) # revert to previous training state
    return nmi, recall


def evaluate_Inshop(model, query_dataloader, gallery_dataloader,rerank=False):
    # nb_classes = query_dataloader.dataset.nb_classes()
    
    # # calculate embeddings with model and get targets
    query_X, query_T = predict_batchwise_inshop(model, query_dataloader)
    gallery_X, gallery_T = predict_batchwise_inshop(model, gallery_dataloader)
    
    query_X=query_X.cpu()
    gallery_X=gallery_X.cpu()

    if(rerank):
         distances_q_g = pairwise_distances(query_X, gallery_X)
         distances_g_g = pairwise_distances(gallery_X, gallery_X)
         distances_q_q = pairwise_distances(query_X, query_X)
         distances=re_rank(distances_q_g, distances_q_q, distances_g_g, k1=6, k2=2, lambda_value=0.4)
    else:
         distances=pairwise_distances(query_X,gallery_X)

    distances=torch.from_numpy(distances)
    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 10, 20, 30, 40, 50]:
        r_at_k = evaluation.calc_recall_at_k_inshop(distances, query_T, gallery_T, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
                
    return recall


def compute_class_distance(elements_p, elements_q, X_unit):

    dist = 0
    for el_p in elements_p:
        for el_q in elements_q:     
            dist += torch.norm(X_unit[el_p]-X_unit[el_q],p=2)

    dist = dist / (len(elements_p) * len(elements_q))
    return dist


def sort_by_distance(mat):
    mat = np.argsort(mat, axis=-1)
    dict_closest = {}
    for i in range(len(mat)):
        row = mat[i]
        dict_closest[i] = row[row != i]
    return dict_closest


def hierarchical_sampling(X, T, nb_classes):
    # normalize X into unit vector
    # X_norm = X.norm(dim=1)
    # X_unit = X / X_norm.view(X.shape[0], 1)
    X_unit = X

    # build a dictionary with key as classes and values as ids of occurrences
    dict_classes = {}
    for i in range(nb_classes):
        dict_classes[i] = []
    for i, el in enumerate(T):
        dict_classes[el.item()].append(i)

    # init the matrix of class distances
    mat_dist = np.zeros((nb_classes, nb_classes))
    for p in dict_classes:
        elements_p = dict_classes[p]
        for q in dict_classes:
            elements_q = dict_classes[q]
            if mat_dist[p, q] == 0.:
                pq_dist = compute_class_distance(elements_p, elements_q, X_unit)
                mat_dist[p, q] = pq_dist
                mat_dist[q, p] = pq_dist

    dict_dist = sort_by_distance(mat_dist)
    return dict_dist


def build_dict_classes(model, dataloader, nb_classes, net_type='bn_inception', dataroot='CARS'):
    model_is_training = model.training
    model.eval()

    # calculate embeddings with model, also get labels (non-batch-wise)
    X, T = predict_batchwise(model, dataloader, net_type)

    dict_class = hierarchical_sampling(X, T, nb_classes)
    model.train(model_is_training)
    return dict_class
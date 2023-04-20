"""Various utility functions."""
import numpy as np
import scipy.sparse as sp
import torch
from typing import Union
import networkx as nx
import random
# from sklearn.preprocessing import normalize, StandardScaler
from collections import Counter
import pandas as pd
from scipy.stats import norm
import pickle


def load_dataset(dir_net, header = False):

    with open(dir_net, mode='r') as f:
        if header:
            next(f)
        net_nA = []
        net_nB = []
        for line in f:
            na, nb = line.strip("\n").split("\t")
            net_nA.append(na)
            net_nB.append(nb)

    G = nx.Graph()
    G.add_edges_from(list(zip(net_nA, net_nB)))
    net_node = list(max(nx.connected_components(G), key = len))
    net_node = sorted(net_node)


    NODE2ID = dict()
    ID2NODE = dict()
    for idx in range(len(net_node)):
        ID2NODE[idx] = net_node[idx]
        NODE2ID[net_node[idx]] = idx

    row_idx = []
    col_idx = []
    val_idx = []
    for n_a, n_b in zip(net_nA, net_nB):
        if n_a in net_node and n_b in net_node:
            row_idx.append(NODE2ID[n_a])
            col_idx.append(NODE2ID[n_b])
            val_idx.append(1.0)
            row_idx.append(NODE2ID[n_b])
            col_idx.append(NODE2ID[n_a])
            val_idx.append(1.0)

    A = sp.csr_matrix((np.array(val_idx), (np.array(row_idx), np.array(col_idx))), shape=(len(net_node), len(net_node)))
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()

    G_lcc = nx.Graph()
    G_lcc.add_edges_from(list(zip(row_idx, col_idx)))

    return A, G_lcc, NODE2ID, ID2NODE


def load_snp(dir_net, header = False):

    with open(dir_net, mode='r') as f:
        if header:
            next(f)
        snp_id = []
        gene_id = []
        for line in f:
            snp, entrez = line.strip("\n").split("\t")
            snp_id.append(snp)
            gene_id.append(entrez)

    return set(gene_id)


def null_dist_score1(non_snp_genes, rand_num, node_num, curr_gene, cluster_results):
    rand_gene_score = []
    gene_clust = set(cluster_results[curr_gene])
    temp_non_snp_genes = non_snp_genes - set([curr_gene])
    for _ in range(rand_num):
        rand_set = random.sample(list(temp_non_snp_genes), node_num)
        rand_score = 0.0
        for rg in rand_set:
            rg_clust = set(cluster_results[rg])
            if len(rg_clust) > 0 and len(gene_clust) > 0:
                rand_score += len(gene_clust.intersection(rg_clust)) / len(rg_clust)
        rand_gene_score.append(rand_score)
    rand_mean = np.array(rand_gene_score).mean()
    rand_std = np.array(rand_gene_score).std()
    return rand_mean, rand_std


def FilterLCC(dir_lcc, all_genes, integrated_score):

    with open(dir_lcc, 'rb') as handle:
        lcc = pickle.load(handle)
    handle.close()

    filter_genes = []
    filter_score = []

    for gene, score in zip(all_genes, integrated_score):
        if int(gene) in lcc:
            filter_genes.append(int(gene))
            filter_score.append(score)


    return filter_genes, filter_score


def load_score(dir_score, header = False):

    with open(dir_score, mode='r') as f:
        if header:
            next(f)
        gene_id = []
        gene_score = []
        for line in f:
            id, score = line.strip("\n").split("\t")
            gene_id.append(int(id))
            gene_score.append(float(score))

    return gene_id, gene_score

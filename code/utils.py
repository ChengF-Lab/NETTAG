"""Various utility functions."""
import numpy as np
import scipy.sparse as sp
import torch
from typing import Union
import networkx as nx
import random
from sklearn.preprocessing import normalize, StandardScaler
from collections import Counter
from scipy.stats import norm
from sklearn.decomposition import TruncatedSVD
from itertools import islice
import os
import logging

def setup_logger(args):
    # Create a directory to store log files
    if not os.path.exists(args.dirlog):
        os.makedirs(args.dirlog)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    # Create a file handler that writes log messages to a file in the specified directory

    log_file = os.path.join(args.dirlog, 'log.txt')
    file_handler = logging.FileHandler(log_file)

    # Set the format of the log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger


def load_dataset(dir_net, bin_num, logger, header = False):
    '''

    change the remove 7316
    :param dir_net:
    :param bin_num:
    :param header:
    :return:
    '''

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
    G.remove_edges_from(nx.selfloop_edges(G))

    logger.info('There are {} of nodes in the loaded PPI!'.format(len(list(G.nodes()))))

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
    node_degree = {n: G_lcc.degree[n] for n in G_lcc.nodes}
    sorted_node_degree = dict(sorted(node_degree.items(), key=lambda item: item[1]))
    node2bin = np.array_split(list(sorted_node_degree.keys()), bin_num)

    return A, G_lcc, NODE2ID, ID2NODE, node2bin


# def load_dataset(dir_net, bin_num, logger):
#     '''
#
#     change the remove 7316
#     :param dir_net:
#     :param bin_num:
#     :param header:
#     :return:
#     '''
#
#     G = nx.read_adjlist(dir_net)
#     G.remove_edges_from(nx.selfloop_edges(G))
#
#     logger.info('There are {} of nodes in the loaded PPI!'.format(len(list(G.nodes()))))
#     net_node = list(max(nx.connected_components(G), key=len))
#     net_node = sorted(net_node)
#
#     NODE2ID = dict()
#     ID2NODE = dict()
#     for idx in range(len(net_node)):
#         ID2NODE[idx] = net_node[idx]
#         NODE2ID[net_node[idx]] = idx
#
#     row_idx = []
#     col_idx = []
#     val_idx = []
#     for e in G.edges:
#         if e[0] in net_node and e[1] in net_node:
#             row_idx.append(NODE2ID[e[0]])
#             col_idx.append(NODE2ID[e[1]])
#             val_idx.append(1.0)
#             row_idx.append(NODE2ID[e[1]])
#             col_idx.append(NODE2ID[e[0]])
#             val_idx.append(1.0)
#
#     A = sp.csr_matrix((np.array(val_idx), (np.array(row_idx), np.array(col_idx))), shape=(len(net_node), len(net_node)))
#     A = A.tolil()
#     A.setdiag(0)
#     A = A.tocsr()
#
#     G_lcc = nx.Graph()
#     G_lcc.add_edges_from(list(zip(row_idx, col_idx)))
#     node_degree = {n: G_lcc.degree[n] for n in G_lcc.nodes}
#     sorted_node_degree = dict(sorted(node_degree.items(), key=lambda item: item[1]))
#     node2bin = np.array_split(list(sorted_node_degree.keys()), bin_num)
#
#     return A, G_lcc, NODE2ID, ID2NODE, node2bin

# def load_dataset(dir_net, bin_num, header = True):
#
#     with open(dir_net, mode='r') as f:
#         if header:
#             next(f)
#         net_nA = []
#         net_nB = []
#         for line in f:
#             na, nb = line.strip("\n").split("\t")
#             if na != '7316' or nb != '7316':
#                 net_nA.append(na)
#                 net_nB.append(nb)
#
#     G = nx.Graph()
#     G.add_edges_from(list(zip(net_nA, net_nB)))
#     G.remove_edges_from(nx.selfloop_edges(G))
#     net_node = list(max(nx.connected_components(G), key = len))
#     net_node = sorted(net_node)
#
#
#     NODE2ID = dict()
#     ID2NODE = dict()
#     for idx in range(len(net_node)):
#         ID2NODE[idx] = net_node[idx]
#         NODE2ID[net_node[idx]] = idx
#
#     row_idx = []
#     col_idx = []
#     val_idx = []
#     for n_a, n_b in zip(net_nA, net_nB):
#         if n_a in net_node and n_b in net_node:
#             row_idx.append(NODE2ID[n_a])
#             col_idx.append(NODE2ID[n_b])
#             val_idx.append(1.0)
#             row_idx.append(NODE2ID[n_b])
#             col_idx.append(NODE2ID[n_a])
#             val_idx.append(1.0)
#
#     A = sp.csr_matrix((np.array(val_idx), (np.array(row_idx), np.array(col_idx))), shape=(len(net_node), len(net_node)))
#     A = A.tolil()
#     A.setdiag(0)
#     A = A.tocsr()
#
#     G_lcc = nx.Graph()
#     G_lcc.add_edges_from(list(zip(row_idx, col_idx)))
#     node_degree = {n: G_lcc.degree[n] for n in G_lcc.nodes}
#     sorted_node_degree = dict(sorted(node_degree.items(), key=lambda item: item[1]))
#     node2bin = np.array_split(list(sorted_node_degree.keys()), bin_num)
#
#     return A, G_lcc, NODE2ID, ID2NODE, node2bin



def l2_reg_loss(model, scale=1e-5):

    loss = 0.0
    for w in model.get_weights():
        loss += w.pow(2.).sum()
    return loss * scale



def to_sparse_tensor(matrix: Union[sp.spmatrix, torch.Tensor, np.array],
                     device, logger
                     ) -> Union[torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor]:


    if sp.issparse(matrix):
        coo = matrix.tocoo()
        indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
        values = torch.FloatTensor(coo.data)
        shape = torch.Size(coo.shape)
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    elif torch.is_tensor(matrix):
        row, col = matrix.nonzero().t()
        indices = torch.stack([row, col])
        values = matrix[row, col]
        shape = torch.Size(matrix.shape)
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    elif isinstance(matrix, np.ndarray):
        sparse_tensor = torch.FloatTensor(matrix)
    else:
        logger.error(f"matrix must be scipy.sparse or torch.Tensor or numpy.array (got {type(matrix)} instead).")
        # raise ValueError(f"matrix must be scipy.sparse or torch.Tensor or numpy.array (got {type(matrix)} instead).")
    if device:
        sparse_tensor = sparse_tensor.to(device)
    if isinstance(matrix, np.ndarray):
        return sparse_tensor
    else:
        return sparse_tensor.coalesce()

    # return sparse_tensor if isinstance(matrix, np.ndarray) else sparse_tensor.coalesce()



def feature_generator(A, n_comp, rand_seed, logger, preprocess = None):
    assert preprocess in ["None", "svd"]
    if preprocess == "None":
        feat = A
    elif preprocess == "svd":
        svd = TruncatedSVD(n_components=n_comp, n_iter=7, random_state=rand_seed)
        feat = svd.fit_transform(A)
        logger.info('svd explained variance ratio = {}'.format(svd.explained_variance_ratio_.sum()))
        # print("svd explained variance ratio = ", svd.explained_variance_ratio_.sum())
    return feat


def normalize_adj(adj, sparse=True):

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    if sparse:
        return res.tocoo()
    else:
        return res.todense()



def adj_polynomials(adj, k, sparse=True):

    adj_normalized = normalize_adj(adj, sparse=sparse)

    p_k = []
    if sparse:
        p_k.append(sp.eye(adj.shape[0]))
    else:
        p_k.append(np.eye(adj.shape[0]))

    p_k.append(adj_normalized)

    for p in range(2, k+1):
        p_k.append(sp.csr_matrix.power(adj_normalized, p))

    return p_k


def _volume(S, n, degG):
    '''
    volume(node) = sum(neighbors' degrees) + degree of itself
    paper link: https://arxiv.org/pdf/1112.0031.pdf
    :param S: node set S = node + node 1st-order neighbors
    :param degG: dictionary, key = node, value = node degree
    :return: volume(S) (numeric)
    '''

    S.add(n)
    vol = 0
    for m in S:
        vol += degG[m]
    return vol


def _edge(nxG, S, n):
    '''
    paper link: https://arxiv.org/pdf/1112.0031.pdf
    :param nxG:
    :param S: node set S = node + node 1st-order neighbors
    :return: edges(S)
    '''
    S.add(n)
    return 2 * len(nxG.subgraph(S).edges)


def graph_property(nxG):
    '''
    compute degree(node), volume(set), edges(set)
    input: networkx graph object
    :return: condG
    '''
    _degG = {n: len(set(nxG[n])) for n in set(nxG.nodes)} # node degree
    _volG = {n: _volume(set(nxG[n]), n, _degG) for n in set(nxG.nodes)} # volume(node's neighborhood)
    _edgeG = {n: _edge(nxG, set(nxG[n]), n) for n in set(nxG.nodes)} # edges(node's neighborhood)
    _cutG = {n: _volG[n] - _edgeG[n] for n in set(nxG.nodes)} # cut(node's neighborhood)
    _cvolG = {n: _volume(set(nxG.nodes) - set(nxG[n]), n, _degG) for n in set(nxG.nodes)} #vol(S bar)
    condG = {n: _cutG[n] / min(_volG[n], _cvolG[n]) for n in set(nxG.nodes)} # conductance(node's neighborhood)

    return condG



def cluster_number(G):
    '''
    determine the community number
    input: networkx object G
    output: community number
    '''
    cond_G = graph_property(G)
    clustercenter = [n for n in set(G.nodes) if
                      cond_G[n] < min([cond_G[m] for m in G[n]])]
    return len(clustercenter)



def cluster_infer(Z_pred, ID2NODE):
    clust_results = dict()
    for idx in range(Z_pred.shape[0]):
        clust_results[ID2NODE[idx]] = []
        clust_sets = np.where(Z_pred[idx, ] > 0)[0]
        for cidx in clust_sets:
            clust_results[ID2NODE[idx]].append(cidx)
    return clust_results



def load_snp(dir_net, header = False):

    with open(dir_net, mode='r') as f:
        if header:
            next(f)
        # snp_id = []
        gene_id = set()
        for line in f:
            _, id = line.strip("\n").split("\t")
            gene_id.add(id)

    return sorted(list(gene_id))




def null_dist_score(non_snp_genes, rand_num, node_num, curr_gene, cluster_results):
    rand_gene_score = []
    gene_clust = set(cluster_results[curr_gene])
    temp_non_snp_genes = non_snp_genes - set(curr_gene)
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




def BuildSumScore(cluster_results, snp_dict):
    all_genes = list(cluster_results.keys())
    all_genes_score = dict()
    for gwas_f in snp_dict:
        snp_clust = {gene: cluster_results[gene] for gene in snp_dict[gwas_f] if gene in cluster_results}
        non_snp_genes = set(all_genes) - set(snp_clust.keys())
        all_genes_score[gwas_f] = []
        for gene in all_genes:
            gene_clust = set(cluster_results[gene])
            gene_score = 0.0
            for sp in snp_clust:
                sp_clust = set(cluster_results[sp])
                if len(sp_clust) > 0 and len(gene_clust) > 0:
                    gene_score += len(gene_clust.intersection(sp_clust)) / len(sp_clust)

            rand_mean, rand_std = null_dist_score(non_snp_genes, 1000, len(snp_clust), gene, cluster_results)
            gene_zscore = (gene_score - rand_mean) / rand_std
            if 1 - norm.cdf(gene_zscore) < 0.05:
                all_genes_score[gwas_f].append(gene_score)
            else:
                all_genes_score[gwas_f].append(0)

    return all_genes, all_genes_score



def BuildIntegratedScore(all_genes, feature_score):
    normalized_feat_score = np.zeros((len(all_genes), len(feature_score)))
    count = 0
    for feat in feature_score:
        normalized_feat_score[:, count] = np.array(feature_score[feat])
        count += 1
    final_score = normalized_feat_score.sum(axis = 1)

    return all_genes, final_score



# def null_dist_score1(non_snp_genes, rand_num, node_num, curr_gene, cluster_results):
#     rand_gene_score = []
#     gene_clust = set(cluster_results[curr_gene])
#     temp_non_snp_genes = non_snp_genes - set(curr_gene)
#     for _ in range(rand_num):
#         rand_set = random.sample(list(temp_non_snp_genes), node_num)
#         rand_score = 0.0
#         for rg in rand_set:
#             rg_clust = set(cluster_results[rg])
#             if len(rg_clust) > 0 and len(gene_clust) > 0:
#                 rand_score += len(gene_clust.intersection(rg_clust)) / len(rg_clust)
#         rand_gene_score.append(rand_score)
#     rand_mean = np.array(rand_gene_score).mean()
#     rand_std = np.array(rand_gene_score).std()
#     return rand_mean, rand_std


# def null_dist_score1(non_snp_genes, rand_num, node_num, curr_gene, cluster_results):
#     rand_gene_score = []
#     gene_clust = set(cluster_results[curr_gene])
#     temp_non_snp_genes = non_snp_genes - set(curr_gene)
#     for _ in range(rand_num):
#         temp_score = []
#         for _ in range(rand_num):
#             rand_set = random.sample(list(temp_non_snp_genes), node_num)
#             rand_score = 0.0
#             for rg in rand_set:
#                 rg_clust = set(cluster_results[rg])
#                 if len(rg_clust) > 0 and len(gene_clust) > 0:
#                     rand_score += len(gene_clust.intersection(rg_clust)) / len(rg_clust)
#             temp_score.append(rand_score)
#         rand_gene_score.append(sum(temp_score) / rand_num)
#     rand_mean = np.array(rand_gene_score).mean()
#     rand_std = np.array(rand_gene_score).std()
#     return rand_mean, rand_std


# def null_dist_score1(non_snp_genes, rand_num, node_num, curr_gene, cluster_results):
#     gene_clust = set(cluster_results[curr_gene])
#     temp_non_snp_genes = non_snp_genes - set(curr_gene)
#     temp_score = []
#     for _ in range(rand_num):
#         rand_set = random.sample(list(temp_non_snp_genes), node_num)
#         rand_score = 0.0
#         for rg in rand_set:
#             rg_clust = set(cluster_results[rg])
#             if len(rg_clust) > 0 and len(gene_clust) > 0:
#                 rand_score += len(gene_clust.intersection(rg_clust)) / len(rg_clust)
#         temp_score.append(rand_score)
#
#     length_to_split = 10 * [int(rand_num // 10)]
#
#     temp_score = iter(temp_score)
#     slice_score = [list(islice(temp_score, elem))
#               for elem in length_to_split]
#
#     rand_gene_score = [sum(ele) / len(ele) for ele in slice_score]
#     rand_mean = np.array(rand_gene_score).mean()
#     rand_std = np.array(rand_gene_score).std()
#     return rand_mean, rand_std


# def null_dist_score1(non_snp_genes, rand_num, node_num, curr_gene, cluster_results, shared_seed):
#     random.seed(shared_seed)
#     rand_gene_score = []
#     gene_clust = set(cluster_results[curr_gene])
#     temp_non_snp_genes = set(non_snp_genes) - set(curr_gene)
#     temp_non_snp_genes = sorted(list(temp_non_snp_genes))
#     for _ in range(rand_num):
#         rand_set = random.sample(temp_non_snp_genes, node_num)
#         rand_score = 0.0
#         for rg in rand_set:
#             rg_clust = set(cluster_results[rg])
#             if len(rg_clust) > 0 and len(gene_clust) > 0:
#                 rand_score += len(gene_clust & rg_clust) / len(rg_clust)
#         rand_gene_score.append(rand_score)
#     return rand_gene_score


def null_dist_score1(non_snp_genes, rand_num, node_num, curr_gene, cluster_results, shared_seed):
    random.seed(shared_seed)
    rand_gene_score = []
    gene_clust = set(cluster_results[curr_gene])
    temp_non_snp_genes = set(non_snp_genes) - set([curr_gene])
    temp_non_snp_genes = sorted(list(temp_non_snp_genes))
    for _ in range(rand_num):
        rand_set = random.sample(temp_non_snp_genes, node_num)
        rand_score = 0.0
        for rg in rand_set:
            rg_clust = set(cluster_results[rg])
            if len(rg_clust) > 0 and len(gene_clust) > 0:
                rand_score += len(gene_clust & rg_clust) / len(rg_clust)
        rand_gene_score.append(rand_score)

    rand_mean = np.array(rand_gene_score).mean()
    rand_std = np.array(rand_gene_score).std()
    return rand_mean, rand_std


# def null_dist_score1(non_snp_genes, rand_num, node_num, curr_gene, cluster_results, shared_seed):
#     random.seed(shared_seed)
#     print("shared_seed = ", shared_seed)
#     rand_gene_score = []
#     gene_clust = set(cluster_results[curr_gene])
#     temp_non_snp_genes = set(non_snp_genes) - set(curr_gene)
#     temp_non_snp_genes = sorted(list(temp_non_snp_genes))
    # for _ in range(rand_num):
    #     rand_set = random.sample(temp_non_snp_genes, node_num)
    #     rand_score = 0.0
    #     for rg in rand_set:
    #         rg_clust = set(cluster_results[rg])
    #         if len(rg_clust) > 0 and len(gene_clust) > 0:
    #             rand_score += len(gene_clust.intersection(rg_clust)) / len(rg_clust)
    #     rand_gene_score.append(rand_score)
    # rand_mean = np.array(rand_gene_score).mean()
    # rand_std = np.array(rand_gene_score).std()
    # return rand_mean, rand_std


def sig_score(gene, cluster_results, snp_clust, non_snp_genes, shared_seed):
    gene_clust = set(cluster_results[gene])
    gene_score = 0.0
    for sp in snp_clust:
        sp_clust = set(cluster_results[sp])
        if len(sp_clust) > 0 and len(gene_clust) > 0:
            gene_score += len(gene_clust & sp_clust) / len(sp_clust)

    # rand_gene_score = null_dist_score1(non_snp_genes, 1000, len(snp_clust), gene, cluster_results, shared_seed)
    # sig_num = sum([s > gene_score for s in rand_gene_score])
    # if sig_num / 1000 < 0.01:
    # if 1 - norm.cdf(gene_zscore) < 0.05:
    rand_mean, rand_std = null_dist_score1(non_snp_genes, 1000, len(snp_clust), gene, cluster_results, shared_seed)
    gene_zscore = (gene_score - rand_mean) / rand_std

    if 1 - norm.cdf(gene_zscore) < 0.05:
        return gene_score
    else:
        return 0

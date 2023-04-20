import argparse
import os
import torch
import pyhocon
from collections import defaultdict
import multiprocessing
import numpy as np
import random
from functools import partial
from train import retrieve_clusters
from utils import load_snp, sig_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PyTorch Implementation of NETTAG')
    parser.add_argument('--pretrained', type=int,
                        help='whether or not using pretrained models')
    parser.add_argument('--rand_seed', type=int,
                        help='random seed')
    parser.add_argument('--bin_num', type=int, default=1000,
                        help='sort nodes into bins according to degrees')
    parser.add_argument('--dirnet', type=str,
                        help='directory of network')
    parser.add_argument('--dirfuncgeno', type=str,
                        help='directory of config file of functional genomics, see our example functional_genomics.config file')
    parser.add_argument('--preprocess', type=str, default=None,
                        help='feature preprocessing: None, svd')
    parser.add_argument('--n_comp', type=int,
                        help='reduced dimension')
    parser.add_argument('--adjpow', type=int, default=2,
                        help='element-wise power of adjacency matrix')
    parser.add_argument('--device', type=int,
                        help='which gpu to use if any')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of workers')
    parser.add_argument('--num_procs', type=int, default=8,
                        help='number of cores with multiprocessing when predict gene score.')
    parser.add_argument('--K', type=int,
                        help='cluster_number')
    parser.add_argument('--batch_norm', type=bool, default=True,
                        help='whether or not perform batch normalization')
    parser.add_argument('--epochs', type=int, default=50000,
                        help='number of epochs for iteration')
    parser.add_argument('--patience', type=int, default=5,
                        help='patience tolerance for early stopping')
    parser.add_argument('--val_step', type=int, default=500,
                        help='validation step to evaluate loss')
    parser.add_argument('--lr', type=float,
                        help='learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='minimum learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.9,
                        help='decrease coefficient of learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout probability')
    parser.add_argument('--hidden_size', nargs='+', help='hidden size of gnn, default = [2048, 1024]',
                        required=True)  # python main.py --hidden_size 2048 1024
    parser.add_argument('--dirresult', type=str,
                        help='output file')
    parser.add_argument('--dirlog', type=str,
                        help='logging file')
    args = parser.parse_args()

    if not os.path.isdir(args.dirresult):
        os.makedirs(args.dirresult)

    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)


    '''
    Step 1: retrieve clusters of PPIs
    '''

    cluster_results, save_file_name = retrieve_clusters(args)

    '''
    Step 2: get predicted scores
    '''
    config = pyhocon.ConfigFactory.parse_file(args.dirfuncgeno)

    snp_input = dict()
    for ele in config['func_genomics']:
        gene_with_ele = load_snp(config['func_genomics'][ele])
        snp_input[ele] = gene_with_ele

    all_genes = sorted(list(cluster_results.keys()))

    gene_reg_ele_score = defaultdict(list)

    for reg_ele in snp_input:

        snp_clust = {gene: cluster_results[gene] for gene in snp_input[reg_ele] if gene in cluster_results}

        non_snp_genes = sorted(list(set(all_genes) - set(snp_clust.keys())))

        pool = multiprocessing.Pool(processes=args.num_procs)

        partial_sig_score = partial(sig_score, cluster_results = cluster_results, snp_clust = snp_clust, non_snp_genes = non_snp_genes, shared_seed = args.rand_seed)

        all_genes_score = pool.map(partial_sig_score, all_genes)

        pool.close()

        pool.join()

        # all_genes_score = [sig_score(gene = gene, cluster_results = cluster_results, snp_clust = snp_clust, non_snp_genes = non_snp_genes) for gene in all_genes]

        f_out = open(args.dirresult + save_file_name + '_' + str(reg_ele) + '_' + 'score.txt', "w")

        for gene_id, score in zip(all_genes, all_genes_score):
            f_out.write(str(gene_id) + '\t' + str(score) + '\n')
            gene_reg_ele_score[gene_id].append(score)

        f_out.close()




    '''
    Step 3: generate likely AD-associated genes
    '''
    f_out = open(args.dirresult + save_file_name + "_integrated_score.txt", "w")
    f_out.write('node_id' + "\t" + 'score' + "\n")
    for gene_id in gene_reg_ele_score:
        score = sum(gene_reg_ele_score[gene_id])
        f_out.write(str(gene_id) + "\t" + str(score) + "\n")
    f_out.close()

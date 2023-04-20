import pickle
import numpy as np
from scipy.stats import norm

if __name__ == '__main__':

    with open('../data/ID2NODE.pkl', 'rb') as handle:
        ID2NODE = pickle.load(handle)
    handle.close()


    CpG_island_rwr = np.expand_dims(np.loadtxt('../data/CpG_island_imputation_0.6.txt', delimiter = '\t'), axis = 1)
    CTCF_rwr = np.expand_dims(np.loadtxt('../data/CTCF_imputation_0.6.txt', delimiter = '\t'), axis = 1)
    enhancer_rwr = np.expand_dims(np.loadtxt('../data/enhancer_imputation_0.6.txt', delimiter = '\t'), axis = 1)
    eqtl_rwr = np.expand_dims(np.loadtxt('../data/eQTL_imputation_0.6.txt', delimiter = '\t'), axis = 1)
    histone_rwr = np.expand_dims(np.loadtxt('../data/histone_imputation_0.6.txt', delimiter = '\t'), axis = 1)
    open_chromatin_rwr = np.expand_dims(np.loadtxt('../data/open_chromatin_imputation_0.6.txt', delimiter = '\t'), axis = 1)
    pfr_rwr = np.expand_dims(np.loadtxt('../data/pfr_imputation_0.6.txt', delimiter = '\t'), axis = 1)
    promoter_rwr = np.expand_dims(np.loadtxt('../data/promoter_imputation_0.6.txt',delimiter = '\t'), axis = 1)
    tf_rwr = np.expand_dims(np.loadtxt('../data/tf_imputation_0.6.txt', delimiter = '\t'), axis = 1)

    rwr_integrated_score = np.concatenate((CpG_island_rwr, CTCF_rwr, enhancer_rwr, eqtl_rwr,
                                   histone_rwr, open_chromatin_rwr, pfr_rwr,
                                   promoter_rwr, tf_rwr), axis = 1).sum(axis = 1)

    rwr_mean = np.mean(rwr_integrated_score)
    rwr_std = np.std(rwr_integrated_score)
    rwr_z_score = (rwr_integrated_score - rwr_mean) / rwr_std

    rwr_gene_id = []
    rwr_gene_score = []

    for idx in range(rwr_z_score.shape[0]):

        if rwr_z_score[idx] >= norm.ppf(0.99):
            rwr_gene_id.append(ID2NODE[idx])
            rwr_gene_score.append(rwr_z_score[idx])

    rwr_gene_score, rwr_gene_id = zip(*sorted(zip(rwr_gene_score, rwr_gene_id), reverse = True))

    f_out = open('../data/' + "rwr_prediction_0.6.txt", "w")
    f_out.write('node_id' + "\t" + 'score' + "\n")
    for gene_id, score in zip(rwr_gene_id, rwr_gene_score):
      f_out.write(str(gene_id) + "\t" + str(score) + "\n")
    f_out.close()
    

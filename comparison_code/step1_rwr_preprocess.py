import networkx as nx
from collections import defaultdict
import pandas as pd
import pickle

# === load protein-protein interactions === #
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

    G_lcc = G.subgraph(net_node)

    #add code:
    if len(list(nx.selfloop_edges(G_lcc))) > 0:
        G_lcc.remove_edges_from(nx.selfloop_edges(G_lcc))



    NODE2ID = dict()
    ID2NODE = dict()
    for idx in range(len(net_node)):
        ID2NODE[idx] = int(net_node[idx])
        NODE2ID[int(net_node[idx])] = idx



    ppi_remove_ubc_no_self_loop = defaultdict(set)

    for n in G_lcc.nodes:
        for m in G_lcc.neighbors(n):
            ppi_remove_ubc_no_self_loop[NODE2ID[int(n)]].add(NODE2ID[int(m)])

    row_idx = []
    col_idx = []
    for n in ppi_remove_ubc_no_self_loop:
        for m in ppi_remove_ubc_no_self_loop[n]:
            row_idx.append(n+1)
            col_idx.append(m+1)

    return NODE2ID, ID2NODE, row_idx, col_idx

# === load input genes associated with each regulatory element === #
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


if __name__ == '__main__':

    NODE2ID, ID2NODE, row_idx, col_idx = load_dataset('../data/ppi_remove_self_loop.txt')

    with open('../data/ID2NODE.pkl', 'wb') as handle:
        pickle.dump(ID2NODE, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

    rwr_mat = pd.DataFrame({
        'row_idx': row_idx,
        'col_idx': col_idx
    })

    rwr_mat.to_csv('../data/rwr_mat.txt', sep = '\t', index = None)


    # === Convert genes associated with CpG Island into index === #

    CpG_island = load_snp('../data/CpG_island_AD_mapped_snps_entrez_id_v3.txt', header=True)

    CpG_island_input = [0] * len(NODE2ID)
    for n in CpG_island:
        if int(n) in NODE2ID:
            CpG_island_input[NODE2ID[int(n)]] = 1


    CpG_island_input_df = pd.DataFrame({
        'val': CpG_island_input
    })

    CpG_island_input_df.to_csv('../data/rwr_CpG_island_input.txt', sep = '\t', index = None)


    # === Convert genes associated with CTCF into index === #

    CTCF = load_snp('../data/CTCF_AD_mapped_snps_entrez_id_v3.txt', header=True)

    CTCF_input = [0] * len(NODE2ID)
    for n in CTCF:
        if int(n) in NODE2ID:
            CTCF_input[NODE2ID[int(n)]] = 1


    CTCF_input_df = pd.DataFrame({
        'val': CTCF_input
    })

    CTCF_input_df.to_csv('../data/rwr_CTCF_input.txt', sep = '\t', index = None)



    # === Convert genes associated with Enhancer into index === #

    enhancer = load_snp('../data/enhancer_AD_mapped_snps_entrez_id_v3.txt', header=True)

    enhancer_input = [0] * len(NODE2ID)
    for n in enhancer:
        if int(n) in NODE2ID:
            enhancer_input[NODE2ID[int(n)]] = 1


    enhancer_input_df = pd.DataFrame({
        'val': enhancer_input
    })

    enhancer_input_df.to_csv('../data/rwr_enhancer_input.txt', sep = '\t', index = None)


    # === Convert genes associated with eQTL into index === #

    eQTL = load_snp('../data/eQTL_AD_mapped_snps_entrez_id_v3.txt', header=True)

    eQTL_input = [0] * len(NODE2ID)
    for n in eQTL:
        if int(n) in NODE2ID:
            eQTL_input[NODE2ID[int(n)]] = 1


    eQTL_input_df = pd.DataFrame({
        'val': eQTL_input
    })

    eQTL_input_df.to_csv('../data/rwr_eQTL_input.txt', sep = '\t', index = None)


    # === Convert genes associated with Histone into index === #

    histone = load_snp('../data/histone_AD_mapped_snps_entrez_id_v3.txt', header=True)

    histone_input = [0] * len(NODE2ID)
    for n in histone:
        if int(n) in NODE2ID:
            histone_input[NODE2ID[int(n)]] = 1


    histone_input_df = pd.DataFrame({
        'val': histone_input
    })

    histone_input_df.to_csv('../data/rwr_histone_input.txt', sep = '\t', index = None)


    # === Convert genes associated with Open Chromatin into index === #

    open_chromatin = load_snp('../data/open_chromatin_AD_mapped_snps_entrez_id_v3.txt', header=True)

    open_chromatin_input = [0] * len(NODE2ID)
    for n in open_chromatin:
        if int(n) in NODE2ID:
            open_chromatin_input[NODE2ID[int(n)]] = 1


    open_chromatin_input_df = pd.DataFrame({
        'val': open_chromatin_input
    })

    open_chromatin_input_df.to_csv('../data/rwr_open_chromatin_input.txt', sep = '\t', index = None)


    # === Convert genes associated with Promoter into index === #

    promoter = load_snp('../data/promoter_AD_mapped_snps_entrez_id_v3.txt', header=True)

    promoter_input = [0] * len(NODE2ID)
    for n in promoter:
        if int(n) in NODE2ID:
            promoter_input[NODE2ID[int(n)]] = 1


    promoter_input_df = pd.DataFrame({
        'val': promoter_input
    })

    promoter_input_df.to_csv('../data/rwr_promoter_input.txt', sep = '\t', index = None)


    # === Convert genes associated with Promoter Flanking Region into index === #

    pfr = load_snp('../data/pfr_AD_mapped_snps_entrez_id_v3.txt', header=True)

    pfr_input = [0] * len(NODE2ID)
    for n in pfr:
        if int(n) in NODE2ID:
            pfr_input[NODE2ID[int(n)]] = 1


    pfr_input_df = pd.DataFrame({
        'val': pfr_input
    })

    pfr_input_df.to_csv('../data/rwr_pfr_input.txt', sep = '\t', index = None)

    # === Convert genes associated with TF into index === #

    tf = load_snp('../data/TF_AD_mapped_snps_entrez_id_v3.txt', header=True)

    tf_input = [0] * len(NODE2ID)
    for n in tf:
        if int(n) in NODE2ID:
            tf_input[NODE2ID[int(n)]] = 1


    tf_input_df = pd.DataFrame({
        'val': tf_input
    })

    tf_input_df.to_csv('../data/rwr_tf_input.txt', sep = '\t', index = None)
    

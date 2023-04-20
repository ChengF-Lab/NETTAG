import os
import glob
import sys


def load_results(dir_file, header = True):
    with open(dir_file, mode='r') as f:
        if header:
            next(f)

        gene_id = []
        gene_score = []
        for line in f:
            nid, nscore = line.strip("\n").split("\t")
            gene_id.append(nid)
            gene_score.append(float(nscore))

    return gene_id, gene_score

if __name__ == "__main__":

    dir_output = sys.argv[1]

    sub_dirs = os.listdir(dir_output)

    pred_genes = []
    pred_scores = []

    for dir in sub_dirs:
        dir_pred = glob.glob(os.path.join(dir_output, dir, "*_integrated_score.txt"))[0]
        curr_genes, curr_scores = load_results(dir_pred)
        pred_genes.append(curr_genes)
        pred_scores.append(curr_scores)

    for glst in pred_genes[1:]:
        assert glst == pred_genes[0], "Order of genes are different!"
    
    sum_scores = [sum(score) for score in zip(*pred_scores)]
    ensembled_scores = [score / len(pred_scores) for score in sum_scores]

    if not os.path.isdir(os.path.join(dir_output, 'ensemble')):
        os.makedirs(os.path.join(dir_output, 'ensemble'))
    

    f_out = open(os.path.join(dir_output, 'ensemble', 'ensembled_complete_predictions.txt'), "w")

    for gid, score in zip(pred_genes[0], ensembled_scores):
      f_out.write(str(gid) + "\t" + str(score) + "\n")
    f_out.close()

    sorted_preds = sorted(zip(ensembled_scores, pred_genes[0]), reverse = True)
    top_pred_scores, top_pred_genes = zip(*sorted_preds)

    pred_num = int(sys.argv[2])

    f_out = open(os.path.join(dir_output, 'ensemble', 'ensembled_top_predictions.txt'), "w")

    for gid, score in zip(top_pred_genes[:pred_num], top_pred_scores[:pred_num]):
      f_out.write(str(gid) + "\t" + str(score) + "\n")
    f_out.close()


    


    


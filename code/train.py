import torch
import torch.nn.functional as F
from utils import setup_logger, load_dataset, feature_generator, adj_polynomials, l2_reg_loss, cluster_infer, cluster_number, \
    to_sparse_tensor
from sampler import get_edge_sampler
from model import PolyGCN
from loss import BerPossionLoss
import stopping
from optimizer import NAdam
import numpy as np
import math
import pickle
import os



def retrieve_clusters(args):

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
        torch.cuda.set_device(device)

    logger = setup_logger(args)

    A, G_lcc, NODE2ID, ID2NODE, node2bin = load_dataset(args.dirnet, args.bin_num, logger = logger)

    N = A.shape[0]  # total nodes

    logger.info('There are {} of nodes in the largest connected component (LCC) of PPI!'.format(N))

    batch_size = int(N * 0.8)

    K = cluster_number(G_lcc)
    
    if args.K > K:
        K = args.K


    feat = feature_generator(A, args.n_comp, rand_seed = args.rand_seed, logger = logger, preprocess=args.preprocess)
    #
    adj_polys = adj_polynomials(A, args.adjpow, sparse=True)

    adj_polys = [to_sparse_tensor(p, device = device, logger = logger) for p in adj_polys]

    sampler = get_edge_sampler(A, node2bin, args.bin_num, batch_size, num_workers=args.num_workers)

    gnn = PolyGCN(input_dim=feat.shape[1], hidden_dims=args.hidden_size, output_dim=K,
                  batch_norm=args.batch_norm, adj_pow=(args.adjpow + 1), dropout=args.dropout).to(device)

    criterion = BerPossionLoss(N, A.nnz)

    opt = NAdam(gnn.parameters(), lr=args.lr)

    val_loss = np.inf
    validation_fn = lambda: val_loss
    early_stopping = stopping.NoImprovementStopping(validation_fn, patience=args.patience)

    temp_hs = "_hidden-size"
    for idx in range(len(args.hidden_size)):
        temp_hs += ("_" + str(args.hidden_size[idx]))

    save_file_name = f"NAdam_A_poly{args.adjpow}_wd{args.weight_decay}_dropout{args.dropout}_lr_{args.lr}_K{K}_bs{batch_size}_patience{args.patience}_lrmin{args.lr_min}_max-epochs{args.epochs}_preprocess{args.preprocess}"
    save_file_name += temp_hs

    model_out = args.dirresult + save_file_name + ".pth"
    model_saver = stopping.ModelSaver(model = gnn, optimizer = opt, dir_model = model_out, logger = logger)


    f_out = open(args.dirresult + save_file_name + ".txt", "a")
    lr = args.lr

    if args.pretrained:

        with open(args.dirresult + save_file_name + '_cluster_results.pickle', 'rb') as handle:
            clust_results = pickle.load(handle)
        handle.close()

    else:
        for epoch, batch in enumerate(sampler):

            if epoch >= args.epochs:
                break

            if (epoch + 1) % args.val_step == 0:
                lr = max(lr * args.lr_decay, args.lr_min)
                opt = NAdam(gnn.parameters(), lr=lr)

            ones_idx, zeros_idx, batch_nodes = batch

            # Training step
            gnn.train()
            opt.zero_grad()
            A_batch = A[batch_nodes][:, batch_nodes]
            Z = F.relu(gnn(to_sparse_tensor(feat, device = device, logger = logger), adj_polys))
            Z_batch = Z[batch_nodes]
            criterion_batch = BerPossionLoss(len(batch_nodes), A_batch.nnz)

            loss = criterion_batch.loss_batch(Z_batch, ones_idx, zeros_idx)
            loss += l2_reg_loss(gnn, scale=args.weight_decay)
            loss.backward()
            opt.step()

            if epoch == 0 or (epoch + 1) % args.val_step == 0:

                with torch.no_grad():

                    gnn.eval()
                    # pos_full, neg_full, full_loss = criterion.loss_cpu(Z.cpu().detach().numpy(), A)
                    # pos_batch, neg_batch, batch_loss = criterion_batch.loss_cpu(Z_batch.cpu().detach().numpy(), A_batch)
                    # pos_val = (pos_full * criterion.num_edges - pos_batch * criterion_batch.num_edges) / (criterion.num_edges - criterion_batch.num_edges)
                    # neg_val = (neg_full * criterion.num_nonedges - neg_batch * criterion_batch.num_nonedges) / (criterion.num_nonedges - criterion_batch.num_nonedges)
                    # val_loss = (pos_val + neg_val) / 2

                    pos_full, neg_full, full_loss = criterion.loss_cpu(Z.cpu().detach().numpy(), A)
                    pos_batch, neg_batch, batch_loss = criterion_batch.loss_cpu(Z_batch.cpu().detach().numpy(), A_batch)
                    pos_val = (pos_full * criterion.num_edges - pos_batch * criterion_batch.num_edges) / (
                            criterion.num_edges - criterion_batch.num_edges)
                    neg_val = (neg_full * criterion.num_nonedges - neg_batch * criterion_batch.num_nonedges) / (
                            criterion.num_nonedges - criterion_batch.num_nonedges)
                    # val_loss = (pos_val + neg_val) / 2

                    val_ratio = (criterion.num_nonedges - criterion_batch.num_nonedges) / (
                                criterion.num_edges - criterion_batch.num_edges)
                    val_loss = (pos_val + val_ratio * neg_val) / (1 + val_ratio)

                    # print('*' * 100)
                    # print("#s of existing edges (total) = ", int(criterion.num_edges // 2))
                    # print("#s of existing edges  (training)= ", int(criterion_batch.num_edges // 2))
                    # print("#s of non-existing edges (total) = ", int(criterion.num_nonedges // 2))
                    # print("#s of non-existing edges (trainning) = ", int(criterion_batch.num_nonedges // 2))
                    # print("pos_batch = {}, neg_batch = {}".format(pos_batch, neg_batch))
                    # print("pos_val = {}, neg_val = {}".format(pos_val, neg_val))
                    # print("pos_full = {}, neg_full = {}".format(pos_full, neg_full))

                    logger.info('*' * 100)
                    logger.info("#s of existing edges (total) = {}".format(int(criterion.num_edges // 2)))
                    logger.info("#s of existing edges  (training) = {}".format(int(criterion_batch.num_edges // 2)))
                    logger.info("#s of non-existing edges (total) = {}".format(int(criterion.num_nonedges // 2)))
                    logger.info("#s of non-existing edges (training) = {}".format(int(criterion_batch.num_nonedges // 2)))
                    logger.info(f'Epoch {epoch:4d}, loss.train = {batch_loss:.4f}, loss.val = {val_loss:.4f}, loss.full = {full_loss:.4f}')
                    logger.info("pos_batch = {}, neg_batch = {}".format(pos_batch, neg_batch))
                    logger.info("pos_val = {}, neg_val = {}".format(pos_val, neg_val))
                    logger.info("pos_full = {}, neg_full = {}".format(pos_full, neg_full))


                    # Check if it's time for early stopping / to save the model
                    early_stopping.next_step()
                    if early_stopping.should_save():
                        logger.info('======= Write Loss to Output! ======')
                        f_out.write(
                            f'Epoch {epoch:4d}, loss.train = {batch_loss:.4f}, loss.val = {val_loss:.4f}, loss.full = {full_loss:.4f}')
                        f_out.write('\n')
                        model_saver.save()
                    if early_stopping.should_stop():
                        logger.info(f'Breaking due to early stopping at epoch {epoch}')
                        break

        f_out.close()

        model_saver.restore()

        gnn.eval()
        Z = F.relu(gnn(to_sparse_tensor(feat, device = device, logger = logger), adj_polys))

        Z_cpu = Z.cpu().detach().numpy()

        _, _, full_loss = criterion.loss_cpu(Z_cpu, A)

        logger.info(f'loss.full = {full_loss:.4f}')

        thresh = math.sqrt(-math.log(1 - 1 / N))
        Z_pred = Z_cpu > thresh

        clust_results = cluster_infer(Z_pred, ID2NODE)

        # with open(args.dirresult + save_file_name + '_cluster_results.pickle', 'wb') as handle:
        #     pickle.dump(clust_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # handle.close()

        # remove .pth file
        for out_file in os.listdir(args.dirresult):
            # Check if the file has the specified extension
            if out_file == model_out:
                # If the file has the extension, delete it
                os.remove(os.path.join(args.dirresult, out_file))

    return clust_results, save_file_name

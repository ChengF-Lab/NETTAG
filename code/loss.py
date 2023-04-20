import numpy as np
import torch


class BerPossionLoss():

    def __init__(self, num_nodes, num_edges):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_possible_edges = num_nodes**2 - num_nodes
        self.num_nonedges = self.num_possible_edges - self.num_edges
        edge_prob = self.num_edges / (self.num_nodes ** 2 - self.num_nodes)
        self.eps = -np.log(1 - edge_prob)
        self.true_ratio = self.num_nonedges / self.num_edges

    def loss_batch(self, emb, ones_idx, zeros_idx):

        # Loss for edges
        e1, e2 = ones_idx[:, 0], ones_idx[:, 1]
        edge_dots = torch.sum(emb[e1] * emb[e2], dim=1)
        loss_edges = -torch.mean(torch.log(-torch.expm1(-self.eps - edge_dots)))

        # Loss for non-edges
        ne1, ne2 = zeros_idx[:, 0], zeros_idx[:, 1]
        loss_nonedges = torch.mean(torch.sum(emb[ne1] * emb[ne2], dim=1))

        # sampled #s connected edges = #s non-connected edges
        return (loss_edges + loss_nonedges) / 2


    # def loss_cuda(self, emb, adj):
    #
    #     e1, e2 = adj.nonzero()
    #     edge_dots = torch.sum(emb[e1] * emb[e2], dim=1)
    #     loss_edges = -torch.sum(torch.log(-torch.expm1(-self.eps - edge_dots)))
    #
    #     # Correct for overcounting F_u * F_v for edges and nodes with themselves
    #     self_dots_sum = torch.sum(emb * emb)
    #     correction = self_dots_sum + torch.sum(edge_dots)
    #     sum_emb = torch.sum(emb, dim=0, keepdim=True).t()
    #     loss_nonedges = torch.sum(emb @ sum_emb) - correction
    #
    #     return (loss_edges / self.num_edges + loss_nonedges / self.num_nonedges) / 2


    def loss_cpu(self, emb, adj):

        e1, e2 = adj.nonzero()
        edge_dots = np.sum(emb[e1] * emb[e2], axis=1)
        loss_edges = -np.sum(np.log(-np.expm1(-self.eps - edge_dots)))

        # Correct for overcounting F_u * F_v for edges and nodes with themselves
        self_dots_sum = np.sum(emb * emb)
        correction = self_dots_sum + np.sum(edge_dots)
        sum_emb = np.transpose(np.sum(emb, axis = 0))
        loss_nonedges = np.sum(emb @ sum_emb) - correction

        pos_loss = loss_edges / self.num_edges
        neg_loss = loss_nonedges / self.num_nonedges

        # return pos_loss, neg_loss, (pos_loss + neg_loss) / 2
        return pos_loss, neg_loss, (pos_loss + self.true_ratio * neg_loss) / (1 + self.true_ratio)
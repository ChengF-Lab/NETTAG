import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def sparse_or_dense_dropout(x, p=0.5, training=True):
    if isinstance(x, (torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor)):
        new_values = F.dropout(x.values(), p=p, training=training)
        return torch.cuda.sparse.FloatTensor(x.indices(), new_values, x.size())
    else:
        return F.dropout(x, p=p, training=training)


class PolyGraphConvolution(nn.Module):

    def __init__(self, adj_pow, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(adj_pow, in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):

        # nn.init.xavier_normal_(self.weight, gain=1.0)
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x, poly_ls):
        node_num = x.shape[0]
        out = torch.zeros(node_num, self.out_features)
        out = out.to(self.weight.device)
        for idx in range(len(poly_ls)):
            out += poly_ls[idx] @ (x @ self.weight[idx,:,:])
        return out


class PolyGCN(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim, adj_pow, dropout, batch_norm=False):
        super().__init__()
        self.dropout = dropout
        layer_dims = np.concatenate([hidden_dims, [output_dim]]).astype(np.int32)
        self.layers = nn.ModuleList([PolyGraphConvolution(adj_pow, input_dim, layer_dims[0])])
        for idx in range(len(layer_dims) - 1):
            self.layers.append(PolyGraphConvolution(adj_pow, layer_dims[idx], layer_dims[idx + 1]))
        if batch_norm:
            self.batch_norm = [
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_dims
            ]
        else:
            self.batch_norm = None



    def forward(self, x, poly_ls):
        for idx, poly_gcn in enumerate(self.layers):
            if self.dropout != 0:
                x = sparse_or_dense_dropout(x, p=self.dropout, training=self.training)
            x = poly_gcn(x, poly_ls)
            if idx != len(self.layers) - 1:
                x = F.relu(x)
                if self.batch_norm is not None:
                    x = self.batch_norm[idx](x)
        return x

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]
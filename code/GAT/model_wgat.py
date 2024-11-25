import torch
import torch.nn as nn
import torch.nn.functional as F
from wgat_layer import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj, weighted_adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, weighted_adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj, weighted_adj))
        # print("forward - input")
        # print(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        # print("forward - step 1")
        # print(x)
        # x = torch.cat([att(x, adj, weighted_adj) for att in self.attentions], dim=1)
        # print("forward - step 2")
        # print(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        # print("forward - step 3")
        # print(x)
        # x = F.elu(self.out_att(x, adj, weighted_adj))
        # print("forward - step 4")
        # print(x)
        return x, F.softmax(x, dim=1) #was log_softmax before

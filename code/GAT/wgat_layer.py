import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, weighted_adj):
        #print('layer_forward')
        #print(self.W)
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        #print(Wh)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec) # the nxn matrix with calculated e_ij, which is the attention scores
        #print("attention", attention)
        attention = F.softmax(attention, dim=1)# the nxn matrix with calculated a_ij, which is the attention scores after softmax
        #print("softmax attention",attention)

        #print("the weighted_adj",weighted_adj)
        attention_weighted = torch.mul(attention, weighted_adj)
        #print("weighted attention",attention_weighted)

        ##attention_weighted = torch.where(attention_weighted > 0, attention_weighted, zero_vec) #make sure 0 is still 0 after the softmax function
        #print("weighted attention, zero vector", attention_weighted)
        ##attention_weighted = F.softmax(attention_weighted, dim=1)
        #print("weighted attention after softmax",attention_weighted)
        attention = F.dropout(attention_weighted, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        #print("h_prime",h_prime)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



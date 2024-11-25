import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import os
import re

EPS = 1e-15


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot



def load_widata(path="../data/", dataset="wi", hops=5):
    print('Loading {} dataset...'.format(dataset))

    features = np.loadtxt(path+'feature_matrix_f1.csv', delimiter=',')
    features = torch.FloatTensor(features[:,1:])

    intensity_m = np.loadtxt(path + 'Flow_matrix.csv', delimiter=',')
    intensity_neg = np.zeros([len(intensity_m),len(intensity_m)])
    intensity_neg[intensity_m == 0] = 1
    intensity_pos = np.zeros([len(intensity_m),len(intensity_m)])
    intensity_pos[intensity_m > 0] = 1

    intensity_m = np.log(intensity_m + EPS)
    intensity_m_norm = mx_normalize(intensity_m)
    intensity_m_norm = torch.FloatTensor(intensity_m_norm)
    strength = torch.sum(intensity_m_norm, axis = 0)
    strength = torch.FloatTensor(strength)

    hops_m = np.loadtxt(path + 'Spatial_distance_matrix.csv', delimiter=',')
    zero_entries = hops_m < hops
    hops_m = 1/(np.log(hops_m + EPS)+1)
    hops_m[zero_entries] = 0
    hops_m = torch.FloatTensor(hops_m)

    intensity_m = torch.FloatTensor(intensity_m)
    intensity_neg = torch.FloatTensor(intensity_neg)
    intensity_pos = torch.FloatTensor(intensity_pos)

    adj = np.loadtxt(path + 'Spatial_matrix.csv', delimiter=',')
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(adj)

    return adj, features, intensity_m, intensity_neg, intensity_pos, hops_m, intensity_m_norm, strength

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def mx_normalize(mx):
    "element wise normalization"
    mx_std = (mx - mx.min()) / (mx.max() - mx.min())
    return mx_std

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def purge(dir, filename, best_epoch, spill_num):
    del_list = ['Epoch_{}_'.format(i) + filename for i in range(0, best_epoch)]
    if spill_num > 0:
        tmp = ['Epoch_{}_'.format(j) + filename for j in range(best_epoch + 1, best_epoch + spill_num + 1)]
        del_list.extend(tmp)        
    for f in os.listdir(dir):
        if f in del_list:
            os.remove(os.path.join(dir,f))
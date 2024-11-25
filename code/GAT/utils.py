import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import os
import re
import matplotlib.pyplot as plt

from scipy.linalg import fractional_matrix_power

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

EPS = 1e-15


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot



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


def matrix_not_zero_perc(mx):
    mx = np.array(mx)
    non_zero = sum(sum(mx!=0))
    total_node = mx.shape[0]*mx.shape[1]
    print("the precentage of nodes that are not zero in the matrix:{} \n".format(non_zero*(1.0)/total_node))


def mod_B(adj):
    D = np.diag(adj.sum(axis=1))
    m = np.sum(adj > 0)
    B = adj - np.matmul(D, D.T)/(2*m)
    return B,m

def piecewise_norm(array, thre = 5000):
    array_norm = np.piecewise(array, [array<thre, array>=thre], [lambda array : array/thre, lambda array:1])
    return array_norm

def normalized_f1(array, k = 1):
    y = 2/(1 + np.exp(-k*(2* - 1)))
    return y

def load_widata_lap(path="../data/", dataset="wi", hops=5, flow_thre = 50, adj = "queen"):
    print('Loading {} dataset...'.format(dataset))

    features = np.loadtxt(path+'feature_matrix_f1.csv', delimiter=',')
    features = torch.FloatTensor(features[:,1:])

    intensity_m = np.loadtxt(path + 'Flow_matrix.csv', delimiter=',')
    intensity_neg = np.zeros([len(intensity_m),len(intensity_m)])
    intensity_neg[intensity_m == 0] = 1
    intensity_pos = np.zeros([len(intensity_m),len(intensity_m)])
    intensity_pos[intensity_m > 0] = 1

    intensity_m_thre = np.copy(intensity_m)
    intensity_m_thre[intensity_m <= flow_thre] = 0
    #intensity_m_thre = np.log(intensity_m_thre + EPS)

    intensity_m_norm_thre = mx_normalize(intensity_m_thre)
    
    intensity_m_norm_thre = torch.FloatTensor(intensity_m_norm_thre) #below threshold is 0, and then take the log, and then normalized.
    intensity_m_thre = torch.FloatTensor(intensity_m_thre)

    intensity_m = np.log(intensity_m + EPS)

    hops_m = np.loadtxt(path + 'Spatial_distance_matrix.csv', delimiter=',')
    zero_entries = hops_m < hops
    hops_m = 1/(np.log(hops_m + EPS)+1)
    hops_m[zero_entries] = 0
    hops_m = torch.FloatTensor(hops_m)

    intensity_m = torch.FloatTensor(intensity_m)
    intensity_neg = torch.FloatTensor(intensity_neg)
    intensity_pos = torch.FloatTensor(intensity_pos)

    if adj == "queen":
        adj = np.loadtxt(path + 'Spatial_matrix.csv', delimiter=',')
    elif adj == "rook":
        adj = np.loadtxt(path + 'Spatial_matrix_rook.csv', delimiter=',')  

    adj = adj + sp.eye(adj.shape[0])

    print("adj matrix")
    matrix_not_zero_perc(adj)
    adj = torch.FloatTensor(adj)

    return adj, features, intensity_m, intensity_neg, intensity_pos, hops_m, intensity_m_thre

#for thre in [200, 300, 500, 800, 1000]:


def load_widata_updatedGAT(pos_thre, path="../data/", dataset="wi", hops=5, flow_thre = 50, adj = "queen"):
    print('Loading {} dataset...'.format(dataset))

    features = np.loadtxt(path+'feature_matrix_f1.csv', delimiter=',')
    features = torch.FloatTensor(features[:,1:])

    print("The flow threshold is {}".format(flow_thre))

    intensity_m = np.loadtxt(path + 'Flow_matrix.csv', delimiter=',')
    intensity_neg = np.zeros([len(intensity_m),len(intensity_m)])
    intensity_neg[intensity_m == 0] = 1
    intensity_pos = np.zeros([len(intensity_m),len(intensity_m)])
    #intensity_pos[intensity_m > 0] = 1
    intensity_pos[intensity_m > pos_thre] = 1 ##would there be some values missing because neg is 0 and pos is another threshold?

    intensity_pos_thre = np.zeros([len(intensity_m),len(intensity_m)])
    print("spatial pos threshold:", pos_thre)
    intensity_pos_thre[intensity_m > pos_thre] = 1
    print("spatial pos matrix")
    matrix_not_zero_perc(intensity_pos_thre)

    intensity_m_thre = np.copy(intensity_m)
    intensity_m_thre[intensity_m <= flow_thre] = 0
    print("The flow intensity matrix with flow threshold")
    matrix_not_zero_perc(intensity_m_thre)

    intensity_m_norm = piecewise_norm(intensity_m_thre)
   # print("The intensity after normalization.")
    intensity_m_norm = torch.FloatTensor(intensity_m_norm)

#     intensity_m_thre_log = np.log(intensity_m_thre + EPS)
#     #print(intensity_m_thre_log)
#    # intensity_m_thre_log[intensity_m_thre_log < 0] = 0 #make all negative values into 0
#     intensity_m_thre_log = torch.FloatTensor(intensity_m_thre_log)
#     intensity_m_thre_log_norm = mx_normalize(intensity_m_thre_log)
#     #print(intensity_m_thre_log_norm)
#     intensity_m_thre_log_norm = torch.FloatTensor(intensity_m_thre_log_norm) #below threshold is 0, and then take the log, and then normalized.

    intensity_m = np.log(intensity_m + EPS)

    hops_m = np.loadtxt(path + 'Spatial_distance_matrix.csv', delimiter=',')
    zero_entries = hops_m < hops
    hops_m = 1/(np.log(hops_m + EPS)+1)
    hops_m[zero_entries] = 0
    print("The spatial hops matrix")
    matrix_not_zero_perc(hops_m)
    hops_m = torch.FloatTensor(hops_m)

    intensity_m = torch.FloatTensor(intensity_m)
    intensity_neg = torch.FloatTensor(intensity_neg)
    intensity_pos = torch.FloatTensor(intensity_pos)
    intensity_pos_thre = torch.FloatTensor(intensity_pos_thre)

    if adj == "queen":
        adj = np.loadtxt(path + 'Spatial_matrix.csv', delimiter=',')
    elif adj == "rook":
        adj = np.loadtxt(path + 'Spatial_matrix_rook.csv', delimiter=',')  

    adj = adj + sp.eye(adj.shape[0])
    adj = torch.FloatTensor(adj)
    print("The geographic adjacency matrix with diagonal filled with 1")
    matrix_not_zero_perc(adj)
            

    return adj, features, intensity_m, intensity_neg, intensity_pos, intensity_pos_thre, hops_m, intensity_m_norm

#load_widata_updatedGAT(flow_thre=5, pos_thre = 0)




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
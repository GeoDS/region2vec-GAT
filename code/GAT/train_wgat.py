from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_widata_lap, purge, mod_B, load_widata_updatedGAT, matrix_not_zero_perc
from model_wgat import GAT as weighted_GAT
#from analytics import run_kmeans, plot_map
#import csv
#import os
import math
import random
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=50,
                    help='Early stopping control.')
parser.add_argument('--ltype', type=str, default='divreg',
                    help='divide or loglike.')
parser.add_argument('--lr', type=float, default=0.001,     
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--output', type=int, default=14,
                    help='Output dim.')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hops', type=int, default=5,
                    help='Contrain with hops')
parser.add_argument('--flow_thre', type=int, default=50,
                    help='The flow threshold for determining whether it is adjacent.')
parser.add_argument('--adj', type=str, default='queen',
                    help='The geographic adjacency relationship')
parser.add_argument('--head', type=int, default=4,
                    help='The number of heads for the attention network.')
parser.add_argument('--pos_thre', type=int, default=200,
                    help='pos thre.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Load data
#adj, features, labels, neg_mask, pos_mask, hops_m, intensity_m_norm, intensity_m_norm_thre, strength, adj_lap, flow_lap, intensity_binary = load_widata_lap(flow_thre = args.flow_thre, adj = args.adj)

adj, features, labels, neg_mask, pos_mask, intensity_pos_thre, hops_m, intensity_m_norm = load_widata_updatedGAT(flow_thre = args.flow_thre, adj = args.adj, pos_thre = args.pos_thre, hops = args.hops)
print(labels)
print(neg_mask)
print(pos_mask)



print(args.output)
# Model and optimizer
model = weighted_GAT(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=args.output,
            dropout=args.dropout,
            nheads=args.head,
            alpha=0.2)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


EPS = 1e-15


N_pos = sum(sum(pos_mask))
N_neg = sum(sum(neg_mask))

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    neg_mask = neg_mask.cuda()
    pos_mask = pos_mask.cuda()
    hops_m = hops_m.cuda()


# Train model
t_total = time.time()
loss_list = []

input_graph = adj + intensity_pos_thre
matrix_not_zero_perc(input_graph)


for epoch in range(args.epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, output_sfx = model(features, input_graph, intensity_m_norm) #output_sfx

    loss_mo = 0
    
    pdist = torch.norm(output[:, None]-output, dim=2, p=2)
    inner_pro = torch.mm(output,output.T)
    loss_hops = 0

# 直接用Flow strength, 最小化Loss = W_pos*Dist - W_zero*Dist, 前半部分W_pos(强度)越大,得到的表征距离越小;后半部分是计算强度为0的节点之间的距离,该部分越大越好
    if args.ltype == 'div2':
        if args.hops > 1:
            loss_hops = pdist.mul(hops_m)/182224
        loss_train = torch.sum(pdist.mul(labels).mul(pos_mask)) - torch.sum(pdist.mul(neg_mask) + loss_hops)
        loss_train = loss_train/(2*len(output)) + loss_mo

    elif args.ltype == 'div':
        if args.hops > 1:
            loss_hops = torch.sum(pdist.mul(hops_m)) + EPS
        loss_train = torch.sum(pdist.mul(labels).mul(pos_mask)) /( (torch.sum(pdist.mul(neg_mask)) + EPS) + loss_hops)
        loss_train = loss_train + loss_mo

    elif args.ltype == 'divreg':
        if args.hops > 1:
            loss_hops = torch.sum(pdist.mul(hops_m)) + EPS
        #print('loss_hops',loss_hops)
        loss_train = torch.sum(pdist.mul(labels).mul(pos_mask))*N_neg /(N_pos*((torch.sum(pdist.mul(neg_mask)) + EPS) + loss_hops))
        loss_train = loss_train + loss_mo
        # print("pos mask dist, ",torch.sum(pdist.mul(labels).mul(pos_mask)))
        # print("N neg, ", N_neg)
        # print("N pos", N_pos)
        # print("neg mask dist", torch.sum(pdist.mul(neg_mask)))
        # print("loss hops, ", loss_hops)
        #print('dinominator:', torch.sum(pdist.mul(neg_mask)) + EPS)
        #print('loss_mo:', loss_mo)

    elif args.ltype == 'divreg_largerhops':
        if args.hops > 1:
            loss_hops = torch.sum(pdist.mul(hops_m)) + EPS
        loss_train = torch.sum(pdist.mul(labels).mul(pos_mask))*N_neg /(N_pos*( (torch.sum(pdist.mul(neg_mask)) + EPS) + 3*loss_hops))
        loss_train = loss_train + loss_mo

    elif args.ltype == 'divreg_spatialpos_hops':
        if args.hops > 1:
            loss_hops = torch.sum(pdist.mul(hops_m)) + EPS
        loss_train = torch.sum(pdist.mul(labels).mul(pos_mask))*N_neg /(N_pos*(loss_hops))
        loss_train = loss_train + loss_mo

    elif args.ltype == 'divreg_spatialall':
        loss_train = torch.sum(pdist.mul(labels).mul(pos_mask))*N_neg /(N_pos*((torch.sum(pdist.mul(neg_mask)) + EPS)))
        loss_train = loss_train + loss_mo
        #print('loss_train:', loss_train)
        #print('loss_mo:', loss_mo)

    elif args.ltype == 'divreg_hops':
        if args.hops > 1:
            loss_hops = torch.sum(pdist.mul(hops_m)) + EPS
        loss_train = loss_hops
 
    elif args.ltype == 'divreg_onlypos':
        if args.hops > 1:
            loss_hops = pdist.mul(hops_m)
        loss_train = torch.sum(pdist.mul(labels).mul(pos_mask))/N_pos #*N_neg  #- torch.sum(loss_hops)/N_neg
        loss_train = loss_train + loss_mo

    elif args.ltype == 'div2reg':
        if args.hops > 1:
            loss_hops = pdist.mul(hops_m)
        loss_train = torch.sum(pdist.mul(labels).mul(pos_mask))/N_pos - torch.sum(pdist.mul(neg_mask) + loss_hops)/N_neg
        loss_train = loss_train + loss_mo


    elif args.ltype == 'tri':
        criterion = torch.nn.MarginRankingLoss(margin=0.5)
        dist_pos = pdist.mul(pos_mask)
        dist_neg = pdist.mul(neg_mask)
        pos = dist_pos[dist_pos > 0]
        neg = dist_neg[dist_neg > 0]
        len_p = len(pos)
        len_n = len(neg)
        pos_n = torch.cat((pos,pos[0:len_n-len_p]))
        target = torch.ones_like(neg)
        loss_train = criterion(pos_n, neg, target)

    elif args.ltype == 'log':
        pos_out = inner_pro.mul(1/labels).mul(pos_mask)
        neg_out = inner_pro.mul(neg_mask)
        pos_loss = -torch.log(torch.sigmoid(pos_out) + EPS).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + EPS).mean()
        # neg_loss = torch.log(torch.sigmoid(neg_out) + EPS).mean()
        if args.hops > 1:
            loss_hops = -torch.log(1 - torch.sigmoid(inner_pro.mul(hops_m)) + EPS).mean()
        loss_train = pos_loss + neg_loss + loss_hops + loss_mo

    elif args.ltype == 'log2':
        pos_out1 = inner_pro.mul(1/labels).mul(pos_mask)
        neg_out1 = inner_pro.mul(neg_mask)
        pos_loss1 = -torch.log(torch.exp(pos_out1) + EPS).sum()
        neg_loss1 = torch.log(torch.exp(neg_out1) + EPS).sum()
        if args.hops > 1:
            loss_hops = torch.log(torch.exp(inner_pro.mul(hops_m)) + EPS).mean()
        loss_train = pos_loss1 + neg_loss1 + loss_hops + loss_mo


    elif args.ltype == 'log3':
        pos_out1 = inner_pro.mul(1/labels).mul(pos_mask)
        neg_out1 = inner_pro.mul(neg_mask)
        if args.hops > 1:
            loss_hops = torch.exp(inner_pro.mul(hops_m))

        loss_train = - torch.sum(torch.exp(pos_out1)) / torch.sum(torch.exp(neg_out1) + loss_hops) + loss_mo



    loss_train.backward()
    optimizer.step()

    loss_list.append(loss_train.item())

    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.5f}'.format(loss_train.item()),
          #'loss_mod: {:.5f}'.format(loss_mo.item()),
          'time: {:.4f}s'.format(time.time() - t))

    if epoch >= 200:
        # print(output)

        save_name = 'test1' + 'dropout_{}_hop_{}_losstype_{}_output_{}_seed_{}_flowthre_{}_adj_{}_head_{}_pos_{}.csv'.format(args.dropout, args.hops, args.ltype, args.output, args.seed, args.flow_thre, args.adj, args.head, args.pos_thre)
        
        np.savetxt('result/' + 'Epoch_{}_'.format(epoch) + save_name, output.detach().numpy())

        if epoch > 200 + args.patience and loss_train > np.average(loss_list[-args.patience:]):
            best_epoch = loss_list.index(min(loss_list))
            print('Lose patience, stop training...')
            print('Best epoch: {}'.format(best_epoch))
            purge('result/', save_name, best_epoch, epoch-best_epoch)
            break

        if epoch == args.epochs -1:
            print('Last epoch, saving...')
            best_epoch = epoch
            purge('result/', save_name, best_epoch, 0)


print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
# result_csv = 'result_v2.csv'

# best_fname = 'Epoch_{}_'.format(best_epoch) + save_name #.replace('.csv','')
# n_clu = 13
# kmeans_labels, inertia, total_ratio, sil_score, global_ineq, median_ineq = run_kmeans(best_fname, n_clusters = n_clu)
# plot_map(best_fname, kmeans_labels)
# # Testing

# if not os.path.exists(os.path.join('../result_v2/', result_csv)):
#     with open(os.path.join('../result_v2/', result_csv), 'w') as f:
#         csv_write = csv.writer(f)
#         csv_head = ['epoch', 'losstype', 'hops', 'mod', 'inertia', 'total_ratio', 'sil_score', 'global_ineq', 'output', 'hidden', 'lr', 'dropout', 'patience', 'median_ineq','n_clu']
#         csv_write.writerow(csv_head)
#         f.close()


# with open(os.path.join('../result_v2/', result_csv), 'a+') as f:
#     csv_write = csv.writer(f)
#     csv_data = [best_epoch, args.ltype, args.hops, args.mod, inertia, total_ratio, sil_score, global_ineq, args.output, args.hidden, args.lr, args.dropout, args.patience, median_ineq, n_clu]
#     csv_write.writerow(csv_data)


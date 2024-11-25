from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_widata, purge
from models import GCN
import csv
import os
import math
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
                    help='divide or loglike with regularization.')
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
parser.add_argument('--hops', type=float, default=5,
                    help='Contrain with hops')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Load data
adj, features, labels, neg_mask, pos_mask, hops_m, intensity_m_norm, strength = load_widata()
print(args.ltype)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nout=args.output,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


EPS = 1e-15

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    neg_mask = neg_mask.cuda()
    pos_mask = pos_mask.cuda()
    hops_m = hops_m.cuda()

N_pos = sum(sum(pos_mask))
N_neg = sum(sum(neg_mask))

# Train model
t_total = time.time()
loss_list = []
for epoch in range(args.epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, output_sfx = model(features, adj) 

    pdist = torch.norm(output[:, None]-output, dim=2, p=2)
    inner_pro = torch.mm(output,output.T)
    loss_hops = 0

# the div loss uses the flow strength directly. It tries to minimize the loss of W_pos*Dist - W_zero*Dist. A larger strength (W_pos) leads to smaller\
#  the distances between embeddings; the W_zero refers nodes with no flow, so their embeddings differences should be as large as possible. 
    if args.ltype == 'div':
        if args.hops > 1:
            loss_hops = torch.sum(pdist.mul(hops_m)) + EPS
        loss_train = torch.sum(pdist.mul(labels).mul(pos_mask)) /( (torch.sum(pdist.mul(neg_mask)) + EPS) + loss_hops)

    elif args.ltype == 'divreg':
        if args.hops > 1:
            loss_hops = torch.sum(pdist.mul(hops_m)) + EPS
        loss_train = torch.sum(pdist.mul(labels).mul(pos_mask))*N_neg /(N_pos*( (torch.sum(pdist.mul(neg_mask)) + EPS) + loss_hops))



    loss_train.backward()
    optimizer.step()

    loss_list.append(loss_train.item())

    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.5f}'.format(loss_train.item()),
          'time: {:.4f}s'.format(time.time() - t))


    result_path = '../result/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    if epoch >= 200:
        save_name = 'lr_{}_dropout_{}_hidden_{}_output_{}_patience_{}_hos_{}_losstype_{}_seed_{}.csv'.format(args.lr, args.dropout, args.hidden, args.output, args.patience, args.hops, args.ltype, args.seed)
       
        np.savetxt(result_path + 'Epoch_{}_'.format(epoch) + save_name, output.detach().numpy())

        if epoch > 200 + args.patience and loss_train > np.average(loss_list[-args.patience:]):
            best_epoch = loss_list.index(min(loss_list))
            print('Lose patience, stop training...')
            print('Best epoch: {}'.format(best_epoch))
            purge(result_path, save_name, best_epoch, epoch-best_epoch)
            break

        if epoch == args.epochs -1:
            print('Last epoch, saving...')
            best_epoch = epoch
            purge(result_path, save_name, best_epoch, 0)


print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


result_csv = 'result.csv'
if not os.path.exists(os.path.join(result_path, result_csv)):
    with open(os.path.join(result_path, result_csv), 'w') as f:
        csv_write = csv.writer(f)
        csv_head = ['epoch', 'losstype', 'hops', 'inertia', 'total_ratio', 'global_ineq', 'output', 'hidden', 'lr', 'dropout', 'patience', 'median_ineq','n_clu']
        csv_write.writerow(csv_head)
        f.close()



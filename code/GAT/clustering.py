import os
from analytics import run_aggclustering, plot_map, kmeans, louvain
import csv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--n_clusters', type=int, default=14,
                    help='Number of clusters.')
parser.add_argument('--affinity', type=str, default='euclidean',
                    help='affinity metric')
parser.add_argument('--filename', type=str, default='Epoch_378_dropout_0.1_hop_5.0_losstype_divreg_mod_False.csv',
                    help='file name')
parser.add_argument('--con', type=str, default='rook',
                    help='the connectivity relationship')
parser.add_argument('--algo', type=str, default='agg',
                    help='algorithm: agg or kmeans')   
                  
args = parser.parse_args()


if '../' in args.filename:
    args.filename = args.filename.split('/')[-1]

linkage = 'ward'
path = '../../GAT_analysis_10262023/files/' #'../result_gat/weighted_gat/' # ../../GAT_analysis_10262023/files/
if args.algo == 'agg':
    labels, total_ratio, ratio_norm, median_ineq, median_cossim, median_dist, homo_score, ineq_ls = run_aggclustering(path, args.filename, args.affinity, args.n_clusters, linkage, args.con)
    #plot_map(args.filename, labels, path, args.n_clusters, savefig = True, con = args.con)
elif args.algo == 'kmeans':
    labels, total_ratio, median_ineq, median_cossim, median_dist, homo_score, ineq_ls = kmeans(path, args.n_clusters)
    #plot_map(args.filename, labels, path, args.n_clusters, savefig = True, con = 'none')

elif args.algo == "louvain":
    labels, total_ratio, median_ineq, median_cossim, median_dist, homo_score, ineq_ls = louvain(path)
    #plot_map(args.filename, labels, path, args.n_clusters, savefig = True, con = 'none')



#seed = args.filename.split('_')[17]
#adj = args.filename.split('_')[-1].split(".")[0]

csv_data = [args.filename, args.n_clusters, linkage, args.affinity, total_ratio, median_ineq, median_cossim, median_dist, homo_score, args.con]
ineq_data = [args.filename, args.n_clusters] + ineq_ls
result_csv = 'cluster_result.csv'
ineq_csv = 'inequality.csv'

if not os.path.exists(os.path.join(path, result_csv)):
    with open(os.path.join(path, result_csv), 'w') as f:
        csv_write = csv.writer(f)
        csv_head = ['file_name', 'n_clusters', 'linkage', 'distance', 'total_ratio', 'median_ineq', 'median_cossim','median_dist', "homo_score", "connectivity"]
        csv_write.writerow(csv_head)
        f.close()

with open(os.path.join(path, result_csv), mode='a', newline='') as f1:
    csv_write = csv.writer(f1)
    csv_write.writerow(csv_data)


if not os.path.exists(os.path.join(path, ineq_csv)):
    with open(os.path.join(path, ineq_csv), 'w') as f2:
        csv_write = csv.writer(f2)
        ineq_head = ['file_name', 'n_clusters','white_perc', 'black_perc', 'asian_perc', 'hispanic_perc', 'households_total', 'mean_inc', 'pvt_50_perc', 'pvt_abv_300_perc', 'pvt_50_300_perc']
        csv_write.writerow(ineq_head)
        f2.close()

with open(os.path.join(path, ineq_csv), mode='a', newline='') as f3:
    csv_write = csv.writer(f3)
    csv_write.writerow(ineq_data)    
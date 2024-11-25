#!/bin/bash 

source D:/Applications/anaconda3/etc/profile.d/conda.sh
conda activate D:/Applications/envs/gcn

nclu=14
find ../result -maxdepth 1 -type f -name Epoch*.csv  -exec python clustering.py --filename {} --n_clusters $nclu \;

#nclu=20
# for nclu in 20 30 50 80
# do
#python clustering.py --n_clusters $nclu --filename algo_line_output_14_epochs_5_new.csv;
#python clustering.py --filename Epoch_378_dropout_0.1_hop_5_losstype_divreg_mod_True.csv --n_clusters $nclu;
#     python clustering.py --filename algo_node2vec_output_13_epochs_5.csv --n_clusters $nclu --nbr none;
#     python clustering.py --filename algo_deepwalk_output_13_epochs_5.csv --n_clusters $nclu --nbr none;
#python clustering.py  --n_clusters $nclu --algo kmeans;
# done#
#!/bin/bash 

source D:/Applications/anaconda3/etc/profile.d/conda.sh
conda activate D:/Applications/Anaconda3/envs/igraph_env

#nclu=14
#for adj in rook #rook1000 queen
for nclu in 14 #8 10 12 16 18 20 25 #30 35 40
do
    find ../../GAT_analysis_10262023/files/ -maxdepth 1 -type f -name Epoch_263*.csv  -exec python clustering.py --filename {} --n_clusters $nclu --con rook \;
    find ../../GAT_analysis_10262023/files/ -maxdepth 1 -type f -name Epoch_378*.csv  -exec python clustering.py --filename {} --n_clusters $nclu --con rook \;
    find ../../GAT_analysis_10262023/files/ -maxdepth 1 -type f -name algo_node*.csv  -exec python clustering.py --filename {} --n_clusters $nclu --con none \;
    #python clustering.py  --n_clusters $nclu --algo kmeans --filename kmeans --con none;
    python clustering.py  --n_clusters $nclu --algo louvain --filename louvain --con none;
done


#../../GAT_analysis_10262023/files/
#../result_gat/weighted_gat/
#nclu=20
# for nclu in 20 30 50 80
# do
#python clustering.py --n_clusters $nclu --filename ../../GAT_analysis_10262023/files/Epoch_243_test33_dropout_0.1_hop_5_losstype_divreg_output_14_seed_43_flowthre_100_adj_queen_head_4_pos_200.csv --con rook;
#python clustering.py --filename Epoch_378_dropout_0.1_hop_5_losstype_divreg_mod_True.csv --n_clusters $nclu;
#     python clustering.py --filename algo_node2vec_output_13_epochs_5.csv --n_clusters $nclu --nbr none;
#     python clustering.py --filename algo_deepwalk_output_13_epochs_5.csv --n_clusters $nclu --nbr none;

# done#


#42 44 45 46 47 48 49 
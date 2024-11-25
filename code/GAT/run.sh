#!/bin/bash 

source D:/Applications/anaconda3/etc/profile.d/conda.sh
conda activate D:/Applications/Anaconda3/envs/igraph_env

#ltype div/log #output 8 13
#for seed in 46 #pos_thre in 10 20 25 #42 43 47 48 49 #49 50 #46 47 48 #
for seed in 46 # 42 44 45 46 47 48 49
do
     python train_wgat.py --ltype divreg --hops 5 --output 14 --hidden 16 --seed $seed --flow_thre 100 --adj queen --epochs 500 --pos_thre 200 #--head 8
     #python train_wgat.py --ltype divreg --hops 5 --output 14 --hidden 16 --seed $seed --flow_thre 50 --adj queen --epochs 500 --pos_thre 200 
      #--pos_thre $pos_thre # --adj rook  # --mod #--adj queen
     #python train_wgat.py --ltype divreg --hops 5 --output 14 --hidden 16 --seed $seed --flow_thre 20 --adj queen --epochs 500 --pos_thre 200
     #python train_gat_0323.py --ltype divreg --hops 5 --output 14 --hidden 16 --seed $seed --flow_thre 5 #--adj queen --epochs 500 --pos_thre 0 
     #python train_gat.py --ltype divreg --hops 5 --output 14 --hidden 16 --seed $seed --flow_thre 5 --adj queen --epochs 500 --pos_thre 0 --head 1  # --mod #--adj queen
done


# for coef in 0.999
# do
#     #python train.py --ltype divreg --hops 5 --output 14 --hidden 16 --seed $seed --adj rook
#     python train_gat.py --ltype divreg --hops 5 --output 14 --hidden 16 --coef $coef --flow_thre 5 --seed 43 --epochs 10 #--adj queen
# done

# for iteration in {0..4..2}
# do
#     python train_gat.py --ltype divreg --hops 5 --output 14 --hidden 16 --seed 43 --flow_thre 5 --epochs 20
# done


# for threshold in 1000 #500 800  #42 #43 44 #45 46 #47 48 49 50
# do
#     #python train.py --ltype divreg --hops 5 --output 14 --hidden 16 --seed $seed --adj rook
#     python train_gat.py --ltype divreg --hops 5 --output 14 --hidden 16 --flow_thre $threshold --seed 43 --coef 0.5 #--adj queen
# done
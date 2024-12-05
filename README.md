# region2vec-GAT

**GeoAI-Enhanced Community Detection on Spatial Networks with Graph Deep Learning**
 
region2vec: Community Detection on Spatial Networks Using Graph Embedding with Node Attributes and Spatial Interactions

![Region2vec](https://github.com/GeoDS/region2vec-GAT/blob/master/Region2Vec_Workflow.jpg)
![Region2vec](https://github.com/GeoDS/region2vec-GAT/blob/master/Region2Vec_results.jpg)

**Abstract:** 
Spatial networks are useful for modeling geographic phenomena where spatial interaction plays an important role. To analyze the spatial networks and their internal structures, graph-based methods such as community detection have been widely used. Community detection aims to extract strongly connected components from the network and reveal the hidden relationships between nodes, but they usually do not involve the attribute information. To consider edge-based interactions and node attributes together, this study proposed a family of GeoAI-enhanced unsupervised community detection methods called *region2vec* based on Graph Attention Networks (GAT) and Graph Convolutional Networks (GCN). The *region2vec* methods generate node neural embeddings based on attribute similarity, geographic adjacency and spatial interactions, and then extract network communities based on node embeddings using agglomerative clustering. The proposed GeoAI-based methods are compared with multiple baselines and perform the best when one wants to maximize node attribute similarity and spatial interaction intensity simultaneously within the spatial network communities. It is further applied in the shortage area delineation problem in public health and demonstrates its promise in regionalization problems. 


## Paper

If you find our code useful for your research, you may cite our paper:

Liang, Y., Zhu, J., Ye, W., and Gao, S.* (2025).  [GeoAI-Enhanced Community Detection on Spatial Networks with Graph Deep Learning](https://doi.org/10.1016/j.compenvurbsys.2024.102228). Computers, Environment and Urban Systems. 117, 102228, 1-20. DOI: https://doi.org/10.1016/j.compenvurbsys.2024.102228


```
@article{liang2025GeoAI,
  title={GeoAI-Enhanced Community Detection on Spatial Networks with Graph Deep Learning},
  author={Liang, Yunlei and Zhu, Jiawei and Ye, Wen and Gao, Song },
  journal={Computers, Environment and Urban Systems},
  volume={117},
  number={1},
  pages={102228},
  year={2025},
  doi = {https://doi.org/10.1016/j.compenvurbsys.2024.102228},
  publisher={Elsevier}
}
```

## Requirements

Region2 uses the following packages with Python 3.7

numpy==1.19.5

pandas==0.24.1

scikit_learn==1.1.2

scipy==1.3.1

torch==1.4.0 (suggest torch>=2.2.0 for security alert)



## Usage

1. run train.py to generate the embeddings.
```
python train.py
```
2. run clustering.py to generate the clustering result. 

```
python clustering.py --filename your_filename
```
Here the 'your_filename' should be replaced with the generated file from step 1.

3. Alternatively, to generate the clustering for all the files, please use bash, and run bash run_clustering.py.

```
bash run_clustering.sh 
```
Notes: the final results (e.g., metric values) may vary depends on different platforms and package versions.
The current result is obtained using Ubuntu with all pacakge versions in requirements.txt. 

## Data
The data files used in our method are listed below with detailed descriptions.

Flow_matrix.csv: The visitor flow matrix between Census Tracts in Wisconsin (The spatial flow interaction matrix).

Spatial_matrix.csv: The adjacency matrix generated based on the geographic adjacency relationship.

Spatial_matrix_rook.csv: The adjacency matrix generated based on the geographic adjacency relationship with the rook-type contiguity relationship.

Spatial_distance_matrix.csv: the hop distance calculated based on the spatial adjacency matrix.

flow_reID.csv: the visitor flows with updated IDs of Census Tracts.

feature_matrix_f1.csv: the features of nodes (Census Tracts).

feature_matrix_lwinc.csv: the low income population feature of nodes used for generating the homogeneous scores.



## Acknowledgement
We acknowledge the funding support from the County Health Rankings and Roadmaps program of the University of Wisconsin Population Health Institute, Wisconsin Department of Health Services, and the National Science Foundation funded AI institute [Grant No. 2112606] for [Intelligent Cyberinfrastructure with Computational Learning in the Environment (ICICLE)](https://icicle.ai/). Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the funders.


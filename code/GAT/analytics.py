import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from numpy import genfromtxt
import geopandas as gpd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import networkx as nx
import networkx.algorithms.community as nx_comm


EPS = 1e-15
def run_aggclustering(path, file_name, affinity, n_clusters, linkage = 'ward', adj = 'rook'):
    #print(n_clusters)
    X = np.loadtxt(path+file_name, delimiter=' ')
    if 'csv' in file_name:
        file_name = file_name[:-4]

    if adj == 'none':
        model = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters, metric=affinity)
    else:
        if adj == 'rook':
            adj = np.loadtxt('../data/Spatial_matrix_rook.csv', delimiter=',')
        elif adj == 'rook1000':
            adj = np.loadtxt('../data/Spatial_matrix_rook1000.csv', delimiter=',')
        elif adj == 'queen':
            adj = np.loadtxt('../data/Spatial_matrix.csv', delimiter=',')
        model = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters, connectivity = adj, metric=affinity)

    model.fit(X)
    labels = model.labels_

    estimate_HPSA(labels)  
    total_ratio, ratio_norm = intra_inter_idx(labels, n_clusters)
    median_ineq = community_inequality(labels, file_name, path, n_clusters)
    ineq_ls = community_inequality_seperate(labels, file_name,path, n_clusters)

    median_sim, median_dist = similarity(labels, file_name, path, n_clusters)
    homo_score = lwinc_purity(labels) 

    
    #plot_embeddings(X, labels, path, file_name, savefig = True)
    
    # print('\nThe normalized cosine similarity is {:.3f}'.format(median_sim*ratio_norm))
    # print('The normalized homogeneous score is {:.3f}'.format(homo_score*ratio_norm))
    # print('The normalized euclidean distance score is {:.3f}\n'.format(median_dist/ratio_norm))


    return labels, total_ratio, ratio_norm, median_ineq, median_sim, median_dist, homo_score, ineq_ls

def estimate_HPSA(labels): 
    fte = pd.read_csv('../data/wi_ct_demo_fte_20230320.csv') # this is a private data we cannot share as it is from the health department; one can create your own custimzed data labels 
    spa_ID = genfromtxt('../data/spa.csv', delimiter=',')
    ID_Union = sorted([str(int(ct)) for ct in spa_ID])

    clu_col = 'clusterid'
    result_df = pd.DataFrame(data = {'ct':ID_Union, clu_col: labels})
    result_df['ct'] = result_df['ct'].astype(float)

    cols = ['ct','pop_lwinc','pop','pc_lwinc','pc_fte']
    fte = fte[cols]
    
    cluster_df = result_df.merge(fte, on = ['ct']).groupby(clu_col).sum().reset_index()
    cluster_df['ct'] = cluster_df['ct'].astype(str)
    cluster_df['popfte_ratio'] = cluster_df['pop_lwinc']/cluster_df['pc_lwinc']

    cluster_df['pop_ratio'] = cluster_df['pop_lwinc']/cluster_df['pop']

    # cluster_df['ratio'] = cluster_df['pop']/cluster_df['pc_fte']
    
    # cluster_df['HPSA_geog'] = False
    # for i, row in cluster_df.iterrows():
    #     if row['ratio'] == math.inf and row['pop'] > 500:
    #         cluster_df.loc[i, 'HPSA_geog'] = True
    #     if row['ratio'] != math.inf and row['ratio'] > 3500:
    #         cluster_df.loc[i, 'HPSA_geog'] = True
    # print(f"There are in total {len(cluster_df)} clusters.")
    # print(f"The estimated GEOG HPSA is {len(cluster_df[cluster_df['HPSA_geog'] == True])}")

    # pop_sum = sum(cluster_df[cluster_df['HPSA_geog'] == True]['pop'])
    # fte_sum = sum(cluster_df[cluster_df['HPSA_geog'] == True]['pc_fte'].dropna())
    # print("pop sum:", pop_sum)
    # print("fte sum:", fte_sum)
    # if fte_sum != 0:
    #     print("GEOG POP to FTE: {}".format(pop_sum/fte_sum))


    cluster_df['HPSA'] = False
    for i, row in cluster_df.iterrows():
        if row['pop_ratio'] >= 0.28:
            if row['popfte_ratio'] == math.inf and row['pop'] > 500:
                cluster_df.loc[i, 'HPSA'] = True
            if row['popfte_ratio'] != math.inf and row['popfte_ratio'] > 3000:
                cluster_df.loc[i, 'HPSA'] = True


    print(f"There are in total {len(cluster_df)} clusters.")
    print(f"The estimated LowIncome HPSA is {len(cluster_df[cluster_df['HPSA'] == True])}")
    
    pop_sum = sum(cluster_df[cluster_df['HPSA'] == True]['pop_lwinc'])
    fte_sum = sum(cluster_df[cluster_df['HPSA'] == True]['pc_lwinc'].dropna())
    print("pop sum:", pop_sum)
    print("fte sum:", fte_sum)
    if fte_sum != 0:
        print("POP to FTE: {}".format(pop_sum/fte_sum))

    return 

#generate the homogeneous score
def lwinc_purity(labels, n_thres = 5, lwinc_file = "../data/feature_matrix_lwinc.csv"):
    X_lwinc = np.loadtxt(lwinc_file, delimiter=',') 
    X_lwinc = X_lwinc[:,1:]
    threshold = np.arange(0, 1+1/n_thres, 1/n_thres)
    lwinc_classes = [np.quantile(X_lwinc, q) for q in threshold] #the classification of lwinc perc
    lwinc_classes[-1] = 1 + EPS #make the upper limit larger than any exsiting values

    X_classes = np.array([next(i-1 for i,t in enumerate(lwinc_classes) if t > v) for v in X_lwinc])

    homo_score = metrics.homogeneity_score(X_classes, labels)
    print("The homogeneous score is {:.3f}".format(homo_score))

    return homo_score
    

def intra_inter_idx(labels, k):
    CIDs = labels
    
    #generate ID to Community ID mapping
    UID = range(0,len(CIDs))
    ID_dict = dict(zip(UID, CIDs))
    
    flow = pd.read_csv('../data/flow_reID.csv')
    if 'Unnamed: 0' in flow.columns:
        flow = flow.drop(columns = 'Unnamed: 0')
        
    flow['From'] = flow['From'].map(ID_dict)
    flow['To'] = flow['To'].map(ID_dict)  
    
    #groupby into communities
    flow_com = flow.groupby(['From','To']).sum(['visitor_flows','pop_flows']).reset_index()
    
    ComIDs = list(flow_com.From.unique())
    intra_flows = list(flow_com[flow_com['From'] == flow_com['To']]['visitor_flows'].values)
    inter_flows = list(flow_com[flow_com['From'] != flow_com['To']].groupby(['From']).sum(['visitor_flows']).reset_index()['visitor_flows'])
    d = {'CID':ComIDs, 'intra': intra_flows, 'inter': inter_flows}
    df = pd.DataFrame(d)
    df['intra_inter'] = df['intra']/df['inter']    

    total_ratio = sum(df['intra'])/sum(df['inter']) 
    print("The total intra/inter ratio is {:.3f}".format(total_ratio))
    #print("test: {}".format(sum(flow_com[flow_com['From'] == flow_com['To']]['visitor_flows'].values)/sum(flow_com[flow_com['From'] != flow_com['To']]['visitor_flows'].values)))
    
    ratio_norm = sum(df['intra'])/(sum(df['inter'])+sum(df['intra']))
    print("The normalized intra/total ratio is {:.3f}".format(ratio_norm))

    return total_ratio, ratio_norm


def similarity(labels, file_name, path, n_clusters, savefig = True, feature_path = '../data/feature_matrix_f1.csv'):
    features = np.loadtxt(feature_path, delimiter=',') 
    X = features[:,1:]

    #calculate cos similarity for all features
    cossim_mx = cosine_similarity(X)

    sim_dict = {}
    for c in range(n_clusters):
        ct_com = np.where(labels == c)[0]
        cossim_com = cossim_mx[ct_com[:,None], ct_com[None,:]]  #slice the matrix so all the included values is for this community
        cossim = cossim_com[np.triu_indices(len(ct_com), k = 0)]
        sim_dict[c] = np.mean(cossim)

    median_sim = np.median(list(sim_dict.values()))

    #calculate the euclidean distance for all features
    eucdist_mx = euclidean_distances(X)

    dist_dict = {}
    for c in range(n_clusters):
        ct_com = np.where(labels == c)[0]
        eucdist_com = eucdist_mx[ct_com[:,None], ct_com[None,:]]  #slice the matrix so all the included values is for this community
        eucdist = eucdist_com[np.triu_indices(len(ct_com), k = 0)]
        dist_dict[c] = np.mean(eucdist)

    median_dist = np.median(list(dist_dict.values()))

    print("The median cosine similarity is {:.3f}".format(median_sim))
    print("The median euclidean distance similarity is {:.3f}".format(median_dist))

    return median_sim, median_dist

def cal_inequality(values):
    mean = np.mean(values)
    std = np.std(values)
    if mean == 0 or std == 0:
        ineq = -1
    else:
        ineq = std/math.sqrt(mean*(1-mean))
    return ineq

def community_inequality(labels, file_name, path, k = 13):
    features = np.loadtxt('../data/feature_matrix_f1.csv', delimiter=',') #use updated features   
    features = features[:,1:]
    pdist = np.linalg.norm(features[:, None]-features, ord = 2, axis=2)

    ineq_dict = {}
    for c in range(k):
        ct_com = np.where(labels == c)[0]
        if len(ct_com) < 2:
            continue
        else:
            pdist_com = pdist[ct_com[:,None], ct_com[None,:]]  #slice the pdist so all the included values is for this community
            dist = pdist_com[np.triu_indices(len(ct_com), k = 1)]
            
            #calculate the inequality
            ineq = cal_inequality(dist)
            ineq_dict[c] = ineq

    median_ineq = np.median(list(ineq_dict.values()))
    print("The median inequality is {:.3f}".format(median_ineq))

    return median_ineq


def community_inequality_seperate(labels, file_name, path, k = 13):
    features = np.loadtxt('../data/feature_matrix_f1.csv', delimiter=',') #use updated features   
    features = features[:,1:]

    # df = pd.DataFrame(features, columns = ['white_perc', 'black_perc', 'asian_perc', 'hispanic_perc',
    #    'households_total', 'mean_inc', 'pvt_50_perc', 'pvt_abv_300_perc',
    #    'pvt_50_300_perc'])

    # #print(df.head())
    # #print(df.shape)

    ineq_list = []
    for i in range(features.shape[1]):
        feature_ineq_dict = {}
        for c in range(k):
            ct_com = np.where(labels == c)[0]
            #print(c)
            #print(ct_com)
            if len(ct_com) < 2:
                continue
            else:
                #slice the features so all the included values is for this community
                features_list = features[:,i][ct_com]            
                #print(features_list)
                #calculate the inequality
                ineq = cal_inequality(features_list)
                if ineq == -1:
                    continue
                else:
                    feature_ineq_dict[c] = ineq
        median_ineq = np.median(list(feature_ineq_dict.values()))
        ineq_list.append(median_ineq)

    print("The median inequality is {}".format(ineq_list))

    return ineq_list


def plot_embeddings(X, labels, path, file_name, n_clusters = 14):
    model = TSNE(n_components=2, random_state = 603)
    node_pos = model.fit_transform(X)

    color_idx = {}
    for i in range(X.shape[0]):
        color_idx.setdefault(labels[i], [])
        color_idx[labels[i]].append(i)

    plt.clf()
    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.savefig('{}{}_nclu_{}_TSNE.png'.format(path, file_name, n_clusters))

def plot_map(title, labels, result_path = '../result/', n_clusters = 13, savefig = True, con = "rook"):
    spa_ID = genfromtxt('../../../Data/spa.csv', delimiter=',')

    ID_Union = sorted([str(int(ct)) for ct in spa_ID])
    CID = range(0,len(ID_Union))
    ID_dict = dict(zip(ID_Union, CID))

    shp = "../../../Data/shp/wi_ct_demo_fte_0307.shp"
    map_df = gpd.read_file(shp)
    #map_df.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)

    map_df = map_df[['ct','geometry']]
    map_df['ct'] = map_df['ct'].astype(str).str[0:11]
    map_df['id'] = map_df['ct'].map(ID_dict)
    map_df.sort_values(['ct'])

    plot(labels, map_df, title, result_path, n_clusters, savefig, con)    

def plot(result, map_df, title, path, n_clusters, savefig = True, con = 'rook'):
    map_df['labels'] = map_df.apply(lambda row: result[row['id']], axis=1)
    n = len(map_df)
    fig, ax = plt.subplots(1, 1, figsize = (10,10))
    missing_kwds = None
    #new_cmap = rand_cmap(100, type='bright', first_color_black=True, last_color_black=False)
    map_df.plot(column = 'labels',cmap = 'tab20',ax = ax, categorical=True , legend=True,  \
        legend_kwds={'loc':'upper left','bbox_to_anchor':(1.02, 1), 'title': 'Cluster ID', 'ncol': 1}, \
            alpha = 0.7, edgecolor = "lightgray", missing_kwds = missing_kwds)     
    if savefig == True:
        if 'csv' in title:
            title = title[:-4]
        plt.savefig('{}{}_nclu_{}_{}_visualization.png'.format(path, title, n_clusters, con), dpi = 300)   


def kmeans(path, n_clusters):
    file_name = 'kmeans'
    features = np.loadtxt('../data/feature_matrix_f1.csv', delimiter=',') #use updated features   
    X = features[:,1:]
    kmeans = KMeans(n_clusters=n_clusters, random_state= 425).fit(X)
    labels = kmeans.labels_


    #plot_embeddings(X, labels, path, "kmeans", n_clusters)

    total_ratio, ratio_norm = intra_inter_idx(labels, n_clusters)
    median_ineq = community_inequality(labels, file_name, path, n_clusters)
    ineq_ls = community_inequality_seperate(labels, file_name,path, n_clusters)

    median_sim, median_dist = similarity(labels, file_name, path, n_clusters)
    homo_score = lwinc_purity(labels)
    
    #plot_embeddings(X, labels, path, file_name, savefig = True)
    
    print('\nThe normalized cosine similarity is {:.3f}'.format(median_sim*ratio_norm))
    print('The normalized homogeneous score is {:.3f}'.format(homo_score*ratio_norm))
    print('The normalized euclidean distance score is {:.3f}\n'.format(median_dist/ratio_norm))

    return labels, total_ratio, median_ineq, median_sim, median_dist, homo_score, ineq_ls


def louvain(path):
    flow = genfromtxt('../../../Data/Flow_matrix.csv', delimiter=',')
    G_flow = nx.from_numpy_array(flow)

    comm = nx_comm.louvain_communities(G_flow, seed=524, weight='weight')
    #print(comm)

    labels = np.empty(G_flow.number_of_nodes())
    for i in range(len(comm)):
        labels[list(comm[i])] = i

    n_clusters = len(comm)

    file_name = "louvain"
    estimate_HPSA(labels) 
    total_ratio, ratio_norm = intra_inter_idx(labels, n_clusters)
    median_ineq = community_inequality(labels, file_name, path, n_clusters)
    ineq_ls = community_inequality_seperate(labels, file_name, path, n_clusters)

    median_sim, median_dist = similarity(labels, file_name, path, n_clusters)
    homo_score = lwinc_purity(labels)
    
    #plot_embeddings(X, labels, path, file_name, savefig = True)
    
    print('\nThe normalized cosine similarity is {:.3f}'.format(median_sim*ratio_norm))
    print('The normalized homogeneous score is {:.3f}'.format(homo_score*ratio_norm))
    print('The normalized euclidean distance score is {:.3f}\n'.format(median_dist/ratio_norm))

    return labels, total_ratio, median_ineq, median_sim, median_dist, homo_score, ineq_ls


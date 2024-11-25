import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics


EPS = 1e-15
def run_aggclustering(path, file_name, affinity, n_clusters, linkage = 'ward'):
    #print(n_clusters)
    X = np.loadtxt(path+file_name, delimiter=' ')
    if 'csv' in file_name:
        file_name = file_name[:-4]

    adj = np.loadtxt('../data/Spatial_matrix_rook.csv', delimiter=',')
    model = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters, connectivity = adj, affinity=affinity)

    model.fit(X)
    labels = model.labels_

    homo_score = lwinc_purity(labels)
  
    total_ratio = intra_inter_idx(labels, n_clusters)
    median_ineq = community_inequality(labels, file_name, path, n_clusters)

    median_sim, median_dist = similarity(labels, file_name, path, n_clusters)

    return labels, total_ratio, median_ineq, median_sim, median_dist, homo_score


#generate the homogeneous score
def lwinc_purity(labels, lwinc_file = "../data/feature_matrix_lwinc.csv"):
    X_lwinc = np.loadtxt(lwinc_file, delimiter=',') 
    X_lwinc = X_lwinc[:,1:]
    n_thres = 5
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

    return total_ratio


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


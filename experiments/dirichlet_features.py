import numpy as np
import tensorflow as tf
from pyemd import emd_samples as emd_fn
from pyemd import emd as emd_h
import os,sys
from os.path import join, split
import matplotlib.pyplot as plt

from fedlearning import utils, datautils, dnn, argParser, feature_extraction
from sklearn.cluster import KMeans

parser  = argParser.argParser()
args    = parser.args
print(args)
alpha        = args.alpha
nclients     = args.nclients
dataset_type = args.dataset_type
emd_l_max    = 1.0
utils    = utils.Utils()
dnn      = dnn.DNN()
ds       = datautils.DataUtils(args)
fe       = feature_extraction.FeatureExtraction()
HOMEDIR  = utils.HOMEDIR
DATADIR  = utils.DATADIR


n_cluster_type = {"cifar10":10,"cifar100":10,"fashion_mnist":10,"imdb":10,"reuters":10,
                  "genomic":{30:[10,10,10],
                             10:[3,4,3],
                             5:[1,2,2]}
                  }

def init_kmean(n_clusters):
    kmeans = KMeans(
        init="k-means++",
        n_clusters=n_clusters,
        n_init=10,
        max_iter=300,
        random_state=42,
        verbose = 0,
    )
    return kmeans


def compute_emd(clusters,client_idcs,client_1,client_2,dist_matrix):
    samples1     = clusters[client_idcs[client_1]]
    samples2     = clusters[client_idcs[client_2]]
    nbins = max(int(samples1.max()+1),int(samples2.max()+1))
    hist1        = np.array(np.histogram(samples1,bins=nbins,
                                     density=True)[0]).astype(float)
    hist2        = np.array(np.histogram(samples2,bins=nbins,
                                     density=True)[0]).astype(float)
    return emd_h(hist1,hist2,dist_matrix)

def main(alpha,nclients,dataset_type,emd_l_max):
    OUTDIR   = join(DATADIR,"training","distributed",dataset_type)
    utils.create_dir(OUTDIR)
    PATH_TO_CLUSTER_INDICES = join(OUTDIR,"cluster_indices.npz")

    inputs, labels, features_encoded = ds.get_dataset(dataset_type,encoded=False)
    features_encoded = fe.extract(inputs)
    print("features shape",features_encoded.shape)
    kmeans = init_kmean(n_cluster_type[dataset_type])
    kmeans.fit(features_encoded)
    features_cluster_idc = kmeans.labels_
    # max_emd = emd_fn(np.ones(features_encoded.shape),np.zeros(features_encoded.shape))
    dist_matrix_features = np.ones([n_cluster_type[dataset_type],n_cluster_type[dataset_type]]).astype(float)
    dist_matrix_labels   = np.ones([labels.max()+1,labels.max()+1]).astype(float)
    np.fill_diagonal(dist_matrix_features,0)
    np.fill_diagonal(dist_matrix_labels,0)
    while True:
        emd_feature_hist,emd_label_hist = [],[]
        train_idcs  = np.random.permutation(features.shape[0])
        client_idcs = utils.split_noniid(train_idcs, labels, alpha=alpha, nclients=nclients)
        flag_continue=True
        client_idcs_sizes = np.zeros(nclients)
        for i,client_idc in enumerate(client_idcs):
            client_idcs_sizes[i]=len(client_idc)
            if len(client_idc)<2:
                flag_continue = False
        if not flag_continue:
            print("One client sample to low")
            emd_feature_hist.append(emd_features)
            emd_label_hist.append(emd_labels)
            continue
        # print("cl_1","\t","cl_2","\t","emd","\t","emd_l","\t", "N1","\t","N2")
        for client_1 in range(nclients):
            for client_2 in range(client_1+1,nclients):
                emd_features = compute_emd(features_cluster_idc,client_idcs,client_1,client_2,dist_matrix_features)
                emd_labels   = compute_emd(labels,client_idcs,client_1,client_2,dist_matrix_labels)
                emd_feature_hist.append(emd_features)
                emd_label_hist.append(emd_labels)
        emd_feature_hist,emd_label_hist=np.array(emd_feature_hist),np.array(emd_label_hist)
        if round(np.mean(emd_feature_hist),4)*100 <3.0 \
            and round(np.mean(emd_feature_hist),4)*100 >2.0 \
            and client_idcs_sizes.std()<0.1*client_idcs_sizes.mean():
            for i,client_idc in enumerate(client_idcs):
                print(i,client_idcs_sizes[i])
            print("<emd feature> ",str(round(np.mean(emd_feature_hist),4)*100)[0:5],"+-",str(np.round(np.std(emd_feature_hist),4))[0:5],
                  "\t <emd label>",str(round(np.mean(emd_label_hist),4)*100)[0:5]  ,"+-",str(np.round(np.std(emd_label_hist),4))[0:5])
            emd_f      = str(np.mean(emd_feature_hist)*100)[0:3]
            emd_l      = str(np.mean(emd_label_hist)*100)[0:3]
            subdir     = "N"+str(nclients)+"_alpha"+str(alpha)
            SUBDIR     = join(OUTDIR,subdir)
            utils.create_dir(SUBDIR)
            for client, indices in enumerate(client_idcs):
                filepath = join(OUTDIR,subdir,str(client)+".npz")
                np.savez(filepath,indices=indices)
            print("Saved to "+SUBDIR,"\n")


if __name__ == '__main__':
  
    main(alpha,nclients,dataset_type,emd_l_max)
    # python3 dirichlet_emd.py --nclients 10 --dataset cifar10 --alpha 1

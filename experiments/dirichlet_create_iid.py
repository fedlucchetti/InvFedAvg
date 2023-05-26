import numpy as np
import tensorflow as tf
from pyemd import emd_samples as emd_fn
from pyemd import emd as emd_h
import os,sys
from os.path import join, split
import matplotlib.pyplot as plt
from fedlearning import utils, datautils, dnn
from sklearn.cluster import KMeans

utils    = utils.Utils()
dnn      = dnn.DNN()
ds       = datautils.DataUtils()
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
    if dataset_type=="genomic":
        for id, geolocation in enumerate(["afri","asia","euro"]):
            _features, _labels, _features_encoded = ds.get_dataset(dataset_type,geolocation=geolocation,encoded=True)
            _labels_multiclass = _labels
            _labels=np.sum(_labels,axis=1)
            print(_labels_multiclass.shape)
            print(_labels.shape)
            for id_l, l in enumerate(_labels):
                if l>0: _labels[id_l]=np.where(_labels_multiclass[id_l]==1)[0][0]+1
                else:   _labels[id_l]=0
            if id!=0:
                features_encoded = np.concatenate([features_encoded,_features_encoded])
                labels           = np.concatenate([labels,_labels])
            else:features_encoded,labels=_features_encoded,_labels
        dist_matrix_features = np.ones([10,10]).astype(float)
        dist_matrix_labels   = np.ones([int(labels.max()),int(labels.max())]).astype(float)
    else:
        features, labels, features_encoded = ds.get_dataset(dataset_type,encoded=True)
        dist_matrix_features = np.ones([10,10]).astype(float)
        dist_matrix_labels   = np.ones([labels.max()+1,labels.max()+1]).astype(float)
    np.fill_diagonal(dist_matrix_features,0)
    np.fill_diagonal(dist_matrix_labels,0)

    if not os.path.exists(PATH_TO_CLUSTER_INDICES):
        kmeans = init_kmean(n_clusters=10)
        kmeans.fit(features_encoded)
        features_cluster_idc = kmeans.labels_
        np.savez(PATH_TO_CLUSTER_INDICES,features_cluster_idc=features_cluster_idc)
    else:
        print("Loading cluster indices from file")
        features_cluster_idc = np.load(PATH_TO_CLUSTER_INDICES)["features_cluster_idc"]
    print("feature shape",features_encoded.shape,"labels shape",labels.shape)


    while True:
        emd_feature_hist,emd_label_hist = [],[]
        train_idcs  = np.random.permutation(features_encoded.shape[0])
        client_idcs = utils.split_noniid_multiclass(train_idcs,features_cluster_idc, labels, alpha, nclients)
        # client_idcs = utils.split_noniid(train_idcs, labels, alpha=alpha, nclients=nclients)
        flag_continue=True
        for client_idc in client_idcs:
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
        # if round(np.mean(emd_feature_hist),4)*100 >0.0:
        print("<emd feature> ",str(round(np.mean(emd_feature_hist),4)*100)[0:5],"+-",str(np.round(np.std(emd_feature_hist),4))[0:5],
              "\t <emd label>",str(round(np.mean(emd_label_hist),4)*100)[0:5]  ,"+-",str(np.round(np.std(emd_label_hist),4))[0:5])
        if  np.mean(emd_feature_hist)*100<emd_l_max and  np.mean(emd_label_hist)*100<emd_l_max:
            emd_f      = str(np.mean(emd_feature_hist)*100)[0:3]
            emd_l      = str(np.mean(emd_label_hist)*100)[0:3]
            subdir     = "N"+str(nclients)+"_EMD_f"+emd_f+"_EMD_l"+emd_l
            SUBDIR=join(OUTDIR,subdir)
            print("Saving to "+SUBDIR)
            utils.create_dir(SUBDIR)
            for client, indices in enumerate(client_idcs):
                filepath = join(OUTDIR,subdir,str(client)+".npz")
                np.savez(filepath,indices=indices)


if __name__ == '__main__':
    try:
        for idx,arg in enumerate(sys.argv):
            if arg=="--nclients"  or arg=="-n":
                nclients     = int(sys.argv[idx+1])
            elif arg=="--dataset" or arg=="-d":
                dataset_type = sys.argv[idx+1]
            elif arg=="--alpha"   or arg=="-a":
                alpha        = float(sys.argv[idx+1])
            elif arg=="--emd_l_max"   or arg=="-e":
                emd_l_max        = float(sys.argv[idx+1])
    except Exception as e:
        print(e)
        sys.exit()

    main(alpha,nclients,dataset_type,emd_l_max)
    # python3 dirichlet_emd.py --nclients 10 --dataset cifar10 --alpha 1

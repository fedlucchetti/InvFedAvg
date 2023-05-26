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




def emd_random_field(sf1,sf2,nbins=10):
    sf1,sf2=sf1/np.max(sf1)*nbins,sf2/np.max(sf2)*nbins
    dist_matrix = np.ones([nbins,nbins]).astype(float)
    np.fill_diagonal(dist_matrix,0)
    e=np.zeros(sf1.shape[1])
    for idx_feature in range(sf1.shape[1]):
        hist1=np.histogram(sf1[:,idx_feature],bins=nbins,density=True,range=(0,nbins))[0].astype(float)
        hist2=np.histogram(sf2[:,idx_feature],bins=nbins,density=True,range=(0,nbins))[0].astype(float)
        e[idx_feature]=emd_h(hist1,hist2,dist_matrix)
        # print(hist1.sum(),hist2.sum())
    return np.array(e)

def sort_wrt_class(features,labels):
    labels=labels[:,0]
    idx_sort = np.argsort(labels)
    n_samples_per_class=[]
    for _l in range(np.max(labels)+1):
        n_samples_per_class.append(np.where(labels[idx_sort]==_l)[0].shape)
    return features[idx_sort], labels[idx_sort], np.array(n_samples_per_class)[:,0]



def main(nclients,dataset_type):
    OUTDIR                   = join(DATADIR,"training","distributed_1",dataset_type)
    utils.create_dir(OUTDIR)
    features, labels, features_encoded          = ds.get_dataset(dataset_type,encoded=True)
    features_encoded,labels,n_samples_per_class = sort_wrt_class(features_encoded,labels)

    while True:
        n_classes                = labels.max()+1
        emd                      = list()
        client_idcs              = np.empty([nclients,1])
        for idx_class in range(n_classes):
            idc_label            = np.where(labels==idx_class)[0]
            kmeans               = init_kmean(nclients)
            kmeans.fit(features_encoded[idc_label])
            features_cluster_idc = kmeans.labels_
            h                    = np.histogram(features_cluster_idc,bins=nclients)[0]
            n_max                = np.max(h)
            cluster_idcs_per_class               = np.zeros([nclients,n_max]).astype(int)
            cluster_idcs_per_class[np.argmax(h)] = idc_label[np.where(features_cluster_idc==np.argmax(h))[0]]
            for idx_client in range(nclients):
                if idx_client==np.argmax(h): continue
                cluster_idc           = np.where(features_cluster_idc==idx_client)[0]
                n_samples_2add        = n_max - h[idx_client]
                array_idc_2add        = cluster_idc[np.random.randint(0,high=cluster_idc.size, size=n_samples_2add)]
                cluster_idcs_per_class[idx_client] = idc_label[np.concatenate([cluster_idc,array_idc_2add])]
                sf1                   = features_encoded[cluster_idcs_per_class[idx_client]]
                sf2                   = features_encoded[cluster_idcs_per_class[np.argmax(h)]]
                emd.append(emd_random_field(sf1,sf2,nbins=10).mean())
            client_idcs=np.concatenate([client_idcs,cluster_idcs_per_class],axis=1)
        client_idcs=client_idcs[:,1::].astype(int)
        emd_avg=round(np.array(emd).mean()*100)
        print("Distribution with EMD", emd_avg,"found")
        subdir     = join(OUTDIR,"N"+str(nclients)+"_EMD_"+str(emd_avg))
        utils.create_dir(subdir)
        print("Saving to ",subdir)
        for client, indices in enumerate(client_idcs):
            filepath = join(subdir,str(client)+".npz")
            np.savez(filepath,indices=indices)

if __name__ == '__main__':
    try:
        for idx,arg in enumerate(sys.argv):
            if arg=="--nclients"  or arg=="-n":
                nclients     = int(sys.argv[idx+1])
            elif arg=="--dataset" or arg=="-d":
                dataset_type = sys.argv[idx+1]
    except Exception as e:
        print(e)
        sys.exit()

    main(nclients,dataset_type)
    # python3 distribute_dataset.py --nclients 30 --dataset cifar10

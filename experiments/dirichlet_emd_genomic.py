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
OUTDIR   = join(DATADIR,"training","distributed","genomic")
utils.create_dir(OUTDIR)
PATH_TO_CLUSTER_INDICES = join(OUTDIR,"cluster_indices.npz")

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


def compute_emd(clusters,client_idcs,client_1,client_2,dist_matrix,islabel=False):
    samples1     = clusters[client_idcs[client_1]]
    samples2     = clusters[client_idcs[client_2]]
    nbins = max(int(samples1.max()+1),int(samples2.max()+1))
    if islabel:
        samples1 = samples1[samples1!=0]
        samples2 = samples2[samples2!=0]
        nbins = max(int(samples1.max()),int(samples2.max()))
    # print("samples1 mean",samples1.mean(),"samples2 mean",samples2.mean())
    hist1        = np.array(np.histogram(samples1,bins=nbins,
                                     density=True)[0]).astype(float)
    hist2        = np.array(np.histogram(samples2,bins=nbins,
                                     density=True)[0]).astype(float)
    return emd_h(hist1,hist2,dist_matrix)


def main(alpha,nclients,dataset_type,emd_l_max):
    for id, geolocation in enumerate(["afri","asia","euro"]):
        _features, _labels, _features_encoded = ds.get_dataset(dataset_type,geolocation=geolocation,encoded=True)
        _labels_multiclass = _labels
        _labels=np.sum(_labels,axis=1)
        for id_l, l in enumerate(_labels):
            if l>0: _labels[id_l]=np.where(_labels_multiclass[id_l]==1)[0][0]+1
            else:   _labels[id_l]=0
        if id!=0:
            labels_multiclass = np.concatenate([labels_multiclass,_labels_multiclass])
            features_encoded = np.concatenate([features_encoded,_features_encoded])
            labels           = np.concatenate([labels,_labels])
        else:features_encoded,labels,labels_multiclass=_features_encoded,_labels,_labels_multiclass
    if not os.path.exists(PATH_TO_CLUSTER_INDICES):
        kmeans = init_kmean(n_clusters=10)
        kmeans.fit(features_encoded)
        features_cluster_idc = kmeans.labels_
        np.savez(PATH_TO_CLUSTER_INDICES,features_cluster_idc=features_cluster_idc)
    else:
        features_cluster_idc = np.load(PATH_TO_CLUSTER_INDICES)["features_cluster_idc"]

    print("features_encoded shape",features_encoded.shape,"labels shape",labels.shape)
    dist_matrix_features = np.ones([int(features_cluster_idc.max()+1),int(features_cluster_idc.max()+1)]).astype(float)
    dist_matrix_labels   = np.ones([int(labels.max()),int(labels.max())]).astype(float)
    # dist_matrix_labels   = np.ones([labels_multiclass.shape[1],labels_multiclass.shape[1]]).astype(float)
    np.fill_diagonal(dist_matrix_features,0)
    np.fill_diagonal(dist_matrix_labels  ,0)
    dist_matrix_labels[0,:]=0 # zero label at pos 1 of seq overrepresented -> no weight on EMD
    dist_matrix_labels[:,0]=0 # symmetric
    while True:
        emd_feature_hist,emd_label_hist = [],[]
        client_idcs = []
        for id, geolocation in enumerate(["afri","asia","euro"]):
            crop = range(id*20000,20000*(id+1)) # crop those that belong to each geolocation
            train_idcs   = np.random.permutation(features_encoded[crop].shape[0])
            _client_idcs = utils.split_noniid(train_idcs, labels[crop],
                           alpha=alpha, nclients=n_cluster_type[dataset_type][nclients][id])
            for el in _client_idcs: client_idcs.append(el+id*20000)
            flag_continue=True
            for client_idc in client_idcs:
                if len(client_idc)<2:
                    flag_continue = False
            if not flag_continue: print("One client sample size too low: continue \n"); continue
        for client_1 in range(nclients):
            for client_2 in range(client_1+1,nclients):
                emd_features = compute_emd(features_cluster_idc,client_idcs,client_1,client_2,dist_matrix_features)
                emd_labels   = compute_emd(labels,client_idcs,client_1,client_2,dist_matrix_labels,islabel=True)
                emd_feature_hist.append(emd_features)
                emd_label_hist.append(emd_labels)
        emd_feature_hist,emd_label_hist=np.array(emd_feature_hist),np.array(emd_label_hist)
        if round(np.mean(emd_feature_hist),4)*100 >0.5:
            print("<emd feature> ",str(round(np.mean(emd_feature_hist),4)*100)[0:5],"+-",str(np.round(np.std(emd_feature_hist),4))[0:5],
                  "\t <emd label>",str(round(np.mean(emd_label_hist),4)*100)[0:5]  ,"+-",str(np.round(np.std(emd_label_hist),4))[0:5])
        if  np.mean(emd_label_hist)*100<emd_l_max:
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

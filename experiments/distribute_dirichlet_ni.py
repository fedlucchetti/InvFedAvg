import numpy as np
import tensorflow as tf
from pyemd import emd_samples as emd_fn
from pyemd import emd as emd_h
import os,sys, json
from os.path import join, split
import matplotlib.pyplot as plt
from numpy.linalg import norm
from fedlearning import utils, datautils, dnn, argParser, feature_extraction
from sklearn.cluster import KMeans

parser  = argParser.argParser()
args    = parser.args
print(args)
alpha        = args.alpha
nclients     = args.nclients
dataset_type = args.dataset_type
emd_l_max    = 1.0
fe       = feature_extraction.FeatureExtraction()

utils    = utils.Utils()
dnn      = dnn.DNN()
ds       = datautils.DataUtils(args)
HOMEDIR  = utils.HOMEDIR
DATADIR  = utils.DATADIR


def compute_ni(classes,features_encoded,labels,client_idcs,mu_arr):
    ni_per_class=[]
    for class_idx in classes:
        iid = np.zeros(features_encoded[0].shape)
        for client, indices in enumerate(client_idcs):
            labels_sel = np.where(labels[indices]==class_idx)[0]
            if len(labels_sel)==0:continue
            iid       += features_encoded[labels_sel].mean(axis=0)
        ni_per_class.append(norm((mu_arr[class_idx]-iid/len(client_idcs))))
    return np.array(ni_per_class)


def main(alpha,nclients,dataset_type,emd_l_max):
    OUTDIR   = join(DATADIR,"training","distributed",dataset_type)
    utils.create_dir(OUTDIR)
    PATH_TO_CLUSTER_INDICES = join(OUTDIR,"cluster_indices.npz")
    features, labels, features_encoded = ds.get_dataset(dataset_type,encoded=True,rescale=True)
    classes = np.arange(0,np.max(labels)+1)
    mu_arr,sigma_arr,label_distr = [],[],[]
    for class_sel in classes:
        class_idc = np.where(labels==class_sel)[0]
        label_distr.append(len(class_idc))
        mu_arr.append(features_encoded[class_idc].mean(axis=0))
        sigma_arr.append(features_encoded[class_idc].var(axis=0))
    ni_min_max=np.array([100,-1])
    while True:
        emd_feature_hist,emd_label_hist = [],[]
        train_idcs  = np.random.permutation(features.shape[0])
        client_idcs = utils.split_noniid(train_idcs, labels[:,0], alpha=alpha, nclients=nclients)
        flag_continue=True
        client_idcs_sizes = np.zeros(nclients)
        for i,client_idc in enumerate(client_idcs):
            client_idcs_sizes[i]=len(client_idc)
            if len(client_idc)<128:
                flag_continue = False
                break
        if not flag_continue:continue
        ni_per_class = compute_ni(classes,features_encoded,labels,client_idcs,mu_arr)
        for _ni in ni_per_class:
            if np.isnan(_ni): flag_continue = False;break
        if not flag_continue:continue
        ni_avg       = np.nanmean(ni_per_class)
        ni_min_max = np.array([min(ni_avg,ni_min_max[0]),max(ni_avg,ni_min_max[1])])
        print("Feature NonIID:",round(ni_avg,3) )
        # print("Feature NonIID:",ni_min_max )
        meta = {"ni_feature":ni_avg,"ni_feature_per_class":list(ni_per_class.round(3))}
        input_arr = np.array([[alpha,ni_avg]])
        data      = np.load("cifar10_alpha_ni.npz")["data"]
        data      = np.concatenate([data,input_arr])
        np.savez("cifar10_alpha_ni.npz",data=data)
        # if ni_avg> 1.70 or ni_avg < 0.8:
        #     # continue
        #     ni_avg = round(ni_avg,2)
        #     subdir = join(OUTDIR,"N"+str(nclients)+"_alpha"+str(alpha)+"_ni"+str(ni_avg))
        #     utils.create_dir(subdir)
        #     with open(join(subdir,'meta.json'), 'w') as outfile:
        #         json.dump(meta, outfile)
        #     for client, indices in enumerate(client_idcs):
        #         filepath = join(OUTDIR,subdir,str(client)+".npz")
        #         np.savez(filepath,indices=indices)
        #     print("Saved to "+subdir,"\n")
        #     # break


if __name__ == '__main__':
  
    main(alpha,nclients,dataset_type,emd_l_max)
    # python3 dirichlet_emd.py --nclients 10 --dataset cifar10 --alpha 1

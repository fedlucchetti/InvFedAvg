import numpy as np
import tensorflow as tf
from pyemd import emd_samples as emd_fn
import os,sys
from os.path import join, split
import matplotlib.pyplot as plt
from fedlearning import utils, dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

utils = utils.Utils()
ds    = dataset.DataSet()
HOMEDIR = utils.HOMEDIR
DATAPATH = join(HOMEDIR,"data")
utils.create_dir(join(HOMEDIR,"data"))

n_cluster_type = {"cifar10":10,"fashion_mnist":10,"imdb":10,"reuters":10}

def init(dataset_type):
    encoder = tf.keras.models.load_model(join(DATAPATH,"encoders",dataset_type+".h5"))
    encoder.compile(optimizer='adam', loss='mse')
    features, labels = ds.get_dataset(dataset_type)
    features = features/features.max()
    features_encoded = encoder(features)
    features_encoded = np.reshape(features_encoded,[features_encoded.shape[0],
                                 np.prod(features_encoded.shape[1::])])
    return encoder, features, labels, features_encoded

def init_kmean(n_clusters):
    kmeans = KMeans(
        init="k-means++",
        n_clusters=n_clusters,
        n_init=10,
        max_iter=300,
        random_state=42,
        verbose = 0,
    )
    return kmean

def find_optimal_ncluster():
    sse=[]
    encoder, _, _, features_encoded = init(dataset_type)
    n_clusters = [2,5,8,10,16,20,32,64]
    for n_cluster in n_clusters:
        kmean = init_kmean(n_cluster)
        print("kmean clustering with ", n_cluster," clusters")
        kmeans.n_clusters = n_cluster
        kmeans.fit(features_encoded)
        # kmeans.fit(features_encoded)
        sse.append(kmeans.inertia_)
        print("got sse", kmeans.inertia_," and silhouette" ,silscore[-1], "\n")
    plt.plot(n_clusters,sse)

def distribute():
    pass

if __name__ == '__main__':
    try:
        for idx,arg in enumerate(sys.argv):
            elif arg=="--dataset":
                dataset         = sys.argv[idx+1]
            elif arg=="--exp":
                experiment      = sys.argv[idx+1]
    except Exception as e:
        print(e)
        sys.exit()

    if experiment=="distribute":
        distribute()
    if experiment=="optimal":
        find_optimal_ncluster()

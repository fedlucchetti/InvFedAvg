import tensorflow as tf
import pyemd, sys, os, json
from os.path import join, split
from tqdm import tqdm
import numpy as np
from multiprocessing import Process, Manager
from fedlearning import utils, bcolors
# import utils, bcolors
from sklearn.model_selection import train_test_split
from tensorflow.data import Dataset as tf_dataset

tensor2ds = tf_dataset.from_tensor_slices



bcolors  = bcolors.bcolors()
utils    = utils.Utils()
DATADIR  = utils.DATADIR

class DataUtils(object):
    def __init__(self,args=None):
        self.DATASETDIR    = utils.DATASETDIR
        try:
            self.options = args.options
        except Exception:
            self.options = args

    def __reshape_features(self,data_array):
        """
        scales features into [0,1] range and flattens pixel data
        """
        scaled_features = data_array.astype(float)
        scaled_features = np.reshape(scaled_features,[scaled_features.shape[0],np.prod(scaled_features.shape[1::])])
        if self.options.dataset_type == "genomic":
            scaled_features = self.encoder.predict(scaled_features)
            print("encoded feature dimension",scaled_features.shape)
            scaled_features = np.reshape(scaled_features,[scaled_features.shape[0],np.prod(scaled_features.shape[1::])])
            scaled_features += scaled_features.min()
            scaled_features = scaled_features/scaled_features.max()
        return scaled_features


    def get_dataset(self,dataset_type,geolocation=None,encoded=False,flatten_features=True,rescale=True):
        features_encoded = None
        if dataset_type=="genomic" and geolocation not in ["afri","asia","euro","all"]:
            print("get_dataset: Wrong geolocation tag")
            return None,None,None
        if dataset_type=="cifar10":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            features,labels = np.concatenate([x_train,x_test]), np.concatenate([y_train,y_test])
        elif dataset_type=="fashion_mnist":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            features,labels = np.concatenate([x_train,x_test]), np.concatenate([y_train,y_test])
        elif dataset_type=="cifar100":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
            features,labels = np.concatenate([x_train,x_test]), np.concatenate([y_train,y_test])
        elif dataset_type=="imdb":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()
            # print("x_train y_train shape:",x_train.shape,y_train.shape)
            features,labels = np.concatenate([x_train,x_test]), np.concatenate([y_train,y_test])
        elif dataset_type=="reuters":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data()
            features,labels = np.concatenate([x_train,x_test]), np.concatenate([y_train,y_test])
        elif dataset_type=="genomic":
            if geolocation=="all":
                print("Loading ALL GENOMIC DS")
                for i,geolocation in enumerate(["afri","asia","euro"]):
                    path = join(DATADIR,"training","encoded","genomic","data_"+geolocation+".npz")
                    data = np.load(path)
                    _features,_labels = data["features"][:,0:30,:],data["labels"][:,0:30]
                    if i==0:
                        features,labels = _features,_labels
                    else:
                        features,labels = np.concatenate([features,_features]), np.concatenate([labels,_labels])
        else:
            print("Wrong dataset type, EXIT")
            sys.exit()
        if encoded:
            features_encoded = self.encode_format_dataset(features,dataset_type,flatten_features)
            self.feature_set,self.label_set = features_encoded, labels
        else:
            self.feature_set,self.label_set = features, labels
        if rescale:
            features = features/features.max()
        return features,labels, features_encoded


    def encode_format_dataset(self,features, dataset_type,flatten_features=True):
        # features, labels = self.get_dataset(dataset_type)
        print("init dataset_type",dataset_type)
        if dataset_type!='genomic' and dataset_type!='imdb': ## genomic and imdb already encoded
            encoder = tf.keras.models.load_model(join(DATADIR,"encoders",dataset_type+".h5"))
            # encoder.summary()
            # encoder.compile(optimizer='adam', loss='mse')
            features = features/features.max()
            features_encoded = encoder.predict(features)
            features_encoded = np.reshape(features_encoded,[features_encoded.shape[0],
                                         np.prod(features_encoded.shape[1::])])
        elif dataset_type=='imdb':
            m=0
            for i in features:m=max(m,len(i))
            features_encoded = np.zeros([features.shape[0],m])
            for id, el in enumerate(features):
                features_encoded[id,0:len(el)] = el
        elif dataset_type=='genomic':
            features_encoded = features
            if flatten_features:
                features_encoded = np.reshape(features_encoded,[features_encoded.shape[0],
                                             np.prod(features_encoded.shape[1::])])
        return features_encoded




    def load_data_from_partition(self,client_id,emd=0.0,alpha=None,nclients=None,ni=None):
        if emd==None and alpha!=None and ni!=None:
            client_idc_path = join(self.DATASETDIR,
                                   self.options.dataset_type,
                                   "N"+str(nclients)+"_alpha"+str(alpha)+"_ni"+str(ni),
                                    str(client_id)+".npz")
            # print("Get dataset partition \n Load partition from",
            # client_idc_path)
        else:
            # print("load_data_from_partition: Error, no alpha or emd specified")
            sys.exit()           
        if nclients==1:
            X,y = self.feature_set,self.label_set
        else:
            client_indices  = np.load(client_idc_path)["indices"]
            X,y = self.feature_set[client_indices],self.label_set[client_indices]
        if "cifar10" == self.options.dataset_type or "fashion_mnist" == self.options.dataset_type :
            X = X/X.max()
            # X = np.reshape(X,[X.shape[0],X.shape[1],X.shape[2],1])/ scale_factor
        Xtrain,Ytrain,Xval,Yval=self.split_datasets(X,y)
        # print("Client dataset shapes",X.shape,y.shape)
        return Xtrain,Ytrain,Xval,Yval

    def split_datasets(self,features,labels,split=0.8):
        # Xtrain,Ytrain,Xval,Yval = train_test_split(features, labels,
                            # test_size=1-split, random_state=42)

        n_samples = features.shape[0]
        idx = np.arange(0,n_samples)
        np.random.shuffle(idx)
        Xtrain=features[idx[0:int(n_samples*split)]]
        Ytrain=labels[idx[0:int(n_samples*split)]]
        Xval=features[idx[int(n_samples*split)::]]
        Yval=labels[idx[int(n_samples*split)::]]
        return Xtrain,Ytrain,Xval,Yval


    def numpy_to_dataset(self,inputs,labels):
        dataset = tensor2ds((np.array(inputs),np.array(labels)))
        return dataset
    def dataset_to_numpy(self,dataset):
        input_shape=[dataset.cardinality().numpy()]
        label_shape=[dataset.cardinality().numpy()]
        input_shape.extend(dataset.element_spec[0].shape.as_list())
        label_shape.extend(dataset.element_spec[1].shape.as_list())
        inputs,labels = np.zeros(input_shape),np.zeros(label_shape)
        for i,el in enumerate(dataset.as_numpy_iterator()):
            inputs[i],labels[i] = el[0],el[1]
        return inputs,labels

    def load_real_geomic_dataset(self,dataset,partition):
        data=np.load(join(self.DATASETDIR,"genomic_real",dataset+partition+".npz"))
        return data["features"], data["labels"]

import numpy as np
import sys, os, json, glob, re, copy, time
from os.path import join, split
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from fedlearning import utils, dataset,  bcolors, dnn
# import utils, dataset
# import models as Models
from fedlearning import models as Models

utils = utils.Utils()
ds    = dataset.DataSet()




class Autoencoder(object):
    def  __init__(self,**kwargs):
        self.dataset_type = kwargs["dataset"]
        self.batch_size   = kwargs["batch_size"]
        self.rounds       = kwargs["rounds"]
        #####################
        # self.HOMEDIR  = split(split(os.path.realpath(__file__))[0])[0]
        self.HOMEDIR = utils.HOMEDIR
        self.DATAPATH = join(self.HOMEDIR,"data")
        ####################
        dnn = dnn.DNN(None)
        dnn.dataset_type = self.dataset_type
        self.autoencoder,self.encoder=dnn.get_autoencoder_model()
        self.autoencoder.summary()

    def load_dataset(self):
        """
        mode:
            full: load entire data set into X,Y
            train: load entire data set into a training and test set
        type:
            cifar10      : 60,000 32x32 color 100 fine-grained labeled images labels[0-9]
            cifar100     : 60,000 32x32 color labeled images labels[0-9]
            mnist        : 70,000 28x28 grayscal10e labeled handwritten digit images labels[0-9]
            fashion_mnist: 70,000 28x28 grayscale labeled fashion category images  labels[0-9]
        """
        return ds.get_dataset(self.dataset_type)

    def load_geomic_dataset(self):
        ROOTPATH=join(self.DATAPATH,"training",self.dataset_type,"geolocations")
        print("load_geomic_dataset: loading from",ROOTPATH)
        files=glob.glob(join(ROOTPATH,'*.npz'))
        n_reads=0
        for file in tqdm(files):
            print(file)
            n_reads+=np.load(join(ROOTPATH,file))["reads"].shape[0]
        reads,labels=np.zeros([n_reads,150,4]),np.zeros([n_reads,150])
        for id,file in enumerate(tqdm(files)):
            _data=np.load(join(ROOTPATH,file))
            _reads,_labels=_data["reads"],_data["labels"]
            reads[id*_reads.shape[0]:(id+1)*_reads.shape[0]]=_reads
            labels[id*_labels.shape[0]:(id+1)*_labels.shape[0]]=_labels
        print("Loaded reads",reads.shape)
        return reads, labels

    def run(self):
        x,y = self.load_dataset()
        if self.dataset_type!="genomic":
            x=x/x.max()
        ntot=x.shape[0]
        x_train=x[0:int(0.8*ntot)]
        x_test =x[int(0.8*ntot)::]

        early_stop  = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        self.autoencoder.fit(x_train, x_train,
                        epochs=self.rounds,
                        batch_size=self.batch_size,
                        shuffle=True,
                        validation_data=(x_test, x_test),
                        callbacks=[early_stop])
        outpath = join(self.DATAPATH,"encoders")
        utils.create_dir(outpath)
        outpath = join(outpath,self.dataset_type+"_1.h5")
        print("Saving to ",outpath)
        self.encoder.save(outpath)


if __name__ == '__main__':
    try:
        for idx,arg in enumerate(sys.argv):
            if arg=="--dataset" or arg=="--ds":
                dataset         = sys.argv[idx+1]
            if arg=="--batchsize" or arg=="--bs":
                batch_size         = int(sys.argv[idx+1])
            if arg=="--rounds" or arg=="--r":
                rounds         = int(sys.argv[idx+1])
    except Exception as e:
        print(e)
        sys.exit()

    ae = Autoencoder(dataset=dataset,batch_size=batch_size,rounds=rounds)
    ae.run()

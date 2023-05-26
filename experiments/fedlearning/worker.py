import numpy as np
import sys, os, json, glob, re, copy, time
from os.path import join, split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ.pop('TF_CONFIG', None)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.data import Dataset as tf_dataset
from tensorflow.keras.layers import Embedding, Input
import matplotlib.pyplot as plt
from numpy.random import randint
from tqdm import tqdm
import pandas as pd
# eager execution
import cProfile
##################################
import  argParser, datautils
##################################
bcolors = bcolors.bcolors()
utils   = utils.Utils()
##################################
tensor2ds = tf_dataset.from_tensor_slices
##################################




# def load_lossfn_metrics(self):
#     if self.options.dataset_type == "genomic" or self.options.dataset_type == "imdb":
#         self.loss_fn          = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#         self.metrics          =  ['accuracy']
#     elif self.options.dataset_type == "fashion_mnist" or self.options.dataset_type == "cifar10":
#         self.loss_fn          = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#         self.metrics          = ['accuracy']
#     elif self.options.dataset_type == "regression":
#         self.loss_fn          = tf.keras.losses.MeanAbsoluteError()
#         self.metrics          = ['accuracy']
#     else:
#         print(bcolors.FAIL,"wrong data type specs, could not specify loss function and metric", bcolors.ENDC)
#         sys.exit()
#     self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.options.lr)

# def client_update_graph(self,submodel,client_id,t,return_history=False,verbose=1):
#     # if "genomic" in self.options.dataset_type: submodel=self.stack_encoder(client_id,submodel)
#     # print("client_update_graph",client_id)
#     early_stop  = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
#     self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.options.lr)
#     datasets_train = self.datasets_train[client_id].batch(self.options.local_batchsize)
#     datasets_val = self.datasets_val[client_id].batch(self.options.local_batchsize)
#     submodel.compile(loss=self.loss_fn,optimizer=self.optimizer,metrics=self.metrics)
#     # print(bcolors.OKCYAN,"client_update_graph: Worker ",client_id, " training with batch size", self.options.local_batchsize,"and loss fn",self.loss_fn.name)
#     history = submodel.fit(datasets_train,
#                             epochs=self.options.local_epochs,
#                             validation_data=datasets_val,
#                             verbose=verbose)
#     # print("client_update_graph:","Worker ",client_id, " finished training model ",bcolors.ENDC)
#     # print('--------------------------------------------------')
#     self.purge_memory()
#     if return_history:
#         return submodel,history.history
#     else:
#         return submodel


# def worker_train(self,client_id,round_id):
#     # utils.printout('CLIENT '+str(client_id)+" ROUND "+str(round_id),bcolors.LIGHTGREY)
#     print("worker_train: client ",client_id)
#     __local_model = self.init_model()
#     if round_id==1:
#         if not args.flag_common_weight_init:
#             pass
#         else:
#             __local_model.set_weights(aggregate_weights)
#     else:
#         __local_model.set_weights(aggregate_weights)
#     __local_model = self.client_update_graph(__local_model,client_id,round_id,verbose=0)
#     __local_model._name    = args.dataset_type+"_"+str(client_id)+"_"+"t"+str(round_id)
#     print("worker_train: ",client_id,"FINISH")

# def __limit_memory(self):
#     gpus = tf.config.list_physical_devices('GPU')
#     if gpus:
#         # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#         try:
#         tf.config.set_logical_device_configuration(
#             gpus[0],
#             [tf.config.LogicalDeviceConfiguration(memory_limit=int(7143/self.options.nclients))])
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#         except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)

# def init_model(self,t=0,client_id=0,aggregation=False):
#     models=[]
#     client_id=int(client_id)
#     rootpath = os.path.realpath(__file__)
#     model = tf.keras.models.clone_model(self.dnn.get_model(client_id))

#     if t==0:
#         model._name=model.name+"_"+"t"+str(0)
#     else:
#         modelfolder = self.checkpointpath
#         modelpath   = join(modelfolder,"single_client_model_"+str(client_id)+".ckpt")
#         print("init_model: Loaded ",modelpath)
#         model.load_weights(modelpath).expect_partial()
#         model.compile(loss=self.loss_fn,optimizer=self.optimizer,metrics=self.metrics)
#         print(bcolors.OKGREEN,"init_model: Success opened ",modelpath,bcolors.ENDC)
#     return model


# def purge_memory(self):
#     tf.keras.backend.clear_session()

if __name__=='__main__':
    print(sys.argv)
    # client_id = int(sys.argv[1])
    # round     = int(sys.argv[2])
    # Xtrain,Ytrain,Xval,Yval = dutils.load_data_from_partition(client_id,
    #                                                         emd=self.options.emd,
    #                                                         alpha=self.options.alpha,
    #                                                         ni=self.options.ni,
    #                                                         nclients=self.options.nclients)
    # model = init_model()


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
from fedlearning import bcolors, dnn, utils, argParser, datautils
##################################
bcolors = bcolors.bcolors()
utils   = utils.Utils()
##################################
tensor2ds = tf_dataset.from_tensor_slices
##################################



class Server(object):

    def __init__(self,opt_dict=None):
        file_path           = os.path.realpath(__file__)
        self.HOMEDIR        = utils.HOMEDIR
        self.DATAPATH       = utils.DATADIR
        self.DATASETDIR     = utils.DATASETDIR
        if opt_dict==None:
            parser   = argParser.argParser()
            opt_dict = parser.args
        self.options = opt_dict
        if self.options.nclients == 1:
            self.options.emd       = 0.0
            self.options.algorithm = "sgd"
        self.dnn              = dnn.DNN(self)
        self.dutils           = datautils.DataUtils(self)


    def get_mc_inputs(self,size=None):
        if size==None:
            size=self.options.n_montecarlo
        inshape     = [size]
        _template_model = self.init_model()
        inshape.extend(_template_model.input.shape.as_list()[1::])
        if self.options.dataset_type=="genomic":
            inputs_mc = tf.random.uniform(shape=[self.options.n_montecarlo,30,64],
            minval=self.Xtrain.min(), maxval=self.Xtrain.max())
        elif self.options.dataset_type=="regression":
            inputs_mc = tf.random.uniform(shape=inshape, minval=-1.0, maxval=1.0)
        else:
            inputs_mc = tf.random.uniform(shape=inshape, minval=0.0, maxval=1.0)
        return inputs_mc

    def init_gmcc(self):
        self.inputs_mc = self.get_mc_inputs()

    def init_model(self,t=0,client_id=0,aggregation=False):
        models=[]
        client_id=int(client_id)
        rootpath = os.path.realpath(__file__)
        model = tf.keras.models.clone_model(self.dnn.get_model(client_id))

        if t==0:
            model._name=model.name+"_"+"t"+str(0)
        else:
            modelfolder = self.checkpointpath
            if self.algorithm=="fedoutavg":
                modelpath   = join(modelfolder,"single_client_model_"+str(client_id)+".ckpt")
            elif self.algorithm=="fedavg" or  self.algorithm=="fedprox" :
                if aggregation:
                    modelpath   = join(modelfolder,"single_client_model_"+str(client_id)+".ckpt")
                else:
                    modelpath   = join(modelfolder,"fedavg_model"+".ckpt")
            print("init_model: Loaded ",modelpath)
            model.load_weights(modelpath).expect_partial()
            model.compile(loss=self.loss_fn,optimizer=self.optimizer,metrics=self.metrics)
            print(bcolors.OKGREEN,"init_model: Success opened ",modelpath,bcolors.ENDC)
        return model


    def __limit_memory(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
          # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
          try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=int(7143/self.options.nclients))])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
          except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    def covfedavg(self,gammas,steps=10):
        '''
        apply geometric monte carlo clusteringself.loss_fn
        '''
        model1=self.init_model()
        model2=self.init_model()
        n_models_received = len(gammas)

        agg_model_weights=[]
        t_array_corrections = np.nan*np.zeros([n_models_received,n_models_received])
        for id_client_ref in  tqdm(range(n_models_received)):
            for id_client in range(id_client_ref+1,n_models_received):
                gamma_pair=[]
                gamma_pair.append(gammas[id_client_ref])
                gamma_pair.append(gammas[id_client])
                t1,t2 = self.model_interaction(gamma_pair,flag_return_model=False)
                t_array_corrections[id_client_ref,id_client] = t1
                t_array_corrections[id_client,id_client_ref] = t2
                self.purge_memory()
        t_array_corrections = np.mean(t_array_corrections,axis=1)
        for idx,t in enumerate(t_array_corrections):
            w1, w2, theta = gammas[idx][0],gammas[idx][1],gammas[idx][2]
            agg_model_weights.append(utils.gamma_curve(t,theta, w1, w2,mode=self.options.cf_curve))
        return agg_model_weights


    def normalize_client_models(self,weights_list,dataset_sizes):
        tot_n = np.sum(dataset_sizes)
        scaled_local_weight_list=list()
        for client_id, weights in enumerate(weights_list):
            scaling_factor = dataset_sizes[client_id]/tot_n
            scaled_weights = utils.scale_model_weights(weights, scaling_factor)
            scaled_local_weight_list.append(scaled_weights)
        return scaled_local_weight_list

    def fedavg(self,weights_list,dataset_sizes):
        scaled_local_weight_list = self.normalize_client_models(weights_list,dataset_sizes)
        return utils.sum_scaled_weights(scaled_local_weight_list)

    def average_gammas(self,gammas,t_array):
        weights_list=[]
        for t in t_array:
            weights        = utils.gamma_curve(t,gammas[2], gammas[0], gammas[1],mode=self.options.cf_curve)
            scaled_weights = utils.scale_model_weights(weights, 1/len(t_array))
            weights_list.append(scaled_weights)
        return utils.sum_scaled_weights(weights_list)

    def flat_curve(self,u, w1, w2):
        flat=[]
        for  _w1, _w2 in zip(w1, w2):
            flat.append((1-u)*_w1+u*_w2)
        return flat

    def model_interaction(self,gammas,flag_return_model=True):
        weights_1_A,weights_1_B,weights_1_theta = gammas[0]
        weights_2_A,weights_2_B,weights_2_theta = gammas[1]
        loss                    = list()
        u_array                 = np.linspace(0,1,16)
        loss_fn                 = tf.losses.MeanSquaredError()
        inputs_MC               = self.get_mc_inputs(size=1024)
        model_1_inv,model_2_inv = self.init_model(),self.init_model()
        model_1_inv.trainable=False;model_2_inv.trainable=False
        for i,ti in enumerate(u_array):
            model_1_inv.set_weights(utils.gamma_curve(ti,weights_1_theta, weights_1_A, weights_1_B,mode=self.options.cf_curve))
            outputs_1 = model_1_inv(inputs_MC)
            for j,tj in enumerate(u_array):
                model_2_inv.set_weights(utils.gamma_curve(tj,weights_2_theta, weights_2_A, weights_2_B,mode=self.options.cf_curve))
                l = loss_fn(outputs_1,model_2_inv(inputs_MC)).numpy()
                loss.append([int(i),int(j),l])
        loss=np.array(loss)
        t1_opt = u_array[int(loss[np.argmin(loss[:,2])][0])]
        t2_opt = u_array[int(loss[np.argmin(loss[:,2])][1])]
        print("model_interaction: correcting model 1 at:",t1_opt,"witch Lint=",np.min(loss[:,2]))
        print("model_interaction: correcting model 2 at:",t2_opt,"witch Lint=",np.min(loss[:,2]),"\n")

        if flag_return_model:
            model_1_inv.set_weights(utils.gamma_curve(t1_opt,weights_1_theta, weights_1_A, weights_1_B,mode=self.options.cf_curve))
            model_2_inv.set_weights(utils.gamma_curve(t2_opt,weights_2_theta, weights_2_A, weights_2_B,mode=self.options.cf_curve))
            return model_1_inv,model_2_inv
        else:
            return t1_opt,t2_opt


    def save_model_history(self,path,model,round=None):

        if round==None:modelpath = path+'.ckpt'
        else:modelpath = path+"_" +str(round)+ ".ckpt"
        print(bcolors.WARNING,"Saving client model to ", modelpath,bcolors.ENDC)
        model.save(modelpath)
        self.purge_memory()
        del model
        return 1

    def purge_memory(self):
        tf.keras.backend.clear_session()

    def aggregation(self,local_weight_list,dataset_sizes):
        aggregate_weights =  local_weight_list
        if len(local_weight_list)!=1:
            val_loss_covfedavg,metric_covfedavg = [],[]
            utils.printout('Server Model Aggregation for C '+str(self.options.nclients) + " EMD" + str(self.options.emd),bcolors.ORANGE)
            ############################## INIT aggregation ##############################
            agg_model=self.init_model(t=0,client_id=2323)
            ################################# covfedavg ##############################
            if self.options.algorithm=="covfedavg":
                aggregate_weights = self.covfedavg(local_weight_list)
            ################################# FedAVG ##############################
            elif self.options.algorithm=="fedavg":
                aggregate_weights = self.fedavg(local_weight_list,dataset_sizes)
        return aggregate_weights




if __name__=='__main__':
    fl=FederatedLearning()

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
# from federated import models as md
# from federated import utils
# from federated import dataset as ds


##################################
bcolors = bcolors.bcolors()
utils   = utils.Utils()
parser  = argParser.argParser()

##################################
tensor2ds = tf_dataset.from_tensor_slices
##################################



class FederatedLearning(object):

    def __init__(self,opt_dict=None):
        file_path           = os.path.realpath(__file__)
        self.HOMEDIR        = utils.HOMEDIR
        self.DATAPATH       = utils.DATADIR
        self.DATASETDIR     = utils.DATASETDIR
        self.model            = None
        self.trainratio       = 0.8
        self.testratio        = 0.05
        self.sensposition     = 0
        utils.printout("GPU available= " + str(tf.config.experimental.list_physical_devices('GPU')),bcolors.OKCYAN)
        if opt_dict!=None:
            print("Initializing FL neural net Class with following parameters")
            print(opt_dict)
        else:
            opt_dict = parser.parse(None)
            print("Invalid settings. Exit")
            print(opt_dict)
            sys.exit()
        self.dataset_type     = opt_dict.dataset

        self.mu               = opt_dict.mu
        self.nclients         = opt_dict.nclients
        self.local_batchsize  = opt_dict.batchsize
        self.local_epochs     = opt_dict.local_epochs
        self.learning_rate    = opt_dict.nu
        if self.nclients == 1:
            self.emd          = 0.0
            self.emd_l        = 0.0
            self.algorithm    = "sgd"
        else:
            self.emd          = opt_dict.emd
            self.emd_l        = opt_dict.emd_l
            self.algorithm    = opt_dict.algorithm
        self.round            = 1
        self.dnn              = dnn.DNN(self)
        self.dutils           = datautils.DataUtils(self)

    def init_models(self,t,clients,aggregation=False):
        models=[]
        for client in clients:
            models.append(self.init_model(t=t,client=client,aggregation=aggregation))
        return models

    def init_gmcc(self,n_montecarlo=1024):
        _template_mode = self.init_model(t=0,client=23)
        inshape     = [n_montecarlo]
        inshape.extend(_template_mode.input.shape.as_list()[1::])
        if self.dataset_type!="genomic":
            self.gmcc_inputs = tf.random.uniform(shape=inshape, minval=0.0, maxval=1.0)
        else:
            self.gmcc_inputs = tf.random.uniform(shape=[n_montecarlo,30,64], minval=self.Xtrain.min(), maxval=self.Xtrain.max())
            # self.gmcc_inputs=[]
            # for id in range(n_montecarlo):
            #     read=''.join(random.choices(bases, k=150))
            #     self.gmcc_inputs.append(utils.vectorize(read,word_size=5))
        del _template_mode

    def init_encoder(self,client):
        VOCAB_SIZE = 10000
        encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
        if client!=23:
            encoder.adapt(self.datasets_train[0].map(lambda text, label: text))
        else:
            datasets_train = tensor2ds((self.Xtrain[0][0:2],self.Ytrain[0][0:2]))
            encoder.adapt(datasets_train.map(lambda text, label: text))
        self.encoder=encoder
        self.dutils=encoder
        return 1


    def list_all_emd_partitions(self):
        f,emd_array = [],[]
        path2partitions  = join(self.DATASETDIR,self.dataset_type)
        for (dirpath, dirnames, filenames) in os.walk(path2partitions):
            f.extend(filenames)
            break
        for file in f:
            if ".npz" in file and "N"+str(self.nclients) in file:
                emd_array.append(np.load(join(self.DATASETDIR,self.dataset_type,file))["emd"])
        return emd_array

    def init_data_set(self):
        """
        N:Number of available partitions
        """
        flatten_features=False
        encoded=False
        if self.dataset_type=="imdb":encoded=True
        self.dutils.get_dataset(self.dataset_type,geolocation="all",encoded=encoded,flatten_features=flatten_features)
        self.Xtrain,self.Ytrain,self.Xval,self.Yval,self.Xtest,self.Ytest = [],[],[],[],[],[]
        self.datasets_train,self.datasets_val,self.datasets_test = [],[],[]
        for client in tqdm(range(self.nclients)):
            Xtrain,Ytrain,Xval,Yval = self.dutils.load_data_from_partition(client,emd=self.emd,emd_l=self.emd_l)
            print("Xtrain Ytrain shape:",Xtrain.shape,Ytrain.shape)
            self.datasets_train.append(tensor2ds((Xtrain,Ytrain)).batch(self.local_batchsize))
            self.datasets_val.append(tensor2ds((Xval,Yval)).batch(self.local_batchsize))
            self.Xtrain.append(Xtrain);self.Ytrain.append(Ytrain)
            self.Xval.append(Xval);self.Yval.append(Yval)
            self.Xtest.extend(Xval);self.Ytest.extend(Yval)
        self.Xtest,self.Ytest = np.array(self.Xtest),np.array(self.Ytest)
        self.datasets_test    = tensor2ds((self.Xtest,self.Ytest)).batch(self.local_batchsize)

    def load_lossfn_metrics(self):
        if self.dataset_type == "genomic" or self.dataset_type == "imdb":
            self.loss_fn          = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            self.metrics          =  ['accuracy']
        elif self.dataset_type == "fashion_mnist" or self.dataset_type == "cifar10":
            self.loss_fn          = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            self.metrics          = ['accuracy']
        else:
            print(bcolors.FAIL,"wrong data type specs, could not specify loss function and metric", bcolors.ENDC)
            sys.exit()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def client_update_graph(self,submodel,client,t):
        # if "genomic" in self.dataset_type: submodel=self.stack_encoder(client,submodel)
        print("client_update_graph",client,self.datasets_train[client])
        early_stop  = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        submodel.compile(loss=self.loss_fn,optimizer=self.optimizer,metrics=self.metrics)
        print(bcolors.OKCYAN,"client_update_graph: Worker ",client, " training with batch size", self.local_batchsize,"and loss fn",self.loss_fn.name)
        history = submodel.fit(self.datasets_train[client], epochs=self.local_epochs, validation_data=self.datasets_val[client])
        print("client_update_graph:","Worker ",client, " finished training model ",bcolors.ENDC)
        print('--------------------------------------------------')
        # if "genomic" in self.dataset_type: submodel=self.strip_encoder(submodel)
        self.purge_memory()
        return submodel


    def __limit_memory(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
          # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
          try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=int(7143/self.nclients))])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
          except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    def server_gmcc(self,scaled_local_weight_list,steps=10):
        '''
        apply geometric monte carlo clusteringself.loss_fn
        '''
        model1=self.init_model(t=0,client=23)
        model2=self.init_model(t=0,client=23)
        agg_model_weights,gauged_weights,couple_ids=[],[],[]
        if self.round==1:
            self.clusters=[]
            self.pc_value=5

        #### Find curves ####
        losses=[]
        for id1 in range(self.nclients):
            for id2 in range(id1+1,self.nclients):
                couple_ids.append([id1,id2])
        for id in tqdm(couple_ids):
            model1.set_weights(scaled_local_weight_list[id[0]])
            model2.set_weights(scaled_local_weight_list[id[1]])
            loss,_,gamma=self.find_curve(model1,model2,steps=steps)
            gauged_weights.append(gamma)
            losses.append(np.sum(loss))
            self.purge_memory()

        #### Create Clusters ####
        self.epsilon=np.percentile(losses,min(self.pc_value,95))
        print("PERCENTILE",self.pc_value)
        for id,couple in enumerate(couple_ids):
            if losses[id] < self.epsilon:
                clusters=utils.add2cluster(self.clusters,couple[0],couple[1])
            else:pass
        clusters=utils.add_singlton2cluster(clusters,self.nclients)
        self.clusters=utils.remove_duplicate(clusters)
        utils.printout("Found "+str(len(self.clusters)) + "clusters")
        print(self.clusters)

        #### Assign gammas to clusters ####
        for cluster in self.clusters:
            clustered_weights=[]
            _cluster=cluster
            for idx,couple_id in enumerate(couple_ids):
                if len(cluster)!=1:
                    if couple_id[0] in _cluster and couple_id[1] in _cluster:
                        clustered_weights.append(gauged_weights[idx])
                else :
                    if  couple_id[0] in cluster:
                        clustered_weights.append(scaled_local_weight_list[couple_id[0]])
                        break
                    elif couple_id[1] in cluster:
                        clustered_weights.append(scaled_local_weight_list[couple_id[1]])
                        break
            if len(clustered_weights)>1:
                _agg_model_weights=self.dnn.sum_scaled_weights(clustered_weights)
            else:
                _agg_model_weights=clustered_weights[0]
            agg_model_weights.append(self.dnn.scale_model_weights(_agg_model_weights, 1/len(cluster)))

        if len(self.clusters)!=1:
            self.pc_value+=5
            self.pc_value=min(95,self.pc_value)
        else:
            self.pc_value-=5
            self.pc_value=max(self.pc_value,5)

        return agg_model_weights, self.clusters

    def find_curve(self,model_1,model_2,steps=5):
        '''
        apply geometric monte carlo clustering
        '''
        def gamma_curve(u,theta):
            gamma=[]
            for _theta, _w1, _w2 in zip(theta,w1, w2):
                if u>=0 and u <=0.5: gamma.append(2*(u*_theta+(0.5-u)*_w1 ))
                else:                gamma.append(2*((u-0.5)*_w2 + (1-u)*_theta))
            return gamma
        # @tf.function
        def train_step(rand_inputs,labels,u):
            with tf.GradientTape() as tape:
                predictions = model_gamma(rand_inputs, training=False)
                pred_loss   = loss_fn(labels, predictions)
            gradients   = tape.gradient(pred_loss, model_theta.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model_theta.trainable_variables))
            return pred_loss.numpy()

        loss_fn                 = tf.losses.MeanSquaredError()
        optimizer               = tf.keras.optimizers.Adam(learning_rate=0.001)
        w1,w2                   = model_1.trainable_variables,model_2.trainable_variables
        model_gamma,model_theta = model_1,model_1
        average_weights,theta   = x[],[]
        for num1, num2 in zip(w1, w2): average_weights.append(num1 + num2)
        theta                   = average_weights
        gamma=gamma_curve(0.1,theta)
        u_array                 = np.linspace(0,1,steps)
        np.random.shuffle(u_array)
        labels                  = model_2(self.gmcc_inputs)
        self.mcc_dataset        = tf.data.Dataset.from_tensor_slices((self.gmcc_inputs,labels)).batch(512)
        loss                    = []
        ############ Find theta ############
        for u in u_array:
            model_theta.set_weights(theta)
            gamma=gamma_curve(u,theta)
            model_gamma.set_weights(gamma)
            for input, label in self.mcc_dataset:
                train_step(input,label,u)
            loss.append(loss_fn(labels, model_gamma(self.gmcc_inputs)).numpy())

        loss=np.array(loss)
        order_id=np.argsort(u_array)
        loss=loss[order_id]
        u_array=u_array[order_id]
        min_loss_id=np.argmin(loss[3:-3])+3
        gamma=gamma_curve(u_array[min_loss_id],theta)
        self.purge_memory()
        return loss, u_array, gamma

    def client_update_eager(self,submodel,client,t):
        print(bcolors.OKCYAN,"Worker ",client, " training with batch size", self.local_batchsize)

        @tf.function
        def train_step(inputs,labels):
            with tf.GradientTape() as tape:
                predictions = submodel(inputs, training=True)
                pred_loss   = self.loss_fn(labels, predictions)
            gradients   = tape.gradient(pred_loss, submodel.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, submodel.trainable_weights))


        for local_epoch in tqdm(range(self.local_epochs)):
            for x, y in self.datasets_train[client]:
                train_step(x,y)
        submodel.compile(loss=self.loss_fn,optimizer=self.optimizer,metrics=self.metrics)
        loss,metric = submodel.evaluate(self.Xval[client],self.Yval[client])
        print("Worker ",client, " finished training model --- ValLoss = ",loss,bcolors.ENDC)
        print('--------------------------------------------------')
        # if "genomic" in self.dataset_type: submodel=self.strip_encoder(submodel)
        return submodel

    def save_model_history(self,path,model,round=None):

        if round==None:modelpath = path+'.ckpt'
        else:modelpath = path+"_" +str(round)+ ".ckpt"
        print(bcolors.WARNING,"Saving client model to ", modelpath,bcolors.ENDC)
        model.save(modelpath)
        self.purge_memory()
        del model
        return 1

    def evaluate_model(self,avg_model):
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy()
        avg_model.compile(loss=self.loss_fn,metrics=self.metrics)
        now        = time.time()
        #loss, metric    = avg_model.evaluate(self.datasets_test)
        loss, metric    = avg_model.evaluate(self.Xtest,self.Ytest)
        throughput   = (time.time()-now)/self.Xtest.shape[0]
        return loss, metric, avg_model, throughput

    def purge_memory(self):
        tf.keras.backend.clear_session()

    def init_model(self,t=0,client=0,aggregation=False):
        models=[]
        client=int(client)
        rootpath = os.path.realpath(__file__)
        model = tf.keras.models.clone_model(self.dnn.get_model(client))

        if t==0:
            model._name=model.name+"_"+"t"+str(0)
        else:
            modelfolder = self.checkpointpath
            if self.algorithm=="fedoutavg":
                modelpath   = join(modelfolder,"single_client_model_"+str(client)+".ckpt")
            elif self.algorithm=="fedavg" or  self.algorithm=="fedprox" :
                if aggregation:
                    modelpath   = join(modelfolder,"single_client_model_"+str(client)+".ckpt")
                else:
                    modelpath   = join(modelfolder,"fedavg_model"+".ckpt")
            print("init_model: Loaded ",modelpath)
            model.load_weights(modelpath).expect_partial()
            model.compile(loss=self.loss_fn,optimizer=self.optimizer,metrics=self.metrics)
            print(bcolors.OKGREEN,"init_model: Success opened ",modelpath,bcolors.ENDC)
        return model


    def cluster_aggregation(self,scaled_local_weight_list):
        val_loss_fedgmcc,metric_fedgmcc = [],[]
        utils.printout('Server Model Aggregation for C '+str(self.nclients) + " EMD" + str(self.emd),bcolors.ORANGE)
        ############################## FedAVG/FedPROX ##############################
        utils.printout("FedAvg")
        fedavg_model=self.init_model(t=0,client=0)
        average_weights = self.dnn.sum_scaled_weights(scaled_local_weight_list)
        fedavg_model.set_weights(average_weights)
        _loss, _metric,fedavg_model, throughput = self.evaluate_model(fedavg_model)
        val_loss_fedavg = _loss
        metric_fedavg   = _metric
        n_clusters=len(scaled_local_weight_list)
        fedavg_model.save(join(self.checkpointpath, "fedavg_model.ckpt"))
        del fedavg_model
        self.purge_memory()
        clusters=0;aggregate_weights=0
        ################################# FedGMCC ##############################
        if self.algorithm=="fedgmcc":
            utils.printout("FedGMCC")
            fedgmcc_model=self.init_model(t=0,client=0)
            aggregate_weights,clusters = self.server_gmcc(scaled_local_weight_list)
            n_clusters       = len(clusters)
            for id,_aggregate_weights in enumerate(aggregate_weights):
                fedgmcc_model.set_weights(_aggregate_weights)
                _loss, _metric, fedgmcc_model, throughput = self.evaluate_model(fedgmcc_model)
                val_loss_fedgmcc.append(_loss)
                metric_fedgmcc.append(_metric)
                fedgmcc_model.save(join(self.checkpointpath, "fedgmcc_model_"+str(id)+".ckpt"))
                self.purge_memory()
        else:pass
        evaluations={"val_loss_fedavg":val_loss_fedavg,"val_loss_fedgmcc":val_loss_fedgmcc,"metric_fedgmcc":metric_fedgmcc,\
                     "metric_fedavg":metric_fedavg,"n_clusters":n_clusters,"throughput":throughput}
        return evaluations, average_weights, clusters,aggregate_weights




if __name__=='__main__':
    fl=FederatedLearning()

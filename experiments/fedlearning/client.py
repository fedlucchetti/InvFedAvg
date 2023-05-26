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



class Client(object):

    def __init__(self,opt_dict=None):
        file_path           = os.path.realpath(__file__)
        self.HOMEDIR        = utils.HOMEDIR
        self.DATAPATH       = utils.DATADIR
        self.DATASETDIR     = utils.DATASETDIR
        self.init_model    = None
        if opt_dict==None:
            parser  = argParser.argParser()
            opt_dict = parser.args
        self.options = opt_dict
        if self.options.nclients == 1:
            self.options.emd          = 0.0
            self.options.algorithm    = "sgd"
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
            minval=self.dataset_limits[1], maxval=self.dataset_limits[0])
        elif self.options.dataset_type=="regression":
            inputs_mc = tf.random.uniform(shape=inshape, minval=-1.0, maxval=1.0)
        else:
            inputs_mc = tf.random.uniform(shape=inshape, minval=0.0, maxval=1.0)
        return inputs_mc


    def init_encoder(self,client_id):
        VOCAB_SIZE = 10000
        encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
        encoder.adapt(self.datasets_train[client_id].map(lambda text, label: text))
        self.encoder=encoder
        # self.dutils=encoder
        return 1


    def list_all_emd_partitions(self):
        f,emd_array = [],[]
        path2partitions  = join(self.DATASETDIR,self.options.dataset_type)
        for (dirpath, dirnames, filenames) in os.walk(path2partitions):
            f.extend(filenames)
            break
        for file in f:
            if ".npz" in file and "N"+str(self.options.nclients) in file:
                emd_array.append(np.load(join(self.DATASETDIR,self.options.dataset_type,file))["emd"])
        return emd_array

    def init_data_set(self,datasets=None):
        """
        N:Number of available partitions
        """

        flatten_features=False
        encoded=False
        self.dataset_sizes  = np.zeros(self.options.nclients)
        dataset_limits = np.zeros([self.options.nclients,2])
        self.datasets_train,self.datasets_val = [],[]
        Xtest,Ytest = [],[]
        if self.options.dataset_type=="imdb":encoded=True
        if datasets==None:
            self.dutils.get_dataset(self.options.dataset_type,geolocation="all",
            encoded=encoded,flatten_features=flatten_features)
        for client_id in tqdm(range(self.options.nclients)):
            if datasets==None:
                Xtrain,Ytrain,Xval,Yval = self.dutils.load_data_from_partition(client_id,
                emd=self.options.emd,
                alpha=self.options.alpha,
                ni=self.options.ni,
                nclients=self.options.nclients)
            else:
                Xtrain,Ytrain,Xval,Yval=self.dutils.split_datasets(datasets[client_id][0],datasets[client_id][1])
            self.dataset_sizes[client_id] = Xtrain.shape[0]
            dataset_limits[client_id,:] = [Xtrain.max(),Xtrain.min()]
            self.datasets_train.append(tensor2ds((Xtrain,Ytrain)))
            self.datasets_val.append(tensor2ds((Xval,Yval)))
            Xtest.extend(Xval);Ytest.extend(Yval)
        self.dataset_limits   = [np.max(dataset_limits[:,0]),np.min(dataset_limits[:,1])]
        self.datasets_test    = tensor2ds((np.array(Xtest),np.array(Ytest)))
        self.nclasses         = np.max(Ytrain)+1


    def load_lossfn_metrics(self):
        if self.options.dataset_type == "genomic" or self.options.dataset_type == "imdb":
            self.loss_fn          = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            self.metrics          =  ['accuracy']
        elif self.options.dataset_type == "fashion_mnist" or self.options.dataset_type == "cifar10":
            self.loss_fn          = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            self.metrics          = ['accuracy']
        elif self.options.dataset_type == "regression":
            self.loss_fn          = tf.keras.losses.MeanAbsoluteError()
            self.metrics          = ['accuracy']
        else:
            print(bcolors.FAIL,"wrong data type specs, could not specify loss function and metric", bcolors.ENDC)
            sys.exit()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.options.lr)

    def client_update_graph(self,submodel,client_id,t,return_history=False,verbose=0):
        # if "genomic" in self.options.dataset_type: submodel=self.stack_encoder(client_id,submodel)
        # print("client_update_graph",client_id)
        early_stop  = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.options.lr)
        datasets_train = self.datasets_train[client_id].batch(self.options.local_batchsize)
        datasets_val = self.datasets_val[client_id].batch(self.options.local_batchsize)
        submodel.compile(loss=self.loss_fn,optimizer=self.optimizer,metrics=self.metrics)
        # print(bcolors.OKCYAN,"client_update_graph: Worker ",client_id, " training with batch size", self.options.local_batchsize,"and loss fn",self.loss_fn.name)
        history = submodel.fit(datasets_train,
                               epochs=self.options.local_epochs,
                               validation_data=datasets_val,
                               verbose=verbose,
                               callbacks = early_stop)
        # print("client_update_graph:","Worker ",client_id, " finished training model ",bcolors.ENDC)
        # print('--------------------------------------------------')
        self.purge_memory()
        if return_history:
            return submodel,history.history
        else:
            return submodel

    def update_graph_central(self,submodel,epochs=10):
        early_stop  = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.options.lr)
        submodel.compile(loss=self.loss_fn,optimizer=self.optimizer,metrics=self.metrics)
        Xtrain_central,Ytrain_central = [],[]
        for client_id in tqdm(range(self.options.nclients)):
            Xtrain,Ytrain,_,_ = self.dutils.load_data_from_partition(client_id,
                                                                     emd=self.options.emd,
                                                                     alpha=self.options.alpha)
            Xtrain_central.extend(Xtrain);Ytrain_central.extend(Ytrain)
        Xtrain_central,Ytrain_central = np.array(Xtrain_central),np.array(Ytrain_central)
        print("Xtrain Ytrain shape:",Xtrain_central.shape,Ytrain_central.shape)
        # datasets_train = tensor2ds((Xtrain_central,Ytrain_central)).batch(self.options.local_batchsize)
        history = submodel.fit(x=Xtrain_central,y=Ytrain_central, epochs=epochs,batch_size=self.options.local_batchsize)
        print('--------------------------------------------------')
        self.purge_memory()
        return submodel

    def worker_train(self,client_id,round_id):
        # utils.printout('CLIENT '+str(client_id)+" ROUND "+str(round_id),bcolors.LIGHTGREY)
        print("worker_train: client ",client_id)
        __local_model = self.init_model()
        if round_id==1:
            if not args.flag_common_weight_init:
                pass
            else:
                __local_model.set_weights(aggregate_weights)
        else:
            __local_model.set_weights(aggregate_weights)
        __local_model = self.client_update_graph(__local_model,client_id,round_id,verbose=0)
        __local_model._name    = args.dataset_type+"_"+str(client_id)+"_"+"t"+str(round_id)
        print("worker_train: ",client_id,"FINISH")

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

    def flat_curve(self,u, w1, w2):
        flat=[]
        for  _w1, _w2 in zip(w1, w2):
            flat.append((1-u)*_w1+u*_w2)
        return flat


    def select_most_different_models(self,weights_models,size=1024):
        loss          = list()
        loss_fn       = tf.losses.MeanSquaredError()
        models        = list()
        for weights in weights_models:
            __template = self.init_model()
            __template.set_weights(weights)
            models.append(__template)
        inputs        = self.get_mc_inputs(size=size)
        for i in range(len(models)):
            for j in range(i+1,len(models)):
                labels = models[i](inputs)
                model  = models[j]
                model.compile(loss=loss_fn)
                l = model.evaluate(inputs,labels)
                loss.append([int(i),int(j),l])
        loss=np.array(loss)
        model_1 = models[int(loss[np.argmax(loss[:,2])][0])]
        model_2 = models[int(loss[np.argmax(loss[:,2])][1])]
        return [model_1.get_weights(),model_2.get_weights()]

    def evaluate_model(self,model,client_id=None,verbose=1):
        if client_id==None:
            dataset = self.datasets_test.batch(128)
        elif len(client_id)==2:
            # dataset = self.datasets_test.batch(128)
            combined_dataset = self.datasets_val[client_id[0]].concatenate(self.datasets_val[client_id[1]])
            combined_dataset= combined_dataset.shuffle(buffer_size=32)
            dataset =  combined_dataset.batch(128)
        else:
            dataset =  self.datasets_val[client_id].batch(128)
        model.compile(loss=self.loss_fn,metrics=self.metrics)
        now          = time.time()
        loss, metric = model.evaluate(dataset,verbose=verbose)
        throughput   = (time.time()-now)/self.datasets_test.cardinality().numpy()
        return loss, metric, model, throughput

    def find_curve(self,model_1,model_2,steps=10,n_runs=5,flag_real=False,dataset=None,flag_evaluate=False,client_id=None):
        '''
        find affine connection  between model weights
        '''
        if flag_real:
            loss_fn                 = self.loss_fn
            if dataset==None:
                if client_id!=None and len(client_id)!=2:
                    print("find_curve_: Using preloaded datset")
                    self.mcc_dataset = self.datasets_train[client_id].batch(self.options.cf_batchsize)
                    # self.mcc_dataset = tf.data.Dataset.from_tensor_slices((self.Xtrain[client_id],self.Ytrain[client_id])).batch(512) ## bs = 512
                elif client_id!=None and len(client_id)==2:
                    combined_dataset = self.datasets_train[client_id[0]].concatenate(self.datasets_train[client_id[1]])
                    self.mcc_dataset = combined_dataset.batch(self.options.cf_batchsize)
                elif client_id==None:
                    print("find_curve_: Error client dataset not specified? \n")
                    sys.exit()
            else:
                print("find_curve_: Using supplied datset")
                
                self.inputs_mc,labels = dataset
                self.mcc_dataset = tf.data.Dataset.from_tensor_slices((self.inputs_mc,labels)).batch(128) ## bs = 512
        else:
            print("find_curve_: Using MC datset")
            inputs_mc,labels = self.generate_mc_dataset(model_2)
            # inputs_mc        = self.get_mc_inputs()
            # labels           = model_2(inputs_mc,training=False)
            # labels           = np.argmax(labels,axis=1)
            # labels           = tf.keras.layers.BatchNormalization()(labels)
            self.mcc_dataset = tf.data.Dataset.from_tensor_slices((inputs_mc,labels)).batch(self.options.cf_batchsize) ## bs = 512
            loss_fn                 = self.loss_fn

        @tf.function
        def train_step(inputs,labels,scale):
            with tf.GradientTape() as tape:
                predictions   = model_gamma(inputs, training=True)
                loss          = loss_fn(labels, predictions)*scale
            gradients   = tape.gradient(loss, model_gamma.trainable_variables)
            # optimizer.apply_gradients(zip(gradients, model_theta.trainable_weights))
            # optimizer.apply_gradients(zip(gradients, model_gamma.trainable_weights))
            return gradients,loss

        
        optimizer               = tf.keras.optimizers.Adam(learning_rate=self.options.cf_lr)
        # optimizer             = tf.keras.optimizers.SGD(learning_rate=self.options.cf_lr)
        model_1.trainable=False;model_2.trainable=False
        w1,w2                   = model_1.get_weights(),model_2.get_weights()
        model_gamma             = self.init_model()
        model_theta             = self.init_model()
        w1_scaled               = utils.scale_model_weights(model_1.get_weights(), 0.5)
        w2_scaled               = utils.scale_model_weights(model_2.get_weights(), 0.5)
        model_theta.set_weights(utils.sum_scaled_weights([w1_scaled,w2_scaled]))
        model_theta.trainable   = True
        u_array                 = np.linspace(0,1,steps)
        # np.random.shuffle(u_array)
        loss_train_history      = np.zeros([n_runs,u_array.size])
        loss_val_history      = np.zeros([n_runs,u_array.size])
        metric_val_history    = np.zeros([n_runs,u_array.size])
        model_gamma.set_weights(utils.gamma_curve(0.5,model_theta.trainable_variables, w1, w2,mode=self.options.cf_curve))
        loss_val_history[0,:],metric_val_history[0,:] = self.evaluate_gammas(model_theta.get_weights(),
                                                                                 w1, w2,client_id,verbose=0,
                                                                                 steps=steps)
        loss_val_history[:,0],loss_val_history[:,-1]     = loss_val_history[0,0],loss_val_history[0,-1]
        metric_val_history[:,0],metric_val_history[:,-1] = metric_val_history[0,0],metric_val_history[0,-1]
        ############ Find theta ############
        for run in tqdm(range(1,n_runs)):
            for id_batch, (_inputs,_labels) in enumerate(tqdm(self.mcc_dataset)):
                gradients_t = list()
                for id,u in enumerate(u_array):
                    if u==0 or u ==1: continue
                    model_gamma.set_weights(utils.gamma_curve(u,model_theta.trainable_variables, w1, w2,mode=self.options.cf_curve))
                    dtheta = utils.jacobian_dgamma(u,mode=self.options.cf_curve)
                    scale = tf.constant(dtheta/u_array.size,dtype=tf.dtypes.float32)
                    g,loss_train_history[run-1,id] = train_step(_inputs,_labels,scale=scale)
                    gradients_t.append(g)
                gradients = utils.sum_scaled_weights(gradients_t)
                optimizer.apply_gradients(zip(gradients, model_theta.trainable_weights))
            if self.options.cf_verbose:
                loss_val_history[run,:],metric_val_history[run,:] = self.evaluate_gammas(model_theta.get_weights(),
                                                                                         w1, w2,client_id=None,verbose=0,
                                                                                         steps=steps)
            print("### Curve finding loss ###")
            for id_run in range(run):
                print(bcolors.OKBLUE,"LOSS SYM: round",id_run,np.around(loss_train_history[id_run,np.argsort(u_array)], decimals=3))
                print(bcolors.ORANGE,"LOSS VAL: round",id_run,np.around(loss_val_history[id_run,:], decimals=3))
                print(bcolors.ORANGE,"ACCU VAL: round",id_run,np.around(metric_val_history[id_run,:], decimals=3))
                print(bcolors.ENDC)
            # if metric_val_history[id_run,:].sum()<

        idc_sort = np.argsort(u_array)
        self.purge_memory()
        theta = model_theta.trainable_variables
        return theta, loss_train_history[:,idc_sort], np.sort(u_array), loss_val_history, metric_val_history

    def evaluate_gammas(self,weights_theta,w1, w2,client_id=None,verbose=1,steps=7):
        model_gamma = self.init_model()
        u_array     = np.linspace(0,1,steps)
        loss, metric = np.zeros(steps),np.zeros(steps),
        for id,u in enumerate(tqdm(u_array)):
            model_gamma.set_weights(utils.gamma_curve(u,weights_theta, w1, w2,mode=self.options.cf_curve))
            loss[id], metric[id], model, throughput = self.evaluate_model(model_gamma,client_id,verbose=verbose)
        return loss, metric

    def get_mc_inputs(self):
        inshape         = [self.options.mc_size]
        _template_model = self.init_model()
        inshape.extend(_template_model.input.shape.as_list()[1::])
        if self.options.dataset_type=="regression":
            inputs_mc = tf.random.uniform(shape=inshape, minval=-1.0, maxval=1.0)
        else:
            inputs_mc = tf.random.uniform(shape=inshape, minval=0.0, maxval=1.0)
        return inputs_mc
    
    def generate_mc_dataset(self,model,size=None):
        if size==None:
            nsamples = round(self.options.mc_size/self.nclasses)
        else:
            nsamples = round(size/self.nclasses)
        print(nsamples)
        Nymax = np.ones(self.nclasses)*nsamples
        Ny    = np.zeros(self.nclasses)
        inputshape      = [nsamples*self.nclasses]
        sampleshape     = [512]
        inputshape.extend(model.input.shape.as_list()[1::])
        sampleshape.extend(model.input.shape.as_list()[1::])
        inputs_mc = np.zeros(inputshape)
        labels_mc = np.zeros(inputs_mc.shape[0])
        if self.options.dataset_type=="regression":
            minval,maxval = -1.0,1.0
        else:
            minval,maxval = 0.0,1.0
        pbar = tqdm(total=Nymax.sum())
        while Ny.sum() < Nymax.sum():
            flag_continue = False
            for tryid in range(23):
                input = tf.random.uniform(shape=inputshape, minval=minval, maxval=maxval)
                ypred_arr = np.argmax(model(input,training=False),axis=1)
                print(np.histogram(ypred_arr,bins=self.nclasses)[0])
                return np.histogram(ypred_arr,bins=self.nclasses)[0]
                if 0 not in np.histogram(ypred_arr,bins=self.nclasses)[0]:
                    flag_continue = True
                    break
            if flag_continue:
                for idy,ypred in enumerate(ypred_arr):
                    if Ny[ypred] < Nymax[ypred]:
                        inputs_mc[int(Ny.sum())] = input[idy]
                        labels_mc[int(Ny.sum())] = ypred
                        Ny[ypred]+=1
                        pbar.update(1)
            else: return None, None
        return inputs_mc,labels_mc



    def purge_memory(self):
        tf.keras.backend.clear_session()

if __name__=='__main__':
    fl=FederatedLearning()

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
        print(opt_dict)
        if opt_dict!=None:
            print("Initializing FL neural net Class with following parameters")
            print(opt_dict)
        else:
            parser  = argParser.argParser()
            opt_dict = parser.parse(None)
            print("Invalid settings. Exit")
            print(opt_dict)
            # sys.exit()
        self.dataset_type     = opt_dict.dataset
        self.nclients         = opt_dict.nclients
        self.local_batchsize  = opt_dict.batchsize
        self.local_epochs     = opt_dict.local_epochs
        self.learning_rate    = opt_dict.nu
        self.n_montecarlo     = int(opt_dict.mc_size)
        self.cf_lr            = opt_dict.cf_lr
        self.cf_verbose       = opt_dict.cf_verbose
        if self.nclients == 1:
            self.emd          = 0.0
            self.algorithm    = "sgd"
        else:
            self.emd          = opt_dict.emd
            self.algorithm    = opt_dict.algorithm
        self.round            = 1
        self.dnn              = dnn.DNN(self)
        self.dutils           = datautils.DataUtils(self)

    def init_models(self,t,clients,aggregation=False):
        models=[]
        for client in clients:
            models.append(self.init_model(t=t,client=client,aggregation=aggregation))
        return models


    def get_mc_inputs(self,size=None):
        if size==None:
            size=self.n_montecarlo
        inshape     = [size]
        _template_model = self.init_model()
        inshape.extend(_template_model.input.shape.as_list()[1::])
        if self.dataset_type=="genomic":
            inputs_mc = tf.random.uniform(shape=[self.n_montecarlo,30,64],
            minval=self.Xtrain.min(), maxval=self.Xtrain.max())
        elif self.dataset_type=="regression":
            inputs_mc = tf.random.uniform(shape=inshape, minval=-1.0, maxval=1.0)
        else:
            inputs_mc = tf.random.uniform(shape=inshape, minval=0.0, maxval=1.0)
        return inputs_mc

    def init_gmcc(self):
        self.inputs_mc = self.get_mc_inputs()

    def init_encoder(self,client):
        VOCAB_SIZE = 10000
        encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
        if client!=23:
            encoder.adapt(self.datasets_train[0].map(lambda text, label: text))
        else:
            datasets_train = tensor2ds((self.Xtrain[0][0:2],self.Ytrain[0][0:2]))
            encoder.adapt(datasets_train.map(lambda text, label: text))
        self.encoder=encoder
        # self.dutils=encoder
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

    def init_data_set(self,datasets=None):
        """
        N:Number of available partitions
        """

        flatten_features=False
        encoded=False
        self.Xtrain,self.Ytrain,self.Xval,self.Yval,self.Xtest,self.Ytest = [],[],[],[],[],[]
        self.datasets_train,self.datasets_val,self.datasets_test = [],[],[]
        if self.dataset_type=="imdb":encoded=True
        if datasets==None:
            self.dutils.get_dataset(self.dataset_type,geolocation="all",encoded=encoded,flatten_features=flatten_features)
        for client in tqdm(range(self.nclients)):
            if datasets==None:
                Xtrain,Ytrain,Xval,Yval = self.dutils.load_data_from_partition(client,emd=self.emd)
            else:
                Xtrain,Ytrain,Xval,Yval=self.dutils.split_datasets(datasets[client][0],datasets[client][1])
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
        elif self.dataset_type == "regression":
            self.loss_fn          = tf.keras.losses.MeanAbsoluteError()
            self.metrics          = ['accuracy']
        else:
            print(bcolors.FAIL,"wrong data type specs, could not specify loss function and metric", bcolors.ENDC)
            sys.exit()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def client_update_graph(self,submodel,client,t,return_history=False):
        # if "genomic" in self.dataset_type: submodel=self.stack_encoder(client,submodel)
        # print("client_update_graph",client)
        early_stop  = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        submodel.compile(loss=self.loss_fn,optimizer=self.optimizer,metrics=self.metrics)
        print(bcolors.OKCYAN,"client_update_graph: Worker ",client, " training with batch size", self.local_batchsize,"and loss fn",self.loss_fn.name)
        history = submodel.fit(self.datasets_train[client], epochs=self.local_epochs, validation_data=self.datasets_val[client])
        # print("client_update_graph:","Worker ",client, " finished training model ",bcolors.ENDC)
        # print('--------------------------------------------------')
        self.purge_memory()
        if return_history:
            return submodel,history.history
        else:
            return submodel

    def update_graph_central(self,submodel,epochs=10):
        early_stop  = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        submodel.compile(loss=self.loss_fn,optimizer=self.optimizer,metrics=self.metrics)
        Xtrain_central,Ytrain_central = [],[]
        for client in tqdm(range(self.nclients)):
            Xtrain,Ytrain,_,_ = self.dutils.load_data_from_partition(client,emd=self.emd)
            Xtrain_central.extend(Xtrain);Ytrain_central.extend(Ytrain)
        Xtrain_central,Ytrain_central = np.array(Xtrain_central),np.array(Ytrain_central)
        print("Xtrain Ytrain shape:",Xtrain_central.shape,Ytrain_central.shape)
        # datasets_train = tensor2ds((Xtrain_central,Ytrain_central)).batch(self.local_batchsize)
        history = submodel.fit(x=Xtrain_central,y=Ytrain_central, epochs=epochs,batch_size=self.local_batchsize)
        print('--------------------------------------------------')
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

    def server_covfedavg(self,scaled_local_weight_list,steps=10):
        '''
        apply geometric monte carlo clusteringself.loss_fn
        '''
        model1=self.init_model()
        model2=self.init_model()
        agg_model_weights=[]

        id_client_ref = np.random.randint(len(scaled_local_weight_list))
        model1.set_weights(scaled_local_weight_list[id_client_ref])

        gauged_model_weights=[scaled_local_weight_list[id_client_ref]]
        agg_model_weights=[]
        for id_client in tqdm(range(len(scaled_local_weight_list))):
            if id_client==id_client_ref: continue
            model2.set_weights(scaled_local_weight_list[id_client])
            theta, _  = self.find_curve(model1,model2,steps=steps)
            w1,w2     = model1.trainable_variables,model2.trainable_variables
            gamma_avg = self.gamma_average_curve(theta,w1,w2,steps=10)
            gauged_model_weights.append(gamma_avg)
            self.purge_memory()
        agg_model_weights = self.dnn.sum_scaled_weights(gauged_model_weights)
        return agg_model_weights

    def gamma_average_curve(self,theta,w1,w2,steps=10):
        gamma_list=[]
        for u in np.linspace(0,1,steps):
            gamma = self.gamma_curve(u,theta, w1, w2)
            gamma = self.dnn.scale_model_weights(gamma, 1/steps)
            gamma_list.append(gamma)
        gamma_average = self.dnn.sum_scaled_weights(gamma_list)
        return gamma_average



    def gamma_curve(self,u,theta, w1, w2,mode="chain"):
        gamma=[]
        if mode=="chain":
            for _theta, _w1, _w2 in zip(theta,w1, w2):
                if u>=0 and u <=0.5: gamma.append(2*(u*_theta+(0.5-u)*_w1 ))
                else:                gamma.append(2*((u-0.5)*_w2 + (1-u)*_theta))
        elif mode=="bezier":
            for _theta, _w1, _w2 in zip(theta, w1,w2):
                gamma.append( (1-u)**2 * _w1 + 2*u*(1-u)*_theta + u**2*_w2 )
        elif mode=="bezier2":
            for _theta, _w1, _w2 in zip(theta, w1,w2):
                gamma.append( (1-u)**3 * _w1 + 3*u*(1-u)**2*_theta1 + 3*(1-u)*u**2 * _theta2 + u**3*_w2 )
        return gamma

    def jacobian_dgamma(self,u,mode):
        if mode=="chain":
            if u>=0 and u <=0.5: return u
            else:                return 1-u
        elif mode=="bezier":
            return u*(1-u)


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


    def find_curve(self,model_1,model_2,steps=10,n_runs=5,flag_real=False,dataset=None,flag_evaluate=False,client=None):
        '''
        find affine connection  between model weights
        '''
        if flag_real:
            if dataset==None:
                if client!=None:
                    print("find_curve_2: Using preloaded datset")
                    self.mcc_dataset = tf.data.Dataset.from_tensor_slices((self.Xtrain[client],self.Ytrain[client])).batch(512) ## bs = 512
                else:
                    print("find_curve_2: Error client dataset not specified? \n")
                    sys.exit()
            else:
                print("find_curve_2: Using supplied datset")
                self.inputs_mc,labels = dataset
                self.mcc_dataset = tf.data.Dataset.from_tensor_slices((self.inputs_mc,labels)).batch(self.local_batchsize) ## bs = 512
        else:
            print("find_curve_2: Using MC datset")
            labels           = model_2(self.inputs_mc,training=False)
            labels           = tf.keras.layers.BatchNormalization()(labels)
            self.mcc_dataset = tf.data.Dataset.from_tensor_slices((self.inputs_mc,labels)).batch(self.local_batchsize) ## bs = 512

        @tf.function
        def train_step(inputs,labels):
            with tf.GradientTape() as tape:
                predictions   = model_gamma(inputs, training=True)
                pred_loss     = loss_fn(labels, predictions)
            gradients   = tape.gradient(pred_loss, model_gamma.trainable_variables)
            return gradients,pred_loss

        loss_fn                 = tf.losses.MeanSquaredError()
        optimizer               = tf.keras.optimizers.Adam(learning_rate=self.cf_lr)
        # optimizer               = tf.keras.optimizers.SGD(learning_rate=self.cf_lr)
        model_1.trainable=False;model_2.trainable=False
        w1,w2                   = model_1.get_weights(),model_2.get_weights()
        model_gamma             = self.init_model()
        model_theta             = self.init_model()
        w1_scaled               = self.dnn.scale_model_weights(model_1.get_weights(), 0.5)
        w2_scaled               = self.dnn.scale_model_weights(model_2.get_weights(), 0.5)
        model_theta.set_weights(self.dnn.sum_scaled_weights([w1_scaled,w2_scaled]))
        model_theta.trainable   = True
        u_array                 = np.linspace(0.05,1,steps)
        np.random.shuffle(u_array)
        loss_curve              = np.zeros([n_runs,u_array.size])
        loss_gamma_history      = np.zeros([n_runs,u_array.size])
        metric_gamma_history    = np.zeros([n_runs,u_array.size])
        model_gamma.set_weights(self.gamma_curve(0.5,model_theta.trainable_variables, w1, w2,mode="bezier"))
        self.evaluate_gammas(model_theta.get_weights(),w1, w2)
        ############ Find theta ############
        for run in tqdm(range(n_runs)):
            for id_batch, (_inputs,_labels) in enumerate(tqdm(self.mcc_dataset)):
                gradients_t             = list()
                for id,u in enumerate(u_array):
                    model_gamma.set_weights(self.gamma_curve(u,model_theta.trainable_variables, w1, w2,mode="bezier"))
                    g,loss_curve[run,id] = train_step(_inputs,_labels)
                    dtheta = self.jacobian_dgamma(u,mode="bezier")
                    gradients_t.append(self.dnn.scale_model_weights(g, dtheta/u_array.size))
                gradients = self.dnn.sum_scaled_weights(gradients_t)
                optimizer.apply_gradients(zip(gradients, model_theta.trainable_weights))
            print("### Curve finding loss ###")
            for id_run in range(run+1):
                print("round",id_run,np.around(loss_curve[id_run,:], decimals=1))
                # print("round",id,"stat curve loss",round(loss_curve[id,:].mean(),1),"+-",round(loss_curve[id,:].std(),1),
                #                  "sum val   loss",round(loss_gamma_history[id,:].sum(),1))
            if self.cf_verbose:
                self.evaluate_gammas(model_theta.get_weights(),w1, w2)
        idc_sort = np.argsort(u_array)
        del model_gamma
        self.purge_memory()
        return model_theta.trainable_variables, loss_curve[:,idc_sort], np.sort(u_array), loss_gamma_history[:,idc_sort], metric_gamma_history[:,idc_sort]




    def evaluate_gammas(self,weights_theta,w1, w2):
        model_gamma=self.init_model()
        for id,u in enumerate(np.linspace(0,1,7)):
            model_gamma.set_weights(self.gamma_curve(u,weights_theta, w1, w2,mode="bezier"))
            self.evaluate_model(model_gamma)



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


    def model_interaction(self,gammas):
        weights_1_A,weights_1_B,weights_1_theta = gammas[0]
        weights_2_A,weights_2_B,weights_2_theta = gammas[1]
        loss                    = list()
        u_array                 = np.linspace(0,1,10)
        loss_fn                 = tf.losses.MeanSquaredError()
        inputs_MC               = self.get_mc_inputs(size=1024)
        model_1_inv,model_2_inv = self.init_model(),self.init_model()
        model_1_inv.trainable=False;model_2_inv.trainable=False
        for i,ti in enumerate(tqdm(u_array)):
            for j,tj in enumerate(u_array):
                model_1_inv.set_weights(self.gamma_curve(ti,weights_1_theta, weights_1_A, weights_1_B,mode="bezier"))
                model_2_inv.set_weights(self.gamma_curve(tj,weights_2_theta, weights_2_A, weights_2_B,mode="bezier"))
                l = loss_fn(model_1_inv(inputs_MC),model_2_inv(inputs_MC)).numpy()
                loss.append([int(i),int(j),l])
        loss=np.array(loss)
        t1_opt = u_array[int(loss[np.argmin(loss[:,2])][0])]
        t2_opt = u_array[int(loss[np.argmin(loss[:,2])][1])]
        print("\n model_interaction: correcting model 1 at:",t1_opt,"witch Lint=",np.min(loss[:,2]))
        print("   model_interaction: correcting model 2 at:",t2_opt,"witch Lint=",np.min(loss[:,2]),"\n")
        model_1_inv.set_weights(self.gamma_curve(t1_opt,weights_1_theta, weights_1_A, weights_1_B,mode="bezier"))
        model_2_inv.set_weights(self.gamma_curve(t2_opt,weights_2_theta, weights_2_A, weights_2_B,mode="bezier"))

        return model_1_inv,model_2_inv

    def client_find_symmetric(submodel,client,alpha1=0.1,alpha2=0.001,c=4):
        models=[submodel]
        loss_fn                 = tf.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        submodel.compile(loss=loss_fn,optimizer=optimizer)

        @tf.function
        def train_step(inputs,labels):
            with tf.GradientTape() as tape:
                predictions = submodel(inputs, training=True)
                pred_loss   = lr*self.loss_fn(labels, predictions)
            gradients   = tape.gradient(pred_loss, submodel.trainable_variables)
            optimizer.apply_gradients(zip(gradients, submodel.trainable_weights))


        for local_epoch in tqdm(range(self.local_epochs)):
            for iter, (x, y) in enumerate(self.datasets_train[client]):
                ti = 1/c*( (iter)%c + 1)
                if ti<=0.5: lr = (1-2*ti)*alpha1 + 2*ti*alpha2
                else:       lr = (2-2*ti)*alpha2 + (2*ti-1)*alpha1
                optimizer.learning_rate = lr
                train_step(x,y)
                if ti==1/2:
                    models.append(submodel)
                    submodel.evaluate(self.Xval[client],self.Yval[client])

        return models

    def save_model_history(self,path,model,round=None):

        if round==None:modelpath = path+'.ckpt'
        else:modelpath = path+"_" +str(round)+ ".ckpt"
        print(bcolors.WARNING,"Saving client model to ", modelpath,bcolors.ENDC)
        model.save(modelpath)
        self.purge_memory()
        del model
        return 1

    def evaluate_model(self,avg_model):
        avg_model.compile(loss=self.loss_fn,metrics=self.metrics)
        now          = time.time()
        loss, metric = avg_model.evaluate(self.Xtest,self.Ytest)
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


    def aggregation(self,scaled_local_weight_list):
        val_loss_covfedavg,metric_covfedavg = [],[]
        utils.printout('Server Model Aggregation for C '+str(self.nclients) + " EMD" + str(self.emd),bcolors.ORANGE)
        ############################## INIT aggregation ##############################
        aggregate_weights=0
        agg_model=self.init_model(t=0,client=2323)

        ################################# covfedavg ##############################
        if self.algorithm=="covfedavg":
            utils.printout("CovFedAvg")
            aggregate_weights = self.server_covfedavg(scaled_local_weight_list)
            print("fl.aggregation: len(aggregate_weights)",len(aggregate_weights))
            agg_model.set_weights(aggregate_weights)
            agg_model.save(join(self.checkpointpath, "covfedavg_model.ckpt"))

        ################################# FedAVG ##############################
        elif self.algorithm=="fedavg":
            utils.printout("FedAvg")
            print("fl.aggregation: len(scaled_local_weight_list)",len(scaled_local_weight_list))
            aggregate_weights        = self.dnn.sum_scaled_weights(scaled_local_weight_list)
            print("fl.aggregation: len(aggregate_weights)",len(aggregate_weights))
            agg_model.set_weights(aggregate_weights)
            agg_model.save(join(self.checkpointpath, "fedavg_model.ckpt"))

        _loss, _metric,agg_model, throughput = self.evaluate_model(agg_model)
        del agg_model
        self.purge_memory()

        evaluations={"val_loss_fl":_loss,\
                     "metric_fl":_metric,"throughput":throughput}
        return evaluations,aggregate_weights




if __name__=='__main__':
    fl=FederatedLearning()

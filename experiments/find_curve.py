from fedlearning import  bcolors, utils, argParser,dnn, server, client,datautils
import sys,time, glob, json, os,  time,  copy, platform
import tensorflow as tf
from tensorflow import keras
import numpy as np
from os.path import join, split
import pandas as pd

#######################################################################################################################
bcolors = bcolors.bcolors()
parser  = argParser.argParser()
utils   = utils.Utils()
args    = parser.args
args.alpha = 0.1
args.ni = 0.42
args.local_epochs =  15
args.mc_size = 512
client  = client.Client(args)
server  = server.Server(args)
client.init_model = server.init_model
dutils  = datautils.DataUtils(client)
dnn     = dnn.DNN(client)
#######################################################################################################################
utils.printout("GPU available= " + str(tf.config.experimental.list_physical_devices('GPU')),bcolors.OKCYAN)
client.init_data_set()
client.load_lossfn_metrics()
x1,y1 = dutils.dataset_to_numpy(client.datasets_train[0])
x2,y2 = dutils.dataset_to_numpy(client.datasets_train[1])
x1_val,y1_val = dutils.dataset_to_numpy(client.datasets_val[0])
x2_val,y2_val = dutils.dataset_to_numpy(client.datasets_val[1])
x     = np.concatenate([x1,x2])
y     = np.concatenate([y1,y2])
x_val = np.concatenate([x1_val,x2_val])
y_val = np.concatenate([y1_val,y2_val])
early_stop  = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
model = client.init_model()
def get_dataset(classes=[]):
    inputs,labels         = x[0:1],y[0:1]
    inputs_val,labels_val = x_val[0:1],y_val[0:1]
    for class_id in classes: 
        sel_class_idc = np.where(y==class_id)[0]
        inputs = np.concatenate([inputs,x[sel_class_idc]])
        labels = np.concatenate([labels,y[sel_class_idc]])
        sel_class_idc = np.where(y_val==class_id)[0]
        inputs_val = np.concatenate([inputs_val,x_val[sel_class_idc]])
        labels_val = np.concatenate([labels_val,y_val[sel_class_idc]])
    print(inputs.shape,inputs_val.shape)
    client.datasets_train[0] = dutils.numpy_to_dataset(inputs[2::],labels[2::]).shuffle(inputs.shape[0])
    client.datasets_val[0]   = dutils.numpy_to_dataset(inputs_val[2::],labels_val[2::]).shuffle(inputs_val.shape[0])

a,b = dutils.dataset_to_numpy(client.datasets_train[0])
get_dataset(classes=[0,1,2,3,4,5,6,7,8,9])

mcsize = 2**14
model = client.init_model()
client.options.local_epochs=10
model,h  = client.client_update_graph(model,0,1,return_history=True,verbose=1)
py = client.generate_mc_dataset(model,size=mcsize)


print(np.abs(py-mcsize/client.nclasses*np.ones(client.nclasses)).mean())


dataset = dutils.numpy_to_dataset(_x,_y) 


weights_local_models=list()
for client_id, idc in enumerate(range(0,args.nclients)):
    __local_model  = server.init_model(client_id=client_id)
    __local_model,h  = client.client_update_graph(__local_model,client_id,
                                                  1,return_history=True,
                                                  verbose=1)
    weights_local_models.append(__local_model.get_weights())

x,y = client.generate_mc_dataset(__local_model,size=2*4096)
print("#### find_curve ### \n")
gammas = list()
for client_id_1 in range(0,args.nclients):
    for client_id_2 in range(client_id_1+1,args.nclients):
        client_ids = [client_id_1,client_id_2]
        model_A = weights_local_models[client_id_1]
        model_B = weights_local_models[client_id_2]
        args.cf_verbose=1
        args.cf_nruns=30
        weights_theta, loss_curve, nu_array,loss_gamma_history,metric_gamma_history = client.find_curve(model_A,model_B,
                                                                                flag_real=True,client_id=client_ids,
                                                                                steps=7,n_runs=args.cf_nruns)
    gammas.append([model_A,model_B,weights_theta])

    if len(gammas)!=0:
        ####################################
        utils.printout("Server: Find curve intersection",bcolors.HEADER)
        # model_1_inv,model_2_inv = server.model_interaction(gammas)
        agg_model_weights = server.covfedavg(gammas,steps=10)
        ####################################
        utils.printout("Train Central Model",bcolors.HEADER)
        model_central      = server.init_model()
        # if args.flag_common_weight_init:
        #     model_central.set_weights(weights_common_init)
        # if args.flag_preload_models:
        #     model_central = keras.models.load_model(join(CKPTPATH,"model_central"))
        # else:
        #     model_central      = client.update_graph_central(model_central,epochs=10)
        #     model_central.save(join(CKPTPATH,"model_central"))
        loss_central, metric_central,_,_    = client.evaluate_model(model_central,verbose=0)
        ####################################
        print("\n###### INV models ######\n")
        loss_inv, metric_inv = np.zeros(len(agg_model_weights)),np.zeros(len(agg_model_weights))
        for idx,weights in enumerate(agg_model_weights):
            model_inv = server.init_model()
            model_inv.set_weights(weights)
            loss_inv[idx], metric_inv[idx],_,_        = client.evaluate_model(model_inv,verbose=1)
        # loss_inv_2, metric_inv_2,_,_        = client.evaluate_model(model_2_inv,verbose=0)
        # print("\n###### INV AVG models ######\n")
        # model_inv_avg  = server.init_model()
        #     local_weight_list=[]
        # for client_id in range(client.options.nclients):
        #     local_weight_list.append(gammas[client_id][0])
        # model_inv_avg.set_weights(server.fedavg([model_1_inv.get_weights(),model_2_inv.get_weights()],[0.5,0.5]))
        # loss_invfedavg, metric_invfedavg,_,_  = client.evaluate_model(model_inv_avg,verbose=0)
        # print("\n###### local Client models ######\n")
        # model_1_local,model_2_local         = server.init_model(),server.init_model()
        # model_1_local.set_weights(gammas[0][0]);model_2_local.set_weights(gammas[1][0])
        # loss_local_1, metric_local_1,_,_    = client.evaluate_model(model_1_local,verbose=0)
        # loss_local_2, metric_local_2,_,_    = client.evaluate_model(model_2_local,verbose=0)
        print("\n###### FedAvg ######\n")
        local_weight_list=[]
        for client_id in range(len(gammas)):
            local_weight_list.append(gammas[client_id][0])
        fedavg  = server.init_model()
        fedavg.set_weights(server.fedavg(local_weight_list,client.dataset_sizes))
        loss_fedavg, metric_fedavg,_,_  = client.evaluate_model(fedavg,verbose=0)

        # table = [[round(loss_central,3)  , round(metric_central,3)],
        #          [round(loss_inv,3)    , round(metric_inv,3)],
        #          [round(loss_inv_2,3)    , round(metric_inv_2,3)],
        #          [round(loss_local_1,3)  , round(metric_local_1,3)],
        #          [round(loss_local_2,3)  , round(metric_local_2,3)],
        #          [round(loss_invfedavg,3), round(metric_invfedavg,3)],
        #          [round(loss_fedavg,3)   , round(metric_fedavg,3)]]
        # df = pd.DataFrame(table,
        # index = ['Central', 'Inv 1', 'Inv 2', 'Client 1','Client 2','InvFedAvg','FedAvg'],
        #             columns=['Loss', 'Accuracy'])
        # print(df)


        table = [[round(loss_central,3)  , round(metric_central,3)],
                [round(loss_inv.mean(),3)    , round(metric_inv.mean(),3)],
                [round(loss_fedavg,3)   , round(metric_fedavg,3)]]
        df = pd.DataFrame(table,
        index = ['Central', 'Inv', 'FedAvg'],
                    columns=['Loss', 'Accuracy'])
        print(df)


        server.purge_memory()
except KeyboardInterrupt:
    server.purge_memory()
    sys.exit(0)




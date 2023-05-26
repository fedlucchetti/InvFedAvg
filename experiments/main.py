from fedlearning import  bcolors, utils, argParser,dnn, server, client,datautils
import sys,time, glob, json, os,  time,  copy
import tensorflow as tf
from tensorflow import keras
import numpy as np
from os.path import join, split
import pandas as pd
import platform
from tqdm import tqdm
#######################################################################################################################
bcolors = bcolors.bcolors()
parser  = argParser.argParser()
utils   = utils.Utils()
args    = parser.args
client  = client.Client(args)
server  = server.Server(args)
client.init_model = server.init_model
# server.curve = client.
dutils  = datautils.DataUtils(client)
dnn     = dnn.DNN(client)
#######################################################################################################################
utils.printout("GPU available= " + str(tf.config.experimental.list_physical_devices('GPU')),bcolors.OKCYAN)
client.init_data_set()
dataset_sizes=np.zeros(args.nclients)
for client_id in range(args.nclients):
    dataset_sizes[client_id]=client.datasets_train[client_id].cardinality().numpy()
print(dataset_sizes)
client.load_lossfn_metrics()
weights_common_init = server.init_model().get_weights()
#######################################################################################################################
OUTFOLDER         = join(utils.HOMEDIR,"results",
                         args.algorithm,
                         args.dataset_type,
                         str(args.nclients),
                         platform.node())
utils.create_dir(OUTFOLDER)
id_new            = utils.create_run_id(OUTFOLDER)
OUTFOLDER         = join(OUTFOLDER,"run_"+str(id_new))
utils.printout("Starting " + args.algorithm + " on dataset "+ args.dataset_type + " RunID "+str(id_new))
utils.create_dir(OUTFOLDER)
RESULTSPATH       = join(OUTFOLDER,'results.npz')
client.checkpointpath = join(OUTFOLDER,"checkpoint")
utils.create_dir(client.checkpointpath)
for key,value in vars(args).items():
    print(key,"\t",value)
with open(join(OUTFOLDER,'settings.json'), 'w') as outfile:
    json.dump(vars(args), outfile)

############ Init template model ############
__template = server.init_model()
aggregate_weights = __template.get_weights()
############ Init result arrays ############
val_loss_fl_array,metric_fl_array=[],[]
########################################################################################################################
##################################### Training Routine #################################################################
########################################################################################################################
start      = time.time()
try:
    for round_id in np.arange(1,args.rounds+1):
        utils.printout("ROUND "+ str(round_id)+"/"+str(args.rounds),bcolors.HEADER)
        local_weight_list = list()
        for client_id, idc in enumerate(tqdm(range(0,args.nclients))):
            # utils.printout('CLIENT '+str(client_id)+" ROUND "+str(round_id),bcolors.LIGHTGREY)
            if round_id==1:
                if not args.flag_common_weight_init:
                    __local_model = server.init_model()
                else:
                    __local_model.set_weights(aggregate_weights)
            else:
                __local_model = server.init_model()
                __local_model.set_weights(aggregate_weights)
            __local_model = client.client_update_graph(__local_model,client_id,round_id,verbose=0)
            __local_model._name    = args.dataset_type+"_"+str(client_id)+"_"+"t"+str(round_id)
            local_weight_list.append(__local_model.get_weights())
########################################################################################################################
##################################### Server Aggregation ###############################################################
########################################################################################################################
        aggregate_weights=server.aggregation(local_weight_list,dataset_sizes)
        __local_model.set_weights(aggregate_weights)
        _loss, _metric,agg_model, throughput = client.evaluate_model(__local_model)
        del __local_model
        client.purge_memory()
        val_loss_fl_array.append(_loss)
        metric_fl_array.append(_metric)
        np.savez(RESULTSPATH,
                    metric      = np.array(metric_fl_array),
                    loss_val    = np.array(val_loss_fl_array),
                    train_time  = time.time()-start,
                    throughput  = throughput)
        print("Saved results to ",RESULTSPATH)
    client.purge_memory()
except KeyboardInterrupt:
    client.purge_memory()
    sys.exit()

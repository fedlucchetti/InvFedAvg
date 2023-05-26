from fedlearning import federatedlearning, bcolors, utils, argParser,dnn
import sys,time, glob, json, os,  time,  copy
import numpy as np
from os.path import join, split
#######################################################################################################################
bcolors = bcolors.bcolors()
parser  = argParser.argParser()
utils   = utils.Utils()
args    = parser.args
fl      = federatedlearning.FederatedLearning(args)
dnn     = dnn.DNN(fl)
#######################################################################################################################
OUTFOLDER         = join(utils.HOMEDIR,"results","curve_finding")
utils.create_dir(OUTFOLDER)
id_new            = utils.create_run_id(OUTFOLDER)
OUTFOLDER         = join(OUTFOLDER,"run_"+str(id_new))
utils.create_dir(OUTFOLDER)
utils.printout("Starting Curve Finding on dataset "+ args.dataset)
RESULTSPATH       = join(OUTFOLDER,'results.npz')
with open(join(OUTFOLDER,'settings.json'), 'w') as outfile:
    json.dump(vars(args), outfile)
for key,value in vars(args).items():
    print(key,"\t",value)
fl.init_data_set()
fl.init_gmcc()
fl.load_lossfn_metrics()
tot_n = 0
for X in fl.Xtrain:tot_n+=X.shape[0]
aggregate_weights = fl.init_models(t=0,clients=[0],aggregation=False)[0].get_weights()
fl.purge_memory()
############ Init result arrays ############


########################################################################################################################
##################################### Training Routine #################################################################
########################################################################################################################
try:
    for round in np.arange(1,args.rounds+1):
        utils.printout("ROUND "+ str(round)+"/"+str(args.rounds),bcolors.HEADER)
        fl.round=round
        scaled_local_weight_list = list()
        local_weight_list        = list()
        for client, idc in enumerate(range(0,fl.nclients)):
            utils.printout('CLIENT '+str(client)+" ROUND "+str(round),bcolors.LIGHTGREY)
            __local_model  = fl.init_model(t=0,client=client)
            __local_model.set_weights(aggregate_weights)
            __local_model  = fl.client_update_graph(__local_model,client,round)
            scaling_factor = fl.Xtrain[client].shape[0]/tot_n
            scaled_weights = fl.dnn.scale_model_weights(__local_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)
            local_weight_list.append(__local_model.get_weights())
            __local_model.set_weights(scaled_weights)

    utils.printout("Train Central Model",bcolors.HEADER)
    model_central = fl.init_model(t=0,client=23)
    model_central = fl.update_graph_central(model_central,23,0,epochs=10)
    #################################### Curve Finding ####################################
    utils.printout("Find curve",bcolors.HEADER)
    model_1 = fl.init_model(t=0,client=0)
    model_2 = fl.init_model(t=0,client=1)
    model_1.set_weights(local_weight_list[0])
    model_2.set_weights(local_weight_list[1])
    steps=10
    ####################################
    print("\n###### Curve Finding ######\n")
    theta, loss_curve, nu_array,loss_gamma_history,metric_gamma_history = fl.find_curve(model_1,model_2,steps=steps,n_runs=10)
    ####################################
    # print("\n###### Gammas ######\n")
    loss_gamma   = loss_gamma_history[-1]
    metric_gamma = metric_gamma_history[-1]
    print("\n###### Central ######\n")
    loss_central, metric_central,_,_    = fl.evaluate_model(model_central)
    print("\n###### FedAvg ######\n")
    fedavg  = fl.init_model(t=0,client=client)
    fedavg.set_weights(dnn.sum_scaled_weights(scaled_local_weight_list))
    loss_fedavg, metric_fedavg,_,_  = fl.evaluate_model(fedavg)
    ####################################
    print("\n###### CovFedAvg  ######\n")
    model_1.set_weights(local_weight_list[0])
    model_2.set_weights(local_weight_list[1])
    covfedavg=fl.init_model(t=0,client=client)
    covfedavg.set_weights(fl.gamma_average_curve(theta,model_1.get_weights(),model_2.get_weights(),steps=10))
    loss_covfedavg, metric_covfedavg,_,_    = fl.evaluate_model(covfedavg)
    ####################################
    print("\n###### Flat curve ######\n")
    loss_flat, metric_flat = np.zeros(steps),np.zeros(steps)
    flat_gamma = fl.init_model(t=0,client=client)
    for id,u in enumerate(np.linspace(0,1,steps)):
        flat_gamma.set_weights(fl.flat_curve(u,local_weight_list[0],local_weight_list[1]))
        loss_flat[id], metric_flat[id],_,_ = fl.evaluate_model(flat_gamma)
    #################################### Save results ####################################
    np.savez(RESULTSPATH,
            u_array          = np.linspace(0,1,loss_gamma.size),
            loss_curve       = loss_curve,
            loss_gamma       = loss_gamma,
            loss_gamma_history = loss_gamma_history,
            metric_gamma     = metric_gamma,
            loss_covfedavg   = loss_covfedavg,
            metric_covfedavg = metric_covfedavg,
            loss_fedavg      = loss_fedavg,
            metric_fedavg    = metric_fedavg,
            loss_flat        = loss_flat,
            metric_flat      = metric_flat,
            loss_central     = loss_central,
            metric_central   = metric_central)
    utils.printout("Saved results to "+RESULTSPATH,bcolors.OKGREEN)
    fl.purge_memory()
except KeyboardInterrupt:
    fl.purge_memory()
    sys.exit(0)

## python3 curve_finding.py --local_epochs 1 --nclients 2 --batchsize 128 --rounds 1 --dataset cifar10 --emd 30 --algorithm covfedavg --nu 0.001 --mc_size 4096

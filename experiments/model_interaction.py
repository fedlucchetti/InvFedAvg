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
client  = client.Client(args)
server  = server.Server(args)
client.init_model = server.init_model
dutils  = datautils.DataUtils(client)
dnn     = dnn.DNN(client)
#######################################################################################################################
utils.printout("GPU available= " + str(tf.config.experimental.list_physical_devices('GPU')),bcolors.OKCYAN)
client.init_data_set()
client.load_lossfn_metrics()
weights_common_init = server.init_model().get_weights()
iid_scale = "alpha"+str(args.alpha)+"_NI"+str(args.ni)
#######################################################################################################################
OUTFOLDER         = join(utils.HOMEDIR,"results","curve_finding")
utils.create_dir(OUTFOLDER)
OUTFOLDER         = join(OUTFOLDER,
                         args.dataset_type,
                         str(args.nclients),
                         platform.node())
utils.create_dir(OUTFOLDER)
id_new            = utils.create_run_id(OUTFOLDER)
OUTFOLDER         = join(OUTFOLDER,"run_"+str(id_new))
utils.create_dir(OUTFOLDER)
# utils.printout("Starting Curve Finding on dataset "+ args.dataset_type)
RESULTSPATH       = join(OUTFOLDER,'results.npz')
CKPTPATH          = join("workspace","model_interaction",
                        args.dataset_type,
                        str(args.nclients),
                        str(iid_scale),
                         "CI"+str(int(args.flag_common_weight_init)),
                         "B"+str(int(args.local_batchsize)) \
                         +"E"+str(int(args.local_epochs)) )

utils.create_dir(CKPTPATH)
print("CKPTPATH",CKPTPATH)
print("RESULTSPATH",RESULTSPATH)

with open(join(OUTFOLDER,'settings.json'), 'w') as outfile:
    json.dump(vars(args), outfile)
for key,value in vars(args).items():
    print(key,"\t",value)

########################################################################################################################
##################################### Training Routine #################################################################
########################################################################################################################
try:
    # for round in np.arange(1,args.rounds+1):
    comm_round=1
    utils.printout("ROUND "+ str(comm_round)+"/"+str(args.rounds),bcolors.HEADER)
    loss = np.zeros([args.nclients,args.n_local_runs])
    gammas = list()

    for client_id, idc in enumerate(range(0,args.nclients)):
        utils.printout('CLIENT '+str(client_id)+" ROUND "+str(comm_round),bcolors.LIGHTGREY)
        weights_local_models=list()
        if args.flag_preload_models:
            print("#### Loading trained client models from file ### \n")
            __local_model  = server.init_model(client_id=client_id)
            weights_local_models.append(keras.models.load_model(join(CKPTPATH,"model_"+str(client_id+1)+"_A")).get_weights())
            weights_local_models.append(keras.models.load_model(join(CKPTPATH,"model_"+str(client_id+1)+"_B")).get_weights())
            # __local_model.set_weights(weights_local_models[0])
            # client.evaluate_model(__local_model,client_id=client_id,verbose=1)
            # client.evaluate_model(__local_model,verbose=1)
            # __local_model.set_weights(weights_local_models[1])
            # client.evaluate_model(__local_model,client_id=client_id,verbose=1)
            # client.evaluate_model(__local_model,verbose=1)
            # continue
        else:
            for run in range(args.n_local_runs):
                print("\n Run ",run+1,"/",args.n_local_runs,"\n")
                __local_model  = server.init_model(client_id=client_id)
                if args.flag_common_weight_init:
                    print("Common weight init")
                    __local_model.set_weights(weights_common_init)
                __local_model,h  = client.client_update_graph(__local_model,client_id,comm_round,return_history=True)
                # client.evaluate_model(__local_model,verbose=1)
                loss[client_id,run] = h["loss"][-1]
                weights_local_models.append(__local_model.get_weights())
                del __local_model
                if len(weights_local_models)==3:
                    weights_local_models = client.select_most_different_models(weights_local_models)

        model_A,model_B = server.init_model(),server.init_model()
        model_A.set_weights(weights_local_models[0]);model_B.set_weights(weights_local_models[1])
        if not args.flag_preload_models:
            print("#### Saving trained client models ### \n")

            model_A.save(join(CKPTPATH,"model_"+str(client_id+1)+"_A"))
            model_B.save(join(CKPTPATH,"model_"+str(client_id+1)+"_B"))

        print("#### find_curve ### \n")
        if args.cf_nruns!=0:
            if args.cf_preload_theta==0:
                weights_theta, loss_curve, nu_array,loss_gamma_history,metric_gamma_history = client.find_curve(model_A,model_B,
                                                                                    flag_real=True,client_id=client_id,
                                                                                    steps=5,n_runs=args.cf_nruns)
                model_theta = server.init_model()
                model_theta.set_weights(weights_theta)
                model_theta.save(join(CKPTPATH,"theta_"+str(client_id+1)))
                model_A.save(join(CKPTPATH,"model_"+str(client_id+1)+"_A"))
            elif  args.cf_preload_theta==1:
                weights_theta = keras.models.load_model(join(CKPTPATH,"theta_"+str(client_id+1))).get_weights()
                print("gamma(t) models")
                client.evaluate_gammas(weights_theta, weights_local_models[0], weights_local_models[1],client_id=None,verbose=1,steps=100)
            gammas.append([weights_local_models[0],weights_local_models[1],weights_theta])
            # model_gamma_avg = server.init_model()
            # for i in range(50):
            #     t_array = np.random.uniform(0,1,5)
            #     model_gamma_avg.set_weights(server.average_gammas(gammas[client_id],t_array))
            #     print("gamma avg model t = ",t_array)
            #     client.evaluate_model(model_gamma_avg,client_id=None,verbose=1)
        else:
            continue
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

        #################################### Save results ####################################
        # np.savez(RESULTSPATH,
        #         u_array          = np.linspace(0,1,loss_gamma.size),
        #         loss_curve       = loss_curve,
        #         loss_gamma       = loss_gamma,
        #         loss_gamma_history = loss_gamma_history,
        #         metric_gamma     = metric_gamma,
        #         loss_covfedavg   = loss_covfedavg,
        #         metric_covfedavg = metric_covfedavg,
        #         loss_fedavg      = loss_fedavg,
        #         metric_fedavg    = metric_fedavg,
        #         loss_flat        = loss_flat,
        #         metric_flat      = metric_flat,
        #         loss_central     = loss_central,
        #         metric_central   = metric_central)
        # utils.printout("Saved results to "+RESULTSPATH,bcolors.OKGREEN)
        server.purge_memory()
except KeyboardInterrupt:
    server.purge_memory()
    sys.exit(0)




import numpy as np
import matplotlib.pyplot as plt
import sys,json,os, glob
from os.path import join,split
import argParser
import pandas as pd

#######################################################################################################################
parser  = argParser.argParser()
args    = parser.args

algorithm    = args.algorithm
dataset      = args.dataset_type
nclients     = str(args.nclients)
lr           = args.lr
local_epochs = args.local_epochs
alpha_set    = args.alpha
ni_set       = args.ni
flag_common_weight_init = args.flag_common_weight_init
#######################################################################################################################


def load_results():
    table=[]
    dirpath = join(algorithm,dataset,nclients)
    path_list_results  = []
    loss_array,accuracy_array,trial_array = [],[],[]
    alpha_array,ni_array = [],[]
    results = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dirpath) for f in filenames if os.path.splitext(f)[1] == '.npz']
    metas   = len(results)*[""]
    for idx,result_file in enumerate(results):
        metas[idx] = join(split(result_file)[0],"settings.json")
    for i,(path2meta,path2results) in enumerate(zip(metas,results)):
        if os.path.exists(path2meta) and os.path.exists(path2results):
            with open(path2meta,'r') as f: meta = json.load(f)
            if meta["lr"]!=float(lr):continue
            if meta["flag_common_weight_init"]!=float(flag_common_weight_init):continue
            if local_epochs!=None :
                if meta["local_epochs"]!=float(local_epochs):
                    continue
            results      = np.load(path2results)
            host         = split(split(split(path2results)[0])[0])[1]
            if len(results["loss_val"])<120:continue
            runid = split(split(path2meta)[0])[1][4::]
            alpha_array.append(meta["alpha"])
            ni_array.append(meta["ni"])
            loss = results["loss_val"]
            acc = results["metric"]
            mask = np.empty(200-len(loss))
            mask[:] = np.nan
            loss_array.append(np.concatenate([loss,mask]))
            accuracy_array.append(np.concatenate([acc,mask]))
            trial_array.append(int(runid))
            table.append([host,int(runid),meta["alpha"],meta["ni"],round(results["metric"][-1],2),meta["lr"],
            meta["flag_common_weight_init"],meta["local_epochs"],meta["local_batchsize"]])
            # print(host,"ID:\t",int(runid),"alpha",meta["alpha"],"ni",meta["ni"],
            # "acc",round(results["metric"][-1],2),"Nrounds",len(results["metric"]),"lr",meta["lr"])
    alpha_array=np.array(alpha_array)
    ni_array=np.array(ni_array)
    accuracy_array=np.array(accuracy_array)
    loss_array=np.array(loss_array)
    trial_array=np.array(trial_array)
    return alpha_array,ni_array,accuracy_array,loss_array,trial_array,table

fig,axs = plt.subplots(2,2)
title_string = algorithm+" "+" "+dataset+" $n_{clients}=$"+nclients+"  lr = "+str(lr)
fig.suptitle(title_string)


################################## Fix alpha vary NI ##################################

alpha_array,ni_array,accuracy_array,loss_array,trial_array,table = load_results()
df = pd.DataFrame(table, columns=['Host', 'RunID','alpha','NI','Acc','LR','common init','local epochs','local_batchsize'])
print("\n",df)

sel_alpha_idx = np.where(alpha_array==alpha_set)[0]
alpha_array,ni_array,accuracy_array,loss_array,trial_array = alpha_array[sel_alpha_idx],ni_array[sel_alpha_idx],accuracy_array[sel_alpha_idx],loss_array[sel_alpha_idx],trial_array[sel_alpha_idx]
nis      = np.unique(ni_array)
loss_    = np.zeros([len(nis),len(loss_array[0])])
loss_std = np.zeros([len(nis),len(loss_array[0])])
epochs   = np.arange(1,201)
title_string = "  alpha = "+str(alpha_set)
acc_ni_plt = list()
for ni_unique in nis:
    sel_idx_ni = np.where(ni_array==ni_unique)[0]
    loss_avg = np.mean(loss_array[sel_idx_ni],axis=0)
    loss_std = np.std(loss_array[sel_idx_ni],axis=0)
    acc_avg = np.mean(accuracy_array[sel_idx_ni],axis=0)
    acc_std = np.std(accuracy_array[sel_idx_ni],axis=0)

    axs[0,0].fill_between(epochs, loss_avg+loss_std, loss_avg-loss_std,label="NI = "+str(ni_unique),alpha=0.4)
    axs[1,1].fill_between(epochs, acc_avg+acc_std, acc_avg-acc_std,label="NI = "+str(ni_unique),alpha=0.4)
    acc_max = accuracy_array[sel_idx_ni].max(axis=1)
    acc_max = np.percentile(accuracy_array[sel_idx_ni],99.9,axis=1)
    acc_ni_plt.append([ni_unique,round(acc_max.mean(),3),round(acc_max.std(),3)])
    for idx, loss in enumerate(loss_array[sel_idx_ni]):
        axs[0,0].plot(loss,"--",linewidth=0.5)
axs[0,0].legend()
axs[0,0].grid(1)
axs[0,0].set_xlabel("Rounds")
axs[0,0].set_ylabel("Validation Loss")
axs[0,0].set_title(title_string)

################################## Fix NI vary alpha ##################################

alpha_array,ni_array,accuracy_array,loss_array,trial_array,table = load_results()
# sel_ni_idx = np.where(ni_array==ni_set)[0]
sel_ni_idx = np.where(np.logical_and(ni_array<ni_set+0.11,ni_array>ni_set-0.11))[0]
alpha_array,ni_array,accuracy_array,loss_array,trial_array = alpha_array[sel_ni_idx],ni_array[sel_ni_idx],accuracy_array[sel_ni_idx],loss_array[sel_ni_idx],trial_array[sel_alpha_idx]
alphas      = np.unique(alpha_array)
loss_    = np.zeros([len(alphas),len(loss_array[0])])
loss_std = np.zeros([len(alphas),len(loss_array[0])])
epochs   = np.arange(1,201)
title_string = " NI = "+str(ni_set)
acc_alpha_plt = list()
for alpha_unique in alphas:
    sel_idx_alpha = np.where(alpha_array==alpha_unique)[0]
    loss_avg = np.mean(loss_array[sel_idx_alpha],axis=0)
    loss_std = np.std(loss_array[sel_idx_alpha],axis=0)
    acc_avg = -np.mean(accuracy_array[sel_idx_alpha],axis=0)
    acc_std = np.std(accuracy_array[sel_idx_alpha],axis=0)
    axs[1,0].fill_between(epochs, loss_avg+loss_std, loss_avg-loss_std,label="Alpha = "+str(alpha_unique),alpha=0.4)
    axs[1,1].fill_between(epochs, acc_avg+acc_std, acc_avg-acc_std,label="Alpha = "       +str(alpha_unique),alpha=0.4)
    acc_max = accuracy_array[sel_idx_alpha].max(axis=1)
    # acc_max = np.percentile(accuracy_array[sel_idx_alpha],99.9,axis=1)
    acc_alpha_plt.append([alpha_unique,round(np.nanmean(acc_max),3),round(np.nanstd(acc_max),3)])
    for idx, loss in enumerate(loss_array[sel_idx_alpha]):
        axs[1,0].plot(loss,"--",linewidth=0.5)

acc_ni_plt    = np.array(acc_ni_plt)
acc_alpha_plt = np.array(acc_alpha_plt)


axs[1,0].legend()
axs[1,0].grid(1)
axs[1,0].set_xlabel("Rounds")
axs[1,0].set_ylabel("Validation Loss")
axs[1,0].set_title(title_string)

axs[0,1].plot([0,len(acc_ni_plt[:,1])],[0.78,0.78],linewidth=4,alpha=0.5,label="central")
axs[0,1].errorbar(np.arange(0,len(acc_ni_plt[:,1])), acc_ni_plt[:,1], acc_ni_plt[:,2],marker=".",label="NI "+str(ni_set))
x_tick_labels=list(acc_ni_plt[:,0].astype(str))
axs[0,1].set_xticks(np.arange(0,len(acc_ni_plt[:,1])))
axs[0,1].set_xticklabels(x_tick_labels)

axs[0,1].set_xlabel("Feature NIID")
axs[0,1].set_ylabel("Accuracy")
axs[0,1].set_ylim([0.4,1])
axs[0,1].legend()

axs[0,1].grid(1)

axs[1,1].set_ylabel("Accuracy")
axs[1,1].legend()
axs[1,1].grid(1)


ax2_parr = axs[0,1].twiny()
ax2_parr.errorbar(np.arange(0,len(acc_alpha_plt[:,1])), acc_alpha_plt[:,1], acc_alpha_plt[:,2],marker=".",label="$alpha$ "+str(alpha_set))
x_tick_labels=list(acc_alpha_plt[:,0].astype(str))
ax2_parr.set_xticks(np.arange(0,len(acc_alpha_plt[:,1])))
ax2_parr.set_xticklabels(x_tick_labels)
# ax2_parr.legend(loc=0)

ax2_parr.set_xlabel("Label alpha")
fig.tight_layout()


############################# Pretty Print Results #############################
print("########## Alpha ",alpha_set,"##########")
df = pd.DataFrame(acc_ni_plt, columns=['NI', '<Acc>','+-'])
print(df)
print("\n########## NI ",ni_set,"##########")
df = pd.DataFrame(acc_alpha_plt, columns=['Alpha', '<Acc>','+-'])
print(df)

plt.show()
print("---------------\n")
# sys.exit()

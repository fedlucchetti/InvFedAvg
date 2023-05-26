import numpy as np
import matplotlib.pyplot as plt
import sys,json,os, glob
from os.path import join



dir = join("curve_finding")
run_id = sys.argv[1]

data = np.load(join(dir,"run_"+run_id,"results.npz"))
with open(join(dir,"run_"+run_id,"settings.json"),'r') as f:
    meta = json.load(f)
# print(meta)

TITLE = meta["algorithm"]                       \
+"  N clients="    + str(meta["nclients"]) \
+"  dataset="      + str(meta["dataset"])\
+"  EMD"           + str(meta["emd"])\
+"\n local epochs"  + str(meta["local_epochs"])\
+"  MC size"       + str(meta["mc_size"])

print(TITLE)
u_array        = data["u_array"]
loss_curve     = data["loss_curve"]

loss_gamma     = data["loss_gamma"]
loss_central   = data["loss_central"]
loss_fedavg    = data["loss_fedavg"]
loss_covfedavg = data["loss_covfedavg"]
loss_gamma_history = data["loss_gamma_history"]


metric_gamma       = data["metric_gamma"]
metric_covfedavg   = data["metric_covfedavg"]
metric_fedavg      = data["metric_fedavg"]
metric_central     = data["metric_central"]

t_best_id          = np.argmax(metric_gamma)
metric_gamma_best  = metric_gamma[t_best_id]
t_best             = round(u_array[t_best_id],2)
print("t_best",t_best)

fig,axs = plt.subplots(2,2)
fig.suptitle(TITLE, fontsize=16)
axs[0,0].set_title("Curve Loss")
axs[0,0].plot(u_array,loss_curve[0],"r",linewidth=4                  ,label="First")
for id, loss in enumerate(loss_curve):
    axs[0,0].plot(u_array,loss,linewidth=2,label="Round "+str(id))
axs[0,0].plot(u_array,loss_curve[-1],"b",linewidth=2                  ,label="Final")
axs[0,0].grid(1)
axs[0,0].legend()


axs[1,0].set_title("Gamma Loss Validation")
axs[1,0].plot(u_array,loss_gamma_history[0],"r",linewidth=4                  ,label="First")
for id, loss in enumerate(loss_gamma_history):
    axs[1,0].plot(u_array,loss,linewidth=2,alpha=0.5,label="Round "+str(id))
axs[1,0].plot(u_array,loss_gamma_history[-1],"b",linewidth=2                  ,label="Final")
axs[1,0].plot(u_array,loss_central*np.ones(u_array.size),label="central")
axs[1,0].plot(u_array,loss_fedavg*np.ones(u_array.size),label="fedavg")
axs[1,0].plot(u_array,loss_covfedavg*np.ones(u_array.size),label="covfedavg")
axs[1,0].grid(1)
# axs[0,0].set_ylim([0,10])
axs[1,0].legend()

axs[0,1].set_title("Accuracy")
axs[0,1].plot(u_array,metric_gamma,linewidth=2              ,label="gammas")
axs[0,1].plot(u_array,metric_central*np.ones(u_array.size)  ,label="central")
axs[0,1].plot(u_array,metric_fedavg*np.ones(u_array.size)   ,label="fedavg")
axs[0,1].plot(u_array,metric_covfedavg*np.ones(u_array.size),label="covfedavg")
axs[0,1].set_ylim([0,1])
axs[0,1].grid(1)
axs[0,1].legend()

axs[1,1].set_title("Accuracy")
metrics = [metric_gamma_best,metric_covfedavg,metric_fedavg,metric_central]
labels  = ["best gamma \n t="+str(t_best),"covfedavg","fedavg","central"]
axs[1,1].bar([0,1,2,3],metrics      ,tick_label=labels)
axs[1,1].set_ylim([0,1])
axs[1,1].grid(1)


# axs[1,1].legend()
plt.show()


# ids=[]
# for _subdir in os.listdir(dir):
#     if _subdir[4:] == run_id:
#         break
#
# print("new id",id_new )

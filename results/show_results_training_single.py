import numpy as np
import matplotlib.pyplot as plt
import sys,json,os, glob
from os.path import join

algorithm = sys.argv[1]
dataset   = sys.argv[2]
nclients  = sys.argv[3]
run_id    = int(sys.argv[4])



def load_results():
    dir = join(algorithm,dataset,nclients)
    path_list_settings = []
    path_list_results  = []
    accuracy_array,loss_array,meta=[],[],[]
    for _subdir in os.listdir(dir):
        path2meta = join(dir,_subdir,"settings.json")
        path2results = join(dir,_subdir,"results.npz")
        if os.path.exists(path2meta) and os.path.exists(path2results):
            print(_subdir[4::],run_id)
            if int(int(_subdir[4::]))==run_id:
                with open(path2meta,'r') as f: meta = json.load(f)
                print("Found",path2meta)
                results     = np.load(path2results)
                loss        = results["loss_val"]
                acc         = results["metric"]
                mask        = np.empty(100-len(loss))
                mask[:]     = np.nan
                loss_array  = np.concatenate([loss,mask])
                accuracy_array = np.concatenate([acc,mask])
                print("ID:",int(_subdir[4::]),"alpha",meta["alpha"],"ni",meta["ni"],
                "acc",round(results["metric"][-1],2),"Nrounds",len(results["metric"]))
                break
    return accuracy_array,loss_array,meta


accuracy_array,loss_array,meta = load_results()
print(meta)
fig,axs = plt.subplots(1)
title_string = meta["algorithm"]+" "+" "+meta["dataset_type"]+" $n_{clients}=$"+str(meta["nclients"])+"  alpha = "+str(meta["alpha"]) + "  NI="+str(meta["ni"])
axs.plot(np.arange(1,len(loss_array)+1),loss_array,"k",label="Loss")
axs.set_ylabel("Loss")
axs.legend()
axs.grid(1)
axs.set_title(title_string)



ax2_parr = axs.twinx()
ax2_parr.plot(np.arange(1,len(accuracy_array)+1),accuracy_array,"r",label="Accuracy")
ax2_parr.set_ylabel("Accuracy")
ax2_parr.legend(loc=0)

fig.tight_layout()

plt.show()

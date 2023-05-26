import numpy as np
import matplotlib.pyplot as plt
import sys,json,os, glob
from os.path import join
from fnmatch import fnmatch



algorithm = "fedavg"


dir = algorithm


for subpath, subdirs, files in os.walk(join(dir,"fashion_mnist")):
    for filename in files:
        if fnmatch(filename, "*npz"):
            filepath = join(subpath,filename)
            data                  = np.load(filepath)
            val_loss_fedavg_array = data["val_loss_fedavg_array"]
            metric_fedavg_array   = data["metric_fedavg_array"]
            throughput            = data["throughput"]
            train_time            = data["train_time"]
            np.savez(filepath,
                    loss_val=val_loss_fedavg_array,
                    metric=metric_fedavg_array,
                    throughput=throughput,
                    train_time=train_time)
            print("DONE for ",print(filepath),"\n")

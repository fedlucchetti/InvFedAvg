from fedlearning import client, server, bcolors, utils, argParser,dnn
import sys,time, glob, json, os,  time,  copy
import numpy as np
from os.path import join, split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.data import Dataset as tf_dataset
tensor2ds = tf_dataset.from_tensor_slices
#######################################################################################################################
bcolors = bcolors.bcolors()
parser  = argParser.argParser()
utils   = utils.Utils()
args    = parser.args
args.algorithm="covfedavg"
args.emd=100
args.local_epochs=40
args.mc_size=1024
args.nclients=2
args.dataset_type="regression"

client  = client.Client(args)
server  = server.Server(args)
dnn     = dnn.DNN(client)
client.options.dataset_type="regression"



def load_dataset_1():
    datasets=list()
    x1    = np.random.normal(-0.5,0.25,500)
    x2    = np.random.normal(0.5,0.25,500)
    x     = np.concatenate([x1,x2])
    y1,y2 = np.abs(x1),np.abs(x2)
    y                   = np.abs(x)
    y1 = y1+np.random.normal(0.1, 0.1, y1.size)
    y2 = y2+np.random.normal(0.1, 0.1, y2.size)
    y  = y+np.random.normal(0.1, 0.1, y.size)
    y                   = y/y.max()
    y1,y2 = y1/y1.max(), y2/y2.max()
    datasets.append((x1,y1))
    datasets.append((x2,y2))
    return x1,x2,x,y1,y2,y,datasets

def load_dataset_2():
    datasets=list()
    x1    = np.random.normal(0.0,0.5,500)
    x2    = np.random.normal(0.0,0.5,500)
    x     = np.concatenate([x1,x2])
    y1    = np.abs(x1)
    y2    = -1*np.abs(x2)
    y     = np.abs(x)
    y1 = y1+np.random.normal(0.01, 0.01, y1.size)
    y2 = y2+np.random.normal(0.01, 0.01, y2.size)
    y  = y+np.random.normal(0.1, 0.1, y.size)
    # y                   = y/y.max()
    # y1,y2 = y1/y1.max(), y2/y2.max()
    datasets.append((x1,y1))
    datasets.append((x2,y2))
    return x1,x2,x,y1,y2,y,datasets

x1,x2,x,y1,y2,y,datasets = load_dataset_2()
plt.plot(x1,y1,"r.")
# plt.plot(x1,-y1,"r.")

plt.plot(x2,y2,"b.")
plt.show()

client.init_data_set(datasets)
client.load_lossfn_metrics()
tot_n = x.shape[0]
client.purge_memory()



########################################################################################################################
##################################### Training Routine #################################################################
########################################################################################################################
inputs = np.sort(x)
inputs = np.reshape(inputs,[inputs.size,1])
inputs1 = np.sort(x1)
inputs1 = np.reshape(inputs1,[inputs1.size,1])
inputs2 = np.sort(x2)
inputs2 = np.reshape(inputs2,[inputs2.size,1])

model_1_A = keras.models.load_model(join("workspace","saved_models_workspace","model_1_A"))
model_1_B = keras.models.load_model(join("workspace","saved_models_workspace","model_1_B"))
model_2_A = keras.models.load_model(join("workspace","saved_models_workspace","model_2_A"))
model_2_B = keras.models.load_model(join("workspace","saved_models_workspace","model_2_B"))
model_central = keras.models.load_model(join("workspace","saved_models_workspace","model_central"))
steps=10; n_runs=10

theta_1, _, _,_,_ = client.find_curve(model_1_A,model_1_B,steps=steps,n_runs=n_runs,flag_real=True,dataset=(inputs1,y1),flag_evaluate=True,client_id=0)
theta_2, _, _,_,_ = client.find_curve(model_2_A,model_2_B,steps=steps,n_runs=n_runs,flag_real=True,dataset=(inputs2,y2),flag_evaluate=True,client_id=1)

gammas=list()
gammas.append([model_1_A.get_weights(),model_1_B.get_weights(),theta_1])
gammas.append([model_2_A.get_weights(),model_2_B.get_weights(),theta_2])
model_1_inv,model_2_inv = server.model_interaction(gammas)


##################################### Compute Symmetric models #####################################




model_theta_1=fl.init_model()
model_theta_1.set_weights(theta_1)
model_theta_2=fl.init_model()
model_theta_2.set_weights(theta_2)
fl.evaluate_model(model_theta_1)
fl.evaluate_model(model_theta_2)
fl.evaluate_model(model_1_A)
fl.evaluate_model(model_1_B)
fl.evaluate_model(model_2_A)
fl.evaluate_model(model_2_B)

model_gamma       = fl.init_model()

############################## Model interaction ##############################
loss          = list()
u_array       = np.linspace(0,1,50)
loss_fn       = tf.losses.MeanSquaredError()
inputs_MC        = np.random.uniform(-1,1,1024)
inputs_MC        = np.reshape(inputs_MC,[inputs_MC.size,1])
loss_array = np.zeros([len(u_array),len(u_array)])
model_1_inv,model_2_inv = fl.init_model(),fl.init_model()
for i,ti in enumerate(u_array):
    for j,tj in enumerate(u_array):
        model_1_inv.set_weights(fl.gamma_curve(ti,theta_1, model_1_A.get_weights(),model_1_B.get_weights(),mode="bezier"))
        model_2_inv.set_weights(fl.gamma_curve(tj,theta_2, model_2_A.get_weights(),model_2_B.get_weights(),mode="bezier"))
        l = loss_fn(model_1_inv(inputs_MC),model_2_inv(inputs_MC)).numpy()
        loss_array[i,j] = l
        loss.append([int(i),int(j),l])
        print(ti,tj)

loss=np.array(loss)
t1=u_array[int(loss[np.argmin(loss[:,2])][0])]
t2=u_array[int(loss[np.argmin(loss[:,2])][1])]
model_1_inv.set_weights(fl.gamma_curve(t1,theta_1, model_1_A.get_weights(),model_1_B.get_weights(),mode="bezier"))
model_2_inv.set_weights(fl.gamma_curve(t2,theta_2, model_2_A.get_weights(),model_2_B.get_weights(),mode="bezier"))
###########################################################################
path_features_1 = list()
path_features_2 = list()
lxy_1,lxy_2 = np.zeros(len(u_array)),np.zeros(len(u_array))
for i,ti in enumerate(u_array):
    weights_1 = fl.gamma_curve(ti,theta_1, model_1_A.get_weights(),model_1_B.get_weights(),mode="bezier")
    weights_2 = fl.gamma_curve(ti,theta_2, model_2_A.get_weights(),model_2_B.get_weights(),mode="bezier")
    _path_features_1 = np.zeros(1)
    _path_features_2 = np.zeros(1)
    for _,w in enumerate(weights_1):
        _path_features_1 = np.concatenate([_path_features_1,w.numpy().flatten()])
    for _,w in enumerate(weights_2):
        _path_features_2 = np.concatenate([_path_features_2,w.numpy().flatten()])
    path_features_1.append(_path_features_1)
    path_features_2.append(_path_features_2)
    model_1_inv.set_weights(fl.gamma_curve(ti,theta_1, model_1_A.get_weights(),model_1_B.get_weights(),mode="bezier"))
    model_2_inv.set_weights(fl.gamma_curve(ti,theta_2, model_2_A.get_weights(),model_2_B.get_weights(),mode="bezier"))
    lxy_1[i] = fl.evaluate_model(model_1_inv)[0]
    lxy_2[i] = fl.evaluate_model(model_2_inv)[0]

def flatten_weights(model_weights):
    weights_flatten=np.zeros(1)
    for _,w in enumerate(model_weights):
        # print(_)
        # weights_flatten = np.concatenate([weights_flatten,w.numpy().flatten()])
        try:
            weights_flatten = np.concatenate([weights_flatten,w.numpy().flatten()])
        except Exception:
            weights_flatten = np.concatenate([weights_flatten,w.flatten()])
    return weights_flatten
path_features_1=np.array(path_features_1)
path_features_2=np.array(path_features_2)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pc1 = pca.fit_transform(path_features_1)
pc2 = pca.fit_transform(path_features_2)

plt.plot(pc1[:,0],pc1[:,1],label="gamma 1")
plt.plot(pc2[:,0],pc2[:,1],label="gamma 2")

plt.plot(pc1[0,0],pc1[0,1],"r.",label="$w_1 A$",markersize=16)
plt.plot(pc1[-1,0],pc1[-1,1],"b.",label="$w_1 B$",markersize=16)
plt.plot(pc2[0,0],pc2[0,1],"rx",label="$w_2 A$",markersize=16)
plt.plot(pc2[-1,0],pc2[-1,1],"bx",label="$w_2 B$",markersize=16)

idt_1_min=int(loss[np.argmin(loss[:,2])][0])
idt_2_min=int(loss[np.argmin(loss[:,2])][1])
plt.plot(pc1[idt_1_min,0],pc1[idt_1_min,1],"g.",label="$INV 1$",markersize=16)
plt.plot(pc2[idt_2_min,0],pc2[idt_2_min,1],"gx",label="$INV 2$",markersize=16)
plt.legend()
plt.show()

ax = plt.figure().add_subplot(projection='3d')
# ax.set_zlim([0,0.001])
ax.plot(pc1[:,0],pc1[:,1], lxy_1,"r", label='gamma 1')
ax.plot(pc2[:,0],pc2[:,1], lxy_2,"b", label='gamma 2')

ax.plot(pc1[0,0],pc1[0,1],lxy_1[0],"r.",markersize=12,label="$w_1 A$")
ax.plot(pc1[-1,0],pc1[-1,1],lxy_1[-1],"rx",markersize=12,label="$w_1 B$")
ax.plot(pc2[0,0],pc2[0,1],lxy_2[0],"b.",markersize=12,label="$w_2 A$")
ax.plot(pc2[-1,0],pc2[-1,1],lxy_2[-1],"bx",markersize=12,label="$w_2 B$")

ax.plot(pc1[idt_1_min,0],pc1[idt_1_min,1],lxy_1[idt_1_min],"g.",label="$INV 1$",markersize=12)
ax.plot(pc2[idt_2_min,0],pc2[idt_2_min,1],lxy_2[idt_2_min],"g.",label="$INV 2$",markersize=12)
ax.legend()
plt.show()


###########################################################################

plt.plot(inputs[:,0],model_1_A(inputs).numpy()[:,0],"r",linewidth=3,label="1a")
plt.plot(inputs[:,0],model_1_B(inputs).numpy()[:,0],"r",linewidth=3,label="1b")
plt.plot(inputs[:,0],model_2_A(inputs).numpy()[:,0],"b",linewidth=3,label="2a")
plt.plot(inputs[:,0],model_2_B(inputs).numpy()[:,0],"b",linewidth=3,label="2b")
plt.legend()
plt.show()

def plot_gammas(t1,t2):
    inputs_MC        = np.random.uniform(-1,1,1024)
    inputs_MC        = np.reshape(inputs_MC,[inputs_MC.size,1])
    plt.plot(x1,y1,"r.",markersize=0.5,label="client 1 ")
    plt.plot(x2,y2,"b.",markersize=0.5,label="client 2 ")
    model_1_inv,model_1_inv = fl.init_model(),fl.init_model()
    model_1_inv.set_weights(fl.gamma_curve(t1,theta_1, model_1_A.get_weights(),model_1_B.get_weights(),mode="bezier"))
    model_2_inv.set_weights(fl.gamma_curve(t2,theta_2, model_2_A.get_weights(),model_2_B.get_weights(),mode="bezier"))
    # model_1_inv = fl.gamma_curve_model(model_1_A.get_weights(),model_1_B.get_weights(),theta_1,t1)
    # model_2_inv = fl.gamma_curve_model(model_2_A.get_weights(),model_2_B.get_weights(),theta_2,t2)
    plt.plot(inputs_MC[:,0],model_1_inv(inputs_MC).numpy()[:,0],".r",linewidth=1,label="model inv 1 ")
    plt.plot(inputs_MC[:,0],model_2_inv(inputs_MC).numpy()[:,0],".b",linewidth=1,label="model inv 2 ")
    # plt.plot(inputs[:,0],model_1_inv(inputs).numpy()[:,0],"r",linewidth=2,label="model inv 1 ")
    # plt.plot(inputs[:,0],model_2_inv(inputs).numpy()[:,0],"b",linewidth=2,label="model inv 2 ")
    error = (np.abs(model_1_inv(inputs_MC).numpy() - model_2_inv(inputs_MC).numpy())**2).mean()
    l2 = error**2
    loss_fn(model_1_inv(inputs_MC),model_2_inv(inputs_MC))
    print(l)
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.legend()
    plt.show()
##################################### Plot procedure #####################################
plot_gammas(0.9,0.1)
# plt.plot(inputs[:,0],model_1_A(inputs).numpy()[:,0],"r--",linewidth=1,label="model A&B ")
# plt.plot(inputs[:,0],model_1_B(inputs).numpy()[:,0],"r--",linewidth=1)
plt.plot(inputs[:,0],model_1_inv(inputs).numpy()[:,0],"r",linewidth=2)


# plt.plot(inputs[:,0],model_2_A(inputs).numpy()[:,0],"b--",linewidth=1,label="model A&B ")
# plt.plot(inputs[:,0],model_2_A(inputs).numpy()[:,0],"b--",linewidth=1)
plt.plot(inputs[:,0],model_2_inv(inputs).numpy()[:,0],"b",linewidth=2)
plt.plot(x2,y2,"b.",markersize=0.5,label="client 2 ")
plt.plot(x1,y1,"r.",markersize=0.5,label="client 1 ")
plt.show()


for id,u in enumerate(np.linspace(0,1,10)):
    model_gamma   = fl.gamma_curve_model(model_2_A.get_weights(),model_2_B.get_weights(),theta_2,u)
    plt.plot(inputs[:,0],model_gamma(inputs).numpy()[:,0],"r",linewidth=0.5,label=str(round(u,2)))
    model_gamma   = fl.gamma_curve_model(model_1_A.get_weights(),model_1_B.get_weights(),theta_1,u)
    plt.plot(inputs[:,0],model_gamma(inputs).numpy()[:,0],"b",linewidth=0.5,label=str(round(u,2)))

model_gamma_top,model_gamma_down = fl.init_model(),fl.init_model()
model_gamma_top.set_weights(fl.gamma_curve(model_2_A.get_weights(),model_2_B.get_weights(),theta_2,0))
model_gamma_down.set_weights(fl.gamma_curve(model_2_A.get_weights(),model_2_B.get_weights(),theta_2,1))
plt.plot(inputs[:,0],model_1_inv(inputs).numpy()[:,0],"r",linewidth=2)
plt.plot(inputs[:,0],model_2_inv(inputs).numpy()[:,0],"b",linewidth=2)

# plt.plot(inputs[:,0],model_gamma_top(inputs).numpy()[:,0],"r",linewidth=1)
# plt.plot(inputs[:,0],model_gamma_down(inputs).numpy()[:,0],"b",linewidth=1)
plt.plot(x1,y1,"k.",markersize=2,label="client 1 ")
plt.plot(x2,y2,"k.",markersize=0.5,label="client 2 ")
plt.legend()
plt.show()

fig,axs = plt.subplots(3,2)
# fig.suptitle(TITLE, fontsize=16)
axs[0,0].set_title("Client 1 distribution")
axs[0,0].hist(x1,color="r")
axs[0,0].hist(x2,color="b",alpha=0.1)
axs[0,0].set_xlim([-1,1])
axs[0,1].set_title("Client 2 distribution")
axs[0,1].hist(x2,color="b")
axs[0,1].hist(x1,color="r",alpha=0.1)
axs[0,1].set_xlim([-1,1])


# for model in models_1:    axs[1,0].plot(inputs[:,0],model(inputs).numpy()[:,0],"k",linewidth=0.5)

axs[1,0].plot(inputs[:,0],model_1_A(inputs).numpy()[:,0],"r",linewidth=3,label="model A&B ")
axs[1,0].plot(inputs[:,0],model_1_B(inputs).numpy()[:,0],"r",linewidth=3)
axs[1,0].plot(x1,y1,"r.",markersize=2,label="client 1 ")
axs[1,0].plot(x2,y2,"b.",markersize=0.5,label="client 2 ")
# for id,u in enumerate(np.linspace(-0,1,10)):
#     model_gamma   = fl.gamma_curve_model(model_1_A.get_weights(),model_1_B.get_weights(),theta_1,u)
#     axs[1,0].plot(inputs[:,0],model_gamma(inputs).numpy()[:,0],"k",linewidth=0.5)


model_gamma_top,model_gamma_down = fl.init_model(),fl.init_model()
model_gamma_top.set_weights(fl.gamma_curve(0,theta_1, model_1_A.get_weights(), model_1_B.get_weights(),mode="bezier"))
model_gamma_down.set_weights(fl.gamma_curve(1,theta_1, model_1_A.get_weights(), model_1_B.get_weights(),mode="bezier"))
axs[1,0].fill_between(inputs[:,0],model_gamma_top(inputs).numpy()[:,0],
                                  model_gamma_down(inputs).numpy()[:,0],
                                  color="r",alpha=0.23)
axs[1,0].set_xlim([-1,1])
axs[1,0].set_ylim([-1,1])

axs[1,0].legend()

# for model in models_2:    axs[1,1].plot(inputs[:,0],model(inputs).numpy()[:,0],"k",linewidth=0.5)

axs[1,1].plot(inputs[:,0],model_2_A(inputs).numpy()[:,0],"b",linewidth=2,label="model A&B ")
axs[1,1].plot(inputs[:,0],model_2_B(inputs).numpy()[:,0],"b",linewidth=2)
axs[1,1].plot(x2,y2,"b.",markersize=2,label="client 2 ")
axs[1,1].plot(x1,y1,"r.",markersize=0.5,label="client 1 ")
# for id,u in enumerate(np.linspace(-0,1,10)):
#     model_gamma   = fl.gamma_curve_model(model_2_A.get_weights(),model_2_B.get_weights(),theta_2,u)
#     axs[1,1].plot(inputs[:,0],model_gamma(inputs).numpy()[:,0],"k",linewidth=0.5)

model_gamma_top,model_gamma_down = fl.init_model(),fl.init_model()
model_gamma_top.set_weights(fl.gamma_curve(0,theta_2, model_2_A.get_weights(), model_2_B.get_weights(),mode="bezier"))
model_gamma_down.set_weights(fl.gamma_curve(1,theta_2, model_2_A.get_weights(), model_2_B.get_weights(),mode="bezier"))
axs[1,1].fill_between(inputs[:,0],model_gamma_top(inputs).numpy()[:,0],
                                  model_gamma_down(inputs).numpy()[:,0],
                                  color="b",alpha=0.23)

axs[1,1].set_xlim([-1,1])
axs[1,1].set_ylim([-1,1])

axs[1,1].legend()

# model_gamma_top.set_weights(fl.gamma_curve(0,theta_1, model_1_A.get_weights(), model_1_B.get_weights(),mode="bezier"))
# model_gamma_down.set_weights(fl.gamma_curve(1,theta_1, model_1_A.get_weights(), model_1_B.get_weights(),mode="bezier"))
# axs[2,0].fill_between(inputs[:,0],model_gamma_top(inputs).numpy()[:,0],
#                                   model_gamma_down(inputs).numpy()[:,0],
#                                   color="r",alpha=0.1)
#
# model_gamma_top.set_weights(fl.gamma_curve(model_2_A.get_weights(),model_2_B.get_weights(),theta_2,0))
# model_gamma_down.set_weights(fl.gamma_curve(model_2_A.get_weights(),model_2_B.get_weights(),theta_2,1))
# axs[2,0].fill_between(inputs[:,0],model_gamma_top(inputs).numpy()[:,0],
#                                   model_gamma_down(inputs).numpy()[:,0],
#                                   color="b",alpha=0.1)

axs[2,0].plot(inputs[:,0],model_1_inv(inputs).numpy()[:,0],"r",linewidth=2,label="model inv 1 ")
axs[2,0].plot(x2,y2,"b.",markersize=0.5,label="client 2 ")
axs[2,0].plot(x1,y1,"r.",markersize=0.5,label="client 1 ")
axs[2,0].set_xlim([-1,1])
axs[2,0].set_ylim([-1,1])

w1_inv_scaled               = utils.scale_model_weights(model_1_inv.get_weights(), 0.5)
w2_inv_scaled               = utils.scale_model_weights(model_2_inv.get_weights(), 0.5)
w1_scaled                   = utils.scale_model_weights(model_1_A.get_weights(), 0.5)
w2_scaled                   = utils.scale_model_weights(model_2_A.get_weights(), 0.5)

model_inv_avg  = client.init_model()
model_fed_avg  = client.init_model()

model_inv_avg.set_weights(utils.sum_scaled_weights([w1_inv_scaled,w1_inv_scaled]))
model_fed_avg.set_weights(utils.sum_scaled_weights([w1_scaled    ,w2_scaled]    ))

axs[2,1].plot(inputs[:,0],model_2_inv(inputs).numpy()[:,0]       ,"b",linewidth=2,label="model inv 2 ")
axs[2,1].plot(inputs[:,0],model_fed_avg(inputs).numpy()[:,0],    "g",linewidth=2,label="fedavg ")
axs[2,0].plot(inputs[:,0],model_inv_avg(inputs).numpy()[:,0],    "k",linewidth=2,label="model_inv_avg ")
axs[2,1].plot(inputs[:,0],model_inv_avg(inputs).numpy()[:,0],    "k",linewidth=2,label="model_inv_avg ")

# axs[2,1].plot(inputs[:,0],model_central(inputs).numpy()[:,0],    "k",linewidth=2,label="central ")

axs[2,1].plot(x2,y2,"b.",markersize=0.5,label="client 2 ")
axs[2,1].plot(x1,y1,"r.",markersize=0.5,label="client 1 ")
axs[2,1].set_xlim([-1,1])
axs[2,1].set_ylim([-1,1])
plt.legend()

plt.show()

l_central,_,_,_ = client.evaluate_model(model_central)
l_1_inv,_,_,_   = client.evaluate_model(model_1_inv)
l_2_inv,_,_,_   = client.evaluate_model(model_2_inv)
l_1_local,_,_,_ = client.evaluate_model(model_1_A)
l_2_local,_,_,_ = client.evaluate_model(model_2_A)
l_avg_inv,_,_,_ = client.evaluate_model(model_inv_avg)
l_fedavg,_,_,_  = client.evaluate_model(model_fed_avg)

while True:
    print("#####################")
    print("Central \t",round(l_central,3))
    print("INV 1   \t",round(l_1_inv,3))
    print("INV 2   \t",round(l_2_inv,3))
    print("LOC 1   \t",round(l_1_local,3))
    print("LOC 2   \t",round(l_2_local,3))
    print("INV AVG \t",round(l_avg_inv,3))
    print("FED AVG \t",round(l_fedavg,3))
    print("#####################")
    break

arr = [1-round(l_central,2),1-round(l_1_inv,2),1-round(l_2_inv,2),1-round(l_fedavg,2)]

plt.bar([0,1,2,3],arr)
plt.ylim([0,1])
plt.show()

#
# utils.printout("Train Central Model",bcolors.HEADER)
# model_central = fl.init_model(t=0,client=23)
# model_central = fl.update_graph_central(model_central,23,0,epochs=10)
# #################################### Curve Finding ####################################
# utils.printout("Find curve",bcolors.HEADER)
#
# steps=10
# ####################################
# print("\n###### Curve Finding ######\n")
# theta, loss_curve, nu_array,loss_gamma_history,metric_gamma_history = fl.find_curve(model_1,model_2,steps=steps,n_runs=10)
# ####################################
# # print("\n###### Gammas ######\n")
# loss_gamma   = loss_gamma_history[-1]
# metric_gamma = metric_gamma_history[-1]
# print("\n###### Central ######\n")
# loss_central, metric_central,_,_    = fl.evaluate_model(model_central)
# print("\n###### FedAvg ######\n")
# fedavg  = fl.init_model(t=0,client=client)
# fedavg.set_weights(dnn.sum_scaled_weights(scaled_local_weight_list))
# loss_fedavg, metric_fedavg,_,_  = fl.evaluate_model(fedavg)
# ####################################
# print("\n###### CovFedAvg  ######\n")
# model_1.set_weights(local_weight_list[0])
# model_2.set_weights(local_weight_list[1])
# covfedavg=fl.init_model(t=0,client=client)
# covfedavg.set_weights(fl.gamma_average_curve(theta,model_1.get_weights(),model_2.get_weights(),steps=10))
# loss_covfedavg, metric_covfedavg,_,_    = fl.evaluate_model(covfedavg)
# ####################################
# print("\n###### Flat curve ######\n")
# loss_flat, metric_flat = np.zeros(steps),np.zeros(steps)
# flat_gamma = fl.init_model(t=0,client=client)
# for id,u in enumerate(np.linspace(0,1,steps)):
#     flat_gamma.set_weights(fl.flat_curve(u,local_weight_list[0],local_weight_list[1]))
#     loss_flat[id], metric_flat[id],_,_ = fl.evaluate_model(flat_gamma)
# #################################### Save results ####################################
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
# fl.purge_memory()
# fl.purge_memory()
# sys.exit(0)
#
# ## python3 curve_finding.py --local_epochs 1 --nclients 2 --batchsize 128 --rounds 1 --dataset cifar10 --emd 30 --algorithm covfedavg --nu 0.001 --mc_size 4096

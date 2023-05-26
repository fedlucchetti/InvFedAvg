import numpy as np
import matplotlib.pyplot as plt

data      = np.load("cifar10_alpha_ni.npz")["data"][10:,:]
alphas = np.unique(data[:,0])
y_avg,y_std=[],[]
for alpha in alphas:
    sel_ni = data[np.where(data[:,0]==alpha)[0],1]
    y_avg.append(sel_ni.mean())
    y_std.append(sel_ni.std())

y_avg,y_std=np.array(y_avg),np.array(y_std)
print(y_avg.shape,y_std.shape)
# plt.plot(alphas,y_avg+y_std)
# plt.plot(alphas,y_avg-y_std)
plt.fill_between(alphas,y_avg+2*y_std,y_avg-2*y_std,alpha=0.3)
plt.plot(data[:,0],data[:,1],'k.',alpha=0.2,markersize=0.5)
plt.xscale('log')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Non-IID CIFAR10 --- 30 clients")

plt.xlabel("Dirichlet alpha",fontsize=16)
plt.ylabel("NI",fontsize=16)
plt.grid(1)
plt.show()
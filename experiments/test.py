import time
import numpy as np
import matplotlib.pyplot as plt


t_array = np.linspace(0,1,100)

def bezier2(t_array,w1,w2,theta):
    out = np.zeros([len(t_array),2])
    for idt,t in enumerate(t_array):
        out[idt] = (1-t)**2 * w1 + 2*t*(1-t)*theta + t**2*w2
    return out

def bezier3(t_array,w1,w2,theta):
    theta1,theta2=theta
    out = np.zeros([len(t_array),2])
    for idt,t in enumerate(t_array):
        out[idt] = (1-t)**3 * w1 + 3*t*(1-t)**2*theta1 + 3*(1-t)*t**2 * theta2 + t**3*w2
    return out

w1    = np.array([1,1])
w2    = np.array([5,1])



theta = np.array([8,-2])
curve = bezier2(t_array,w1,w2,theta)
plt.plot(curve[:,0],curve[:,1],label="bezier2")
plt.plot(theta[0],theta[1],"x",markersize=16)

theta = np.array([[3.3,-20],[4,20]])
curve = bezier3(t_array,w1,w2,theta)
plt.plot(curve[:,0],curve[:,1],label="$bezier3$")
plt.plot(theta[0,0],theta[0,1],".",label="$theta_1$",markersize=16)
plt.plot(theta[1,0],theta[1,1],".",label="$theta_2$",markersize=16)

plt.legend()
plt.show()

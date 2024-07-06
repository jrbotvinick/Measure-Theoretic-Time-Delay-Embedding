import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
from scipy.spatial import cKDTree
import random 
from sklearn.cluster import KMeans, MiniBatchKMeans

with open("data_embedded.p", "rb") as f:
    data = pickle.load(f)
with open("patches.p", "rb") as f:
    patches = pickle.load(f)
with open("lorenz_measures.p", "rb") as f:
    measures = pickle.load(f)    
with open("lorenz_pointwise.p", "rb") as f:
    points = pickle.load(f) 
PO,PD,samples,output = patches[0], patches[1], patches[2], patches[3]
ys,ys_noise,ys_delay,ys_delay_test,ys_test,resize,dim,tau,ts = data[0],data[1],data[2], data[3],data[4], data[5], data[6], data[7], data[8]
X,Y = PD, PO

PD, PO = PD*resize, PO*resize
ys_P = points[0]*resize
ys_M = measures[0]*resize



fig, ax = plt.subplots(subplot_kw={"projection": "3d"},dpi = 300,figsize = (10,5))
ax.view_init(10,120)

ax.plot(ys_M[:,0],ys_M[:,1],ys_M[:,2],linewidth =  .015,c = 'k')
ax.set_xlabel(r'$x$',fontsize = 15)
ax.set_ylabel(r'$y$',fontsize = 15)
ax.set_zlabel(r'$z$',fontsize = 15)
ax.set_xticks(np.linspace(-20,20,5))
ax.set_yticks(np.linspace(-20,20,5))
ax.set_zticks(np.linspace(10,40,4))
ax.set_xlim(-22,22)
ax.set_ylim(-22,22)
ax.set_zlim(5,40)
plt.suptitle('Measure-Based Reconstruction', size=12,y = .8);      
ax.grid(False)
ax.set_box_aspect(aspect=None, zoom=.8)
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"},dpi = 300,figsize = (10,5))
ax.view_init(10,120)
ax.plot(ys_P[:,0],ys_P[:,1],ys_P[:,2],linewidth =  .015,c = 'k')
ax.set_xlabel(r'$x$',fontsize = 15)
ax.set_ylabel(r'$y$',fontsize = 15)
ax.set_zlabel(r'$z$',fontsize = 15)
ax.set_xticks(np.linspace(-20,20,5))
ax.set_yticks(np.linspace(-20,20,5))
ax.set_zticks(np.linspace(10,40,4))
ax.set_xlim(-22,22)
ax.set_ylim(-22,22)
ax.set_zlim(5,40)
plt.suptitle('Pointwise Reconstruction', size=12,y = .8);      
ax.grid(False)
ax.set_box_aspect(aspect=None, zoom=.8)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"},dpi = 300,figsize = (10,5))
ax.view_init(10,120)
ax.plot(ys_test[:,0]*resize,ys_test[:,1]*resize,ys_test[:,2]*resize,linewidth = .015,c = 'k')
ax.set_xlabel(r'$x$',fontsize = 15)
ax.set_ylabel(r'$y$',fontsize = 15)
ax.set_zlabel(r'$z$',fontsize = 15)
ax.set_xticks(np.linspace(-20,20,5))
ax.set_yticks(np.linspace(-20,20,5))
ax.set_zticks(np.linspace(10,40,4))
ax.set_xlim(-22,22)
ax.set_ylim(-22,22)
ax.set_zlim(5,40)
plt.suptitle('Ground Truth', size=12,y = .8);      
ax.grid(False)
ax.set_box_aspect(aspect=None, zoom=.8)

error1 = np.mean(np.sum((ys_P - ys_test*resize)**2,axis = 1))
error2 = np.mean(np.sum((ys_M - ys_test*resize)**2,axis = 1))
print('Pointwise error:', error1)
print('Measure error:', error2)
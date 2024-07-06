import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
from scipy.spatial import cKDTree
import random 
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
from k_means_constrained import KMeansConstrained
import time

with open("data_embedded.p", "rb") as f:
    data = pickle.load(f)


ys,ys_noise,ys_delay = data[0],data[1],data[2]
resize = data[7]

X, Y = ys_delay, ys_noise
output = Y

_,dim1 = np.shape(output)
start = time.time()
N_samples,dim0 = np.shape(X)
N_patches = 40 ##needs to evenly divide N_samples

clf = KMeansConstrained(
     n_clusters=N_patches,
     size_min= N_samples//N_patches,
     size_max=N_samples//N_patches,
     random_state=0,
     max_iter = 100)

point_idxs = clf.fit_predict(X)
samples = clf.cluster_centers_

bins = [[] for i in range(N_patches)]
for i in range(len(point_idxs)):
    bins[point_idxs[i]].append(i)  
 
patch_input = np.zeros((N_patches,N_samples//N_patches,dim0))    
patch_output = np.zeros((N_patches,N_samples//N_patches,dim1))    


for i in range(N_patches):
    patch_input[i,:,:] = X[bins[i]]
    patch_output[i,:,:] = output[bins[i]]
    plt.scatter(X[bins[i]][:,0],X[bins[i]][:,1],s = 10)

plt.show()


for i in range(N_patches):
    plt.scatter(Y[bins[i]][:,0],Y[bins[i]][:,1],s = 10)
    

patch_input = torch.tensor(patch_input,dtype = torch.float)
patch_output = torch.tensor(patch_output,dtype = torch.float)


end = time.time()
print('total time:', end-start)
with open("patches.p", "wb") as f:
    pickle.dump([patch_output,patch_input,samples,output], f)




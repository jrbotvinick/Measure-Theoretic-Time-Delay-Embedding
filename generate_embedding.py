import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.spatial import KDTree
from teaspoon.parameter_selection.MI_delay import MI_for_delay
from teaspoon.parameter_selection.FNN_n import FNN_n

with open("data.p", "rb") as f:
    data = pickle.load(f)
    
ts = data[2]
ys = data[0]
ys_noise = data[1]
x = ys[:,0] #signal to embed
x_noise = ys_noise[:,0] #signal to embed

N,_ = np.shape(ys)
skips = 10
xcheck = x[::skips]
tau = MI_for_delay(xcheck, plotting = False, method = 'basic', h_method = 'standard', k = 2, ranking = True)
print('Time delay:', tau)

perc_FNN, n = FNN_n(xcheck, tau, threshold = 5,plotting = False,method = 'cao',maxDim=10)
print('Embedding dimension:', n)
tau = tau*skips

ys_delay = np.zeros((N-tau*(n-1),n))
ys_delay_noise= np.zeros((N-tau*(n-1),n))

for i in range(n):
    ys_delay[:,i] = x[tau*(n-1-i):N-i*tau]
    ys_delay_noise[:,i] = x_noise[tau*(n-1-i):N-i*tau]


import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
import random



cutoff = int(1e5)
num_test = int(5e5)
num_train = int(2000)
ixs = random.sample(range(0,cutoff),num_train)



ys = ys[tau*(n-1):]
ys_noise = ys_noise[tau*(n-1):]
resize = np.max(np.abs(ys_noise))
ys = ys/resize
ys_noise =ys_noise/resize
ys_delay = ys_delay/resize
ys_delay_noise = ys_delay_noise/resize


ys_delay_train = ys_delay_noise[:cutoff]
ys_delay_test = ys_delay[cutoff+int(1e5):int(1e5)+cutoff+num_test]
ts = ts[cutoff+int(1e5):int(1e5)+cutoff+num_test]
ys_test = ys[cutoff+int(1e5):int(1e5)+cutoff+num_test]
plt.plot(perc_FNN)
plt.show()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"},dpi = 300,figsize = (10,5))

ax.plot(ys_delay_train[:,0],ys_delay_train[:,1],ys_delay_train[:,2],linewidth = .1)
plt.show()



with open("data_embedded.p", "wb") as f:
    pickle.dump([ys[:cutoff][ixs],ys_noise[:cutoff][ixs],ys_delay_train[ixs],ys_delay_test,ys_test,resize,n,tau,ts], f)



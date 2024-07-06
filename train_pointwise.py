import matplotlib.pyplot as plt
import numpy as np
from time import time
import pickle
from tqdm import tqdm
import torch 
from torch import optim
import random 
from geomloss import SamplesLoss
import torch.nn as nn
from functools import reduce

with open("data_embedded.p", "rb") as f:
    data = pickle.load(f)
with open("patches.p", "rb") as f:
    patches = pickle.load(f)
    

PO,PD,samples,output = patches[0], patches[1], patches[2], patches[3]
ys,ys_noise,ys_delay,ys_delay_test,_,resize,dim,tau,ts = data[0],data[1],data[2], data[3],data[4], data[5], data[6], data[7], data[8]
X,Y = PD, PO
ys_delay_test = ys_delay_test[:int(1e5)]
_,_,dim1 = np.shape(PD)
ll = len(PO)
sh = np.shape(output)
if len(sh) == 1:
    dim0 = 1
else:
    dim0 = sh[1]
#########
nodes = 100
learning_rate = 1e-3
num_training_steps = 50000
plotevery = 1001
_,dim1 = np.shape(ys_delay)
torch.manual_seed(742)
############################# build network
net = nn.Sequential(
    nn.Linear(dim1, nodes,bias=True),
    nn.Tanh(),
    nn.Linear(nodes, nodes,bias=True),
    nn.Tanh(),
    nn.Linear(nodes, nodes,bias=True),
    nn.Tanh(),
    nn.Linear(nodes, nodes,bias=True),
    nn.Tanh(),
    nn.Linear(nodes,dim0,bias = True))

################################## Perform optimization
optimizer = optim.Adam(net.parameters(), lr=learning_rate)    
tbar = tqdm(range(num_training_steps))
loss_list = []

start = time()

X = torch.tensor(ys_delay,dtype = torch.float)
Y = torch.tensor(ys_noise, dtype = torch.float)
for step in tbar:
    # X,Y = PD, PO

    

    net.train()
    optimizer.zero_grad()
    out = net(X)
    L = (torch.linalg.norm(Y-out,dim = 1)**2).mean()
    L.backward()
    
    if step % plotevery == 0:
        net.eval()
        XX = torch.tensor(ys_delay_test,dtype = torch.float)
        YY = net(XX).detach().numpy()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"},dpi = 300,figsize = (10,5))
        ax.plot(YY[:,0],YY[:,1],YY[:,2],linewidth = 0.1)
        ax.view_init(30,120)
        plt.show()
        
    loss_list.append(L.detach().numpy())
    optimizer.step() 
    s = 'Loss: {:.4f}'.format(L.item())
    tbar.set_description(s)  
end = time()
plt.plot(loss_list)
plt.yscale('log')
plt.show()
ys_delay_test = data[3]

XX = torch.tensor(ys_delay_test,dtype =torch.float)
output = net(XX).detach().numpy()
XX = torch.tensor(ys_delay_test,dtype =torch.float)
output = net(XX).detach().numpy()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"},dpi = 300,figsize = (10,5))
ax.plot(output[:,0],output[:,1],output[:,2],linewidth = .05)
ax.view_init(30,120)
with open("lorenz_pointwise.p", "wb") as f:
    pickle.dump([output,end - start], f)


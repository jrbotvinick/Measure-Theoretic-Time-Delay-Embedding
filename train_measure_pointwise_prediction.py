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
import random

seeds = [3580109182, 3037730684, 1047487952, 3844489996, 728317660]



#To run this file, create data_embedded.p, data_embedded_clean.p, patches_noisy.p and patches_clean.p using the generate_data.py, generate_embedding_prediction.py, and generate_patches.py files

#choose whether to do forecasting on clean/noisy data and whether to use pointwise or measure matching
noisy = True
method = 'measure' 




if noisy == True:
    with open("data_embedded.p", "rb") as f:
        data = pickle.load(f)
    with open("patches_noisy.p", "rb") as f:
        patches = pickle.load(f)
if noisy == False:  
    with open("data_embedded_clean.p", "rb") as f:
        data = pickle.load(f)
    with open("patches_clean.p", "rb") as f:
        patches = pickle.load(f)
        

for iiii in range(5):
    
    PO,PD,samples,output = patches[0], patches[1], patches[2], patches[3]
    ys,ys_noise,ys_delay,ys_delay_test,_,resize,dim,tau,ts, Nskip = data[0],data[1],data[2], data[3],data[4], data[5], data[6], data[7], data[8], data[9]
    ys_delay_test = ys_delay_test[:int(1e5)]
    sh = np.shape(output)
    if len(sh) == 1:
        dim0 = 1
    else:
        dim0 = sh[1]
    _,dim1 = np.shape(PD[0])
    ll = len(PO)
    
    #########
    nodes = 100
    learning_rate = 1e-3
    num_training_steps = 50000
    plotevery = 1001
    dt = .01
    Nsteps = 10
    decay_by = 1e-2
    
    torch.manual_seed(seeds[iiii])
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
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_by**(1/num_training_steps))
    tbar = tqdm(range(num_training_steps))
    loss_list = []
    loss = SamplesLoss(loss="energy")
    
    start = time()
    
    for step in tbar:
        net.train()
        optimizer.zero_grad()
        if method == 'measure':
            X,Y = PD, PO
        if method == 'pointwise':
            X = torch.tensor(ys_delay,dtype = torch.float)
            Y = torch.tensor(ys_noise, dtype = torch.float)
            
        
        for i in range(Nsteps):
            X = X+dt*net(X) 
        out = X
        # out = net(X)
        if method == 'measure':
            L = loss(out,Y).mean()  #+torch.linalg.norm(out-Y)
        if method == 'pointwise':
            L = (torch.linalg.norm(Y-out,dim = 1)**2).mean()
        L.backward()
        loss_list.append(L.detach().numpy())
        optimizer.step() 
        scheduler.step()
        s = 'Loss: {:.16f}'.format(L.item())
        tbar.set_description(s)  
        if step % plotevery == 0: 
            
            
            

            net.eval()
            x = torch.tensor(ys_delay_test[0],dtype = torch.float)
            traj = []
            for i in range(int(1e4)):
                traj.append(x.detach().numpy())
                x = x+dt*net(x) 
                # x = net(x)
            traj = np.array(traj)
            plt.plot(traj[:,0][:int(1e3)])
            plt.plot(ys_delay_test[:,0][::Nskip//Nsteps][:int(1e3)])
            plt.show()
            
            
          
            plt.plot(ys_delay_test[:,0],ys_delay_test[:,1])
            plt.plot(traj[:,0],traj[:,1])
            plt.show()
    
            
            
    
    end = time()     
    plt.plot(loss_list)
    plt.yscale('log')
    plt.show()
    ys_delay_test = data[3][int(1e4):]
    XX = torch.tensor(ys_delay_test,dtype =torch.float)
    
    net.eval()
    x = torch.tensor(ys_delay_test[0],dtype = torch.float)
    traj = []
    for i in range(int(3e4)):
        traj.append(x.detach().numpy())
        x = x+dt*net(x) 
    traj = np.array(traj)
    
    plt.plot(ys_delay_test[:,0][::Nskip//Nsteps][:int(1e3)])
    plt.plot(traj[:,0][:int(1e3)])
    plt.show()
    
    
    plt.plot(ys_delay_test[:,0],ys_delay_test[:,1])
    plt.plot(traj[:,0],traj[:,1])
    plt.show()
    
    

    
    if noisy == True:
        if method == 'measure':
            torch.save(net.state_dict(), "noisy_measures{}.pth".format(iiii))
            with open("lorenz_forecast_measure_noisy{}.p".format(iiii), "wb") as f: 
                pickle.dump([traj,ys_delay_test], f)
        if method == 'pointwise':
            torch.save(net.state_dict(), "noisy_pointwise{}.pth".format(iiii))
            with open("lorenz_forecast_pointwise_noisy{}.p".format(iiii), "wb") as f:
                pickle.dump([traj,ys_delay_test], f)
            
    # torch.save(net.state_dict(), "lorenz_predict_clean_measures.pth")

    if noisy == False:
        if method == 'measure':
              torch.save(net.state_dict(), "clean_measures{}.pth".format(iiii))
              with open("lorenz_forecast_measure_clean{}.p".format(iiii), "wb") as f: 
                  pickle.dump([traj,ys_delay_test], f)
        if method == 'pointwise':
            torch.save(net.state_dict(), "clean_pointwise{}.pth".format(iiii))
            with open("lorenz_forecast_pointwise_clean{}.p".format(iiii), "wb") as f:
                pickle.dump([traj,ys_delay_test], f)
                 
             
        
        
        
        
        

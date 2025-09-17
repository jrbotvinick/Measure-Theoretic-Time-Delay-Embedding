import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
with open("data_embedded.p", "rb") as f:
    data = pickle.load(f)
   
with open("data_embedded_clean.p", "rb") as f:
    data2 = pickle.load(f)
    
resize = data[-5]
resize2 = data2[-5]

ys_delay_plot = data[2]*resize

data_m = []
data_p = []
for i in range(5):
    with open("lorenz_forecast_measure_noisy{}.p".format(i), "rb") as f:
        data = pickle.load(f)
    with open("lorenz_forecast_pointwise_noisy{}.p".format(i), "rb") as f:
        data2 = pickle.load(f)
    data_m.append(data)
    data_p.append(data2)
        
data_m2 = []
data_p2 = []
for i in range(5):
    with open("lorenz_forecast_measure_clean{}.p".format(i), "rb") as f:
        data = pickle.load(f)
    with open("lorenz_forecast_pointwise_clean{}.p".format(i), "rb") as f:
        data2 = pickle.load(f)
    data_m2.append(data)
    data_p2.append(data2)
        
    
ix = 0
trajm,ys = data_m[ix][0], data_m[ix][1]
trajp = data_p[ix][0]
trajm = trajm*resize
FWDm, FWDp = data_m[ix][2]*resize, data_p[ix][2]*resize
FWD = data_p[ix][3]*resize
ys = ys*resize
ys = ys[:len(trajp)]
trajp = trajp*resize

trajm2,ys2 = data_m2[ix][0], data_m2[ix][1]
trajp2 = data_p2[ix][0]
trajm2 = trajm2*resize2
FWDm2, FWDp2 = data_m2[ix][2]*resize2, data_p2[ix][2]*resize2
FWD2 = data_p2[ix][3]*resize2
ys2 = ys2*resize2
ys2 = ys2[:len(trajp2)]
trajp2 = trajp2*resize2


fig, ax = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(10, 4),dpi = 300)

# ax[0].scatter(ys_delay_plot[:,0],ys_delay_plot[:,1],ys_delay_plot[:,2],color = 'r',s = .1,alpha = 0.5)
ax[0].plot(ys[:,0],ys[:,1],ys[:,2],linewidth = 0.1,color = 'k')
ax[1].plot(trajp[:,0],trajp[:,1],trajp[:,2],linewidth = 0.1,color = 'k')
ax[2].plot(trajm[:,0],trajm[:,1],trajm[:,2],linewidth = 0.1,color = 'k')

for i in range(3):
    ax[i].view_init(45, 320)
    ax[i].set_xlim(-15,15)
    ax[i].set_ylim(-15,15)
    ax[i].set_zlim(-15,15)
    ax[i].tick_params(pad=-1.5,labelsize = 5)
    ax[i].set_xlabel(r'$x(t)$',fontsize = 10,labelpad = -2)
    ax[i].set_ylabel(r'$x(t-\tau)$',fontsize = 10,labelpad = -2)
    ax[i].set_zlabel(r'$x(t-2\tau)$',fontsize = 10,labelpad = -4)
    ax[i].set_box_aspect(None, zoom=0.8)

ax[0].set_title('Ground Truth',fontsize = 12,y = 0.97)
ax[1].set_title('Pointwise',fontsize = 12,y = 0.97)
ax[2].set_title('Measure',fontsize = 12,y = 0.97)
plt.subplots_adjust(wspace=0.05)
    


    
    
plt.show()
lw = 2.5
Nplot = 1000
ts = np.linspace(0,Nplot*0.01,Nplot)
fig,ax  = plt.subplots(2,2,figsize = (20,6),dpi = 300)
ax[0,0].plot(ts,ys[:,0][:Nplot],linewidth = lw,color = 'k',alpha = 0.4)
ax[1,0].plot(ts,ys[:,0][:Nplot],linewidth = lw,color = 'k',alpha = 0.4)
ax[1,1].plot(ts,ys[:,0][:Nplot],linewidth = lw,color = 'k',alpha = 0.4)
ax[0,1].plot(ts,ys[:,0][:Nplot],linewidth = lw,color = 'k',alpha = 0.4)

ax[0,1].plot(ts,trajp[:,0][:Nplot],'--',linewidth = lw,color = 'mediumblue')
ax[1,1].plot(ts,trajm[:,0][:Nplot],'--',linewidth = lw,color = 'mediumblue')

ax[0,0].plot(ts,trajp2[:,0][:Nplot],'--',linewidth = lw,color = 'mediumblue')
ax[1,0].plot(ts,trajm2[:,0][:Nplot],'--',linewidth = lw,color = 'mediumblue')


for i in range(2):
    for j in range(2):
        ax[i,j].set_xlabel(r'$t$',fontsize = 15)
        ax[i,j].set_ylabel(r'$x(t)$',fontsize = 15)

ax[0,0].set_title('Pointwise Forecast (clean training data)',fontsize = 15)
ax[1,0].set_title('Measure Forecast (clean training data)',fontsize = 15)
ax[0,1].set_title('Pointwise Forecast (noisy training data)',fontsize = 15)
ax[1,1].set_title('Measure Forecast (noisy training data)',fontsize = 15)

plt.subplots_adjust(wspace = 0.1,hspace=0.5)
plt.show()


e1 = []
e2 = []
e3 = []
e4 = []

for i in range(4):
    FWDm, FWDp = data_m[i][2]*resize, data_p[i][2]*resize
    FWD = data_p[i][3]*resize
    FWDm2, FWDp2 = data_m2[i][2]*resize2, data_p2[i][2]*resize2
    FWD2 = data_p2[i][3]*resize2
    
    
    e1.append(torch.mean((FWD2-FWDp2)**2).detach().numpy())
    e2.append(torch.mean((FWD-FWDp)**2).detach().numpy())
    e3.append(torch.mean((FWD2-FWDm2)**2).detach().numpy())
    e4.append(torch.mean((FWD-FWDm)**2).detach().numpy())
    
e1 = np.array(e1)
e2 = np.array(e2)
e3 = np.array(e3)
e4 = np.array(e4)
print('Pointwise, clean: ', np.mean(e1), ' +/- ', np.std(e1))
print('Pointwise, noisy: ', np.mean(e2), ' +/- ', np.std(e2))
print('Measure, clean: ', np.mean(e3), ' +/- ', np.std(e3))
print('Measure, noisy: ', np.mean(e4), ' +/- ', np.std(e4))



# T1 = data_m[ix][-1]
# for i in range(len(T1)):
#     T1[i] = T1[i].detach().numpy()
# T1 = np.array(T1)
# import random
# a,b,c = np.shape(T1)
# ixs = random.sample(range(0,b),1000)
# for i in range(1000):
#     plt.plot(T1[:,ixs[i],0],T1[:,ixs[i],1])
# plt.show()







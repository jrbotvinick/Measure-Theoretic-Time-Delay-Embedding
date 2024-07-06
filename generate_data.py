import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
from scipy.integrate import solve_ivp
import random 



N = 3 # Number of variables
# IC = np.array([12.47341501, -9.66602587,  0.02707709]) #rossler
IC = np.array([ 2.75132057,  3.32303622, 18.66783115]) #lorenz
# IC = np.array([0.36705869, 0.5850737 , 0.16825897, 0.46289586]) #lotka
TSMax = int(1e6)
t_span = (0,3000)

def van_der_pol(t, z):
    mu = 1
    x, v = z
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x
    return [dxdt, dvdt]

def circle(t, z):
    x, v = z
    dxdt = -v
    dvdt = x
    return [dxdt, dvdt]

def lorenz(t, z):
    sigma,rho,beta = 10,28,8/3
    dzdt = [
        sigma * (z[1] - z[0]),
        z[0] * (rho - z[2]) - z[1],
        z[0] * z[1] - beta * z[2]
    ]
    return dzdt

def lotka(t, z):
    r = np.array([1,.72,1.53,1.27])
    alpha = np.array([[1,1.09,1.52,0],
                     [ 0,1,.44,1.36],
                      [2.33,0,1,.47],
                      [1.21,.51,.35,1]])
    dzdt= r*z*(1-np.matmul(alpha,z))
    return dzdt

def rossler(t,z):
    a,b,c = .1,.1,14
    dzdt = [
        -z[1]-z[2],
       z[0]+a*z[1],
       b+z[2]*(z[0]-c)
    ]
    return dzdt

def chua(t,z):
      alpha = 10.00
      beta = 14.87
      m0 = -.68
      m1 = -1.27
      h = m1*z[0]+.5*(m0-m1)*(np.abs(z[0]+1)-np.abs(z[0]-1))
      dzdt = [
          alpha*(z[1]-h),
         z[0]-z[1]+z[2],
        -beta*z[1]
      ]
      return dzdt
  
def L96(t,x):
    F = 8
    d = np.zeros(N)
    for i in range(N):
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return d

t_eval = np.linspace(t_span[0], t_span[1], TSMax)
sol = solve_ivp(lorenz, t_span, IC, t_eval=t_eval) #Change the system here

ys = sol.y.T
ts = sol.t
_,d = np.shape(ys)

cov = np.zeros((3,3))
cov[0,0] = 0.1
cov[1,1] = 0.1
cov[2,2] = 0.1


noise = np.random.multivariate_normal([0,0,0], cov, size=TSMax)
ys_noise = ys+noise
fig, ax = plt.subplots(subplot_kw={"projection": "3d"},dpi = 300,figsize = (10,5))

ax.plot(ys[:,0],ys[:,1],ys[:,2],linewidth = .1)
ax.plot(ys_noise[:int(1e3),0],ys_noise[:int(1e3),1],ys_noise[:int(1e3),2])
ax.view_init(30,120)

plt.show()

with open("data.p", "wb") as f:
    pickle.dump([ys,ys_noise,ts], f)

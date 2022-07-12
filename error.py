# %%
import math
import matplotlib.pyplot as plt
from nbodykit.cosmology import background

# %%
h0=70
om_m=0.3
om_lam=0.7
p=[-3,0,-2]
a_initial=0.00001
om_k=1-om_m-om_lam
om=[om_m,om_lam,om_k]
par=[om,p]

# %%
def calh(a):
    total=0
    for i in range(3):
        total=total+par[0][i]*a**par[1][i]
    h=h0*math.sqrt(total)
    return h

# %%
def func(a):
    temp=1/a/calh(a)
    return temp*temp*temp

# %%
def rk4(x0,y0,dx):
    k=[0,0,0,0]
    k[0]=func(x0)
    k[1]=func(x0+dx/2)
    k[3]=func(x0+dx)
    return y0+(k[0]+4*k[1]+k[3])*dx/6

# %%
def integration(x_initial,x_final,dx):
    x=x_initial
    y=0
    while x<x_final:
        y=rk4(x,y,dx)
        x=x+dx
    return y

# %%
def growth(a):
    temp=integration(a_initial,a,0.00001)
    return 5*om_m/2*h0*h0*calh(a)*temp

# %%
def error():
    test=background.MatterDominated(om_m,om_lam,om_k,a=None,a_normalize=1.0)
    a=0.01
    temp=growth(1)
    while a<=1:
        plt.scatter(a,growth(a)/temp/test.D1(a)-1)
        a=a+0.01
    plt.show()

# %%
error()

# %%
import math
import random
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# %%
def npy_loader(path):
    return torch.from_numpy(np.load(path))

# %%
model=torch.load("model.pth")
model.eval()

# %%
cosmo=npy_loader("data/cosmo.npy")
cosmo_num=random.randrange(1000)
parameters=torch.ones(1,2)
parameters[:,0]=cosmo[cosmo_num,0]
parameters[:,1]=cosmo[cosmo_num,2]
a=torch.flatten(npy_loader("data/"+str(cosmo_num)+".npy").narrow(0,0,1),0,-1)
a=1/(a+1)
d_data=torch.flatten(npy_loader("data/"+str(cosmo_num)+".npy").narrow(0,1,1),0,-1)
d_test=torch.flatten(model(parameters).clone().detach(),0,-1)
discrepancy=d_test/d_data
plt.subplot(211)
plt.plot(a,d_data,label="Reference")
plt.plot(a,d_test,label="Fitting")
plt.ylabel("D")
plt.title("Omega_m="+str(round(parameters[:,0].item(),3))+", H0="+str(round(parameters[:,1].item(),3)))
plt.legend()
plt.subplot(212)
plt.plot(a,discrepancy,label="Discrepancy")
plt.xlabel("a")
plt.ylabel("Predict/Data")
plt.legend()
plt.savefig("cosmo"+str(cosmo_num)+".png")
for i in range(10):
    plt.clf()
    for j in range(1000):
        cosmo_num=j
        d_data=torch.flatten(npy_loader("data/"+str(cosmo_num)+".npy").narrow(0,1,1),0,-1)
        parameters[:,0]=cosmo[cosmo_num,0]
        parameters[:,1]=cosmo[cosmo_num,2]
        d_test=torch.flatten(model(parameters).clone().detach(),0,-1)
        plt.scatter(cosmo[cosmo_num,0].item(),cosmo[cosmo_num,2].item(),c=d_test[i*28]/d_data[i*28]-1,cmap='coolwarm',vmin=-0.03,vmax=0.03)
    plt.colorbar()
    plt.xlabel("Omega_m")
    plt.ylabel("H0")
    plt.title("Error of Cosmos (a="+str(round(a[i*28].item(),3))+")")
    plt.savefig("error"+str(i)+".png")

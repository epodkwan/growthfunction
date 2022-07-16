# %%
import math
import random
import torch
import numpy as np
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
plt.title("Comparison")
plt.legend()
plt.subplot(212)
plt.plot(a,discrepancy,label="Discrepancy")
plt.xlabel("a")
plt.ylabel("Predict/Data")
plt.savefig("cosmo"+str(cosmo_num)+".png")

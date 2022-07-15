# %%
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

# %%
def npy_loader(path):
    return torch.from_numpy(np.load(path))

# %%
model=torch.load('model.pth')
model.eval()

# %%
cosmo=npy_loader("data/cosmo.npy")
parameters=torch.ones(1,2)
parameters[:,0]=0.7
parameters[:,1]=0.3

# %%
a=torch.flatten(npy_loader("data/0.npy").narrow(0,0,1),0,-1)
a=1/(a+1)
d=torch.flatten(model(parameters).clone().detach(),0,-1)
plt.plot(a,d)
plt.xlabel("a")
plt.ylabel("D")
plt.title("Growth Function")
plt.savefig("growthfunction.png")

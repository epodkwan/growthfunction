# %%
import math
import torch

# %%
model=torch.load('model.pth')
model.eval()

# %%
parameters=torch.ones(1,1)
parameters[:,0]=1
# parameters[:,1]=2
# parameters[:,2]=3
# parameters[:,3]=4
# parameters[:,4]=5

# %%
d=model(parameters).clone().detach()
print(d)

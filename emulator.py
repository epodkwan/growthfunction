# %%
import math
import torch

# %%
model=torch.load('model.pth')
model.eval()

# %%
parameters=torch.ones(1,2)
parameters[:,0]=70
parameters[:,1]=0.3

# %%
d=model(parameters).clone().detach()
print(d)

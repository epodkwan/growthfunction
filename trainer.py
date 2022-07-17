# %%
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
torch.set_num_threads(8)

# %%
def npy_loader(path):
    return torch.from_numpy(np.load(path))

# %%
input_data=npy_loader("data/cosmo.npy")
x_train=torch.cat((input_data.narrow(1,0,1),input_data.narrow(1,2,1)),1).narrow(0,0,900).float()
y_train=torch.ones(900,256)
for i in range(900):
    temp=npy_loader("data/"+str(i)+".npy")
    y_train[i,:]=temp[1,:]

# %%
model=torch.nn.Sequential(
    torch.nn.Linear(2,64),
    torch.nn.ReLU(),
    torch.nn.Linear(64,256),
    torch.nn.ReLU(),
    torch.nn.Linear(256,256),
    torch.nn.ReLU(),
    torch.nn.Linear(256,256),
    torch.nn.Flatten(1,-1)
)

# %%
loss_fn=torch.nn.MSELoss(reduction='sum')
learning_rate=1e-5
epochs=4000000
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)

# %%
for t in range(epochs):
    y_pred=model(x_train)
    loss=loss_fn(y_pred,y_train)
    if t % 100 == 99:
        print(t,loss.item())
        plt.scatter(t,torch.log(loss.detach()),c='b')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("Training ended")
torch.save(model,"model.pth")
plt.xlabel("Iteration")
plt.ylabel("ln(loss)")
plt.title("Loss function")
plt.savefig("loss.png")

# %%
x_test=torch.cat((input_data.narrow(1,0,1),input_data.narrow(1,2,1)),1).narrow(0,900,100).float()
y_validate=torch.ones(100,256)
for i in range(100):
    temp=npy_loader("data/"+str(i+900)+".npy")
    y_validate[i,:]=temp[1,:]
y_test=model(x_test).clone().detach()
print(y_test)
error=abs(y_test/y_validate-1)
print("Max error =",torch.max(error).item()*100,"%")

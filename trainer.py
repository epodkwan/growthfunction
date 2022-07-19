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
x_train=torch.cat((input_data.narrow(1,0,1),input_data.narrow(1,2,1)),1).narrow(0,0,768).float()
y_train=torch.ones(768,256)
for i in range(768):
    temp=npy_loader("data/"+str(i)+".npy")
    y_train[i,:]=temp[1,:]
x_validate=torch.cat((input_data.narrow(1,0,1),input_data.narrow(1,2,1)),1).narrow(0,768,132).float()
y_validate=torch.ones(132,256)
for i in range(132):
    temp=npy_loader("data/"+str(i)+".npy")
    y_validate[i,:]=temp[1,:]

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
loss_fn=torch.nn.MSELoss(reduction='mean')
learning_rate=10
epochs=200
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)

# %%
for i in range(epochs):
    order=torch.randperm(768)
    train_loss=0
    for j in range(6):
        x_batch=x_train[order[128*j:128*(j+1)-1],:]
        y_batch=y_train[order[128*j:128*(j+1)-1],:]
        y_pred=model(x_batch)
        loss=loss_fn(y_pred,y_batch)
        train_loss=train_loss+loss_fn(y_pred,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print((i+1),loss.item())
    train_loss=train_loss/12
    y_pred=model(x_validate)
    validate_loss=loss_fn(y_pred,y_validate)
    plt.scatter((i+1),torch.log(train_loss.detach()),c='b')
    plt.scatter((i+1),torch.log(validate_loss.detach()),c='g')
print("Training ended")
torch.save(model,"model.pth")
plt.xlabel("Epoch")
plt.ylabel("ln(loss)")
plt.title("Loss function")
plt.legend(["Training Loss","Validation Loss"])
plt.savefig("loss.png")

# %%
x_test=torch.cat((input_data.narrow(1,0,1),input_data.narrow(1,2,1)),1).narrow(0,900,100).float()
y_test=torch.ones(100,256)
for i in range(100):
    temp=npy_loader("data/"+str(i+900)+".npy")
    y_test[i,:]=temp[1,:]
y_pred=model(x_test).clone().detach()
print(y_pred)
error=abs(y_pred/y_test-1)
print("Max error =",torch.max(error).item()*100,"%")

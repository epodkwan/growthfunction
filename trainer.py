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
    torch.nn.Linear(2,4),
    torch.nn.ReLU(),
    torch.nn.Linear(4,16),
    torch.nn.ReLU(),
    torch.nn.Linear(16,256),
    torch.nn.Flatten(1,-1)
)

# %%
loss_fn=torch.nn.MSELoss(reduction='sum')
learning_rate=1e-6
epochs=5000
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)

# %%
for t in range(epochs):
    y_pred=model(x_train)
    loss=loss_fn(y_pred,y_train)
    if t % 100 == 99:
        print(t,loss.item())
        plt.scatter(t,torch.log(loss.detach()),c="b")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("Training ended")
torch.save(model,'model.pth')
plt.xlabel("Iteration")
plt.ylabel("ln(loss)")
plt.title("Loss function")
plt.show()

# %%
x_test=torch.cat((input_data.narrow(1,0,1),input_data.narrow(1,2,1)),1).narrow(0,900,100).float()
true_y=torch.ones(100)
for i in range(100):
    true_y[i]=normalize(test_x[i],par,par)
test_y=model(x_test).clone().detach()
print(test_y)
error=abs(test_y/true_y-1)
print("Max error =",torch.max(error).item()*100,"%")
plt.subplot(211)
plt.plot(test_x,true_y,label='Reference')
plt.plot(test_x,test_y,label='Fitting')
plt.ylabel("D")
plt.title("Comparison")
plt.legend()
plt.subplot(212)
plt.plot(test_x,error,label='Error')
plt.xlabel("a")
plt.ylabel("Error (*100%)")
plt.title("Error")
plt.show()

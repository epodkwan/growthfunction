# %%
import torch
import math
import matplotlib.pyplot as plt
torch.set_num_threads(8)

# %%
x=torch.linspace(-math.pi,math.pi,2000)
a=1
b=2
c=3
y=a*x*x+b*x+c
data=torch.ones(2000,1)
# data[:,0]=a
# data[:,1]=b
# data[:,2]=c
data[:,0]=x

# %%
model=torch.nn.Sequential(
    torch.nn.Linear(1,16),
    torch.nn.ReLU(),
    torch.nn.Linear(16,256),
    torch.nn.ReLU(),
    torch.nn.Linear(256,1),
    torch.nn.Flatten(0, 1)
)

# %%
loss_fn=torch.nn.MSELoss(reduction='sum')
learning_rate=1e-6
epochs=10000
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)

# %%
for t in range(epochs):
    y_pred=model(data)
    loss=loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t,loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("Training ended")

# %%
yf=model(data).clone().detach()
print(yf)
error=abs(yf/y-1)
print("Max error =",torch.max(error).item()*100,"%")
ax1=plt.subplot(211)
plt.plot(x,y,label='Reference')
plt.plot(x,yf,label='Fitting')
plt.ylabel("y")
plt.title("Comparison")
plt.legend()
plt.subplot(212,sharex=ax1)
plt.plot(x,error,label='Error')
plt.xlabel("x")
plt.ylabel("Error")
plt.title("Error")
plt.legend()
plt.show()

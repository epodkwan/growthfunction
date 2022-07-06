# %%
import torch
import math

# %%
x=torch.linspace(-math.pi,math.pi,2000)
a=1
b=2
c=3
y=a*x*x+b*x+c
data=torch.ones(2000,4)
data[:,0]=a
data[:,1]=b
data[:,2]=c
data[:,3]=x

# %%
model=torch.nn.Sequential(
    torch.nn.Linear(4,16),
    torch.nn.ReLU(),
    torch.nn.Linear(16,1),
    torch.nn.Flatten(0, 1)
)
loss_fn=torch.nn.MSELoss(reduction='sum')
learning_rate=1e-5

# %%
for t in range(5000):
    y_pred=model(data)
    loss=loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t,loss.item())
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param-=learning_rate*param.grad
print("Training ended")

# %%
yf=model(data).clone().detach()
print(yf)
error=abs(yf/y-1)
print("Max error =",torch.max(error).item()*100,"%")

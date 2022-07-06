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
for i in range(2000):
    data[i,0]=a
    data[i,1]=b
    data[i,2]=c
    data[i,3]=x[i]

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
for i in range(2000):
    error=abs(yf[i]/(a*x[i]*x[i]+b*x[i]+c)-1)
print("Max error=",torch.max(error))

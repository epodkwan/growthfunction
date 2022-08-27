# %%
import math
import torch
import matplotlib.pyplot as plt
torch.set_num_threads(8)

# %%
def func(x):
    a=-1
    b=2
    c=-5
    return a*x*x+b*x+c

# %%
x_data=torch.linspace(-math.pi,math.pi,2000)
y_data=func(x_data)
data=torch.ones(2000,1)
# data[:,0]=a
# data[:,1]=b
# data[:,2]=c
data[:,0]=x_data

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
epochs=5000
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)

# %%
for t in range(epochs):
    y_pred=model(data)
    loss=loss_fn(y_pred,y_data)
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
test=torch.ones(600,1)
test_x=torch.linspace(-3,3,600)
test[:,0]=test_x
true_y=func(test_x)
test_y=model(test).clone().detach()
print(test_y)
error=abs(test_y/true_y-1)
print("Max error =",torch.max(error).item()*100,"%")
plt.subplot(211)
plt.plot(test_x,true_y,label='Reference')
plt.plot(test_x,test_y,label='Fitting')
plt.ylabel("y")
plt.title("Comparison")
plt.legend()
plt.subplot(212)
plt.plot(test_x,error,label='Error')
plt.xlabel("x")
plt.ylabel("Error (*100%)")
plt.title("Error")
plt.show()

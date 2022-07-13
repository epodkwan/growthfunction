# %%
import math
import torch
import matplotlib.pyplot as plt
torch.set_num_threads(8)

# %%
h0=70
om_m=0.3
om_lam=0.7
p=[-3,0,-2]
a_initial=0.00001
om_k=1-om_m-om_lam
om=[om_m,om_lam,om_k]
par=[om,p]

# %%
def calh(a,par):
    total=0
    for i in range(3):
        total=total+par[0][i]*a**par[1][i]
    h=h0*math.sqrt(total)
    return h

# %%
def func(a,par):
    temp=1/a/calh(a,par)
    return temp*temp*temp

# %%
def rk4(x0,y0,dx,par):
    k=[0,0,0,0]
    k[0]=func(x0,par)
    k[1]=func(x0+dx/2,par)
    k[3]=func(x0+dx,par)
    return y0+(k[0]+4*k[1]+k[3])*dx/6

# %%
def integration(x_initial,x_final,dx,par):
    x=x_initial
    y=0
    while x<x_final:
        y=rk4(x,y,dx,par)
        x=x+dx
    return y

# %%
def growth(a,par):
    temp=integration(a_initial,a,0.00001,par)
    return 5*par[0][0]/2*h0*h0*calh(a,par)*temp

# %%
def normalize(a,par1,par0):
    return growth(a,par1)/growth(1,par0)

# %%
x_data=torch.linspace(0.01,1,200)
y_data=torch.ones(200)
for i in range(200):
    y_data[i]=normalize(x_data[i],par,par)
plt.plot(x_data,y_data)
plt.show()
data=torch.ones(200,5)
data[:,0]=om_m
data[:,1]=om_lam
data[:,2]=om_k
data[:,3]=h0
data[:,4]=x_data

# %%
model=torch.nn.Sequential(
    torch.nn.Linear(5,32),
    torch.nn.ReLU(),
    torch.nn.Linear(32,512),
    torch.nn.ReLU(),
    torch.nn.Linear(512,1),
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
test=torch.ones(100,5)
test_x=torch.linspace(0.01,1,100)
test[:,0]=om_m
test[:,1]=om_lam
test[:,2]=om_k
test[:,3]=h0
test[:,4]=test_x
true_y=torch.ones(100)
for i in range(100):
    true_y[i]=normalize(test_x[i],par,par)
test_y=model(test).clone().detach()
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

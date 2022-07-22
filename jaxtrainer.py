# %%
import math
import jax
import numpy as np
import jax.numpy as jnp
from jax import grad,jit,vmap,random
from jax.experimental import optimizers as jax_opt
import matplotlib.pyplot as plt

# %%
def npy_loader(path):
    return jnp.load(path)

# %%
input_data=npy_loader("data/cosmo.npy")
x_train=torch.cat((input_data.narrow(1,0,1),input_data.narrow(1,2,1)),1).narrow(0,0,800).float()
y_train=torch.ones(800,256)
for i in range(800):
    temp=npy_loader("data/"+str(i)+".npy")
    y_train[i,:]=temp[1,:]
x_validate=torch.cat((input_data.narrow(1,0,1),input_data.narrow(1,2,1)),1).narrow(0,800,100).float()
y_validate=torch.ones(100,256)
for i in range(100):
    temp=npy_loader("data/"+str(i+800)+".npy")
    y_validate[i,:]=temp[1,:]

# %%
def random_layer_params(m,n,key,scale=1e-2):
    w_key,b_key=random.split(key)
    return scale*random.normal(w_key,(n,m)),scale*random.normal(b_key,(n,))

# %%
def init_network_params(sizes,key):
    keys=random.split(key,len(sizes))
    return [random_layer_params(m,n,k) for m,n,k in zip(sizes[:-1],sizes[1:],keys)]

# %%
layer_sizes=[2,64,256,256,256]
learning_rate=1e1
epochs=30000
params=init_network_params(layer_sizes,random.PRNGKey(0))

# %%
def predict(params,data):
    activations=data
    for w,b in params[:-1]:
        outputs=jnp.dot(w,activations)+b
        activations=jax.nn.relu(outputs)
    final_w,final_b=params[-1]
    return jnp.dot(final_w,activations)+final_b

# %%
batched_predict=vmap(predict,in_axes=(None,0))

# %%
def mse_loss(params,loss_data):
    X_tbatch,targets=loss_data
    preds=batched_predict(params,X_tbatch)
    diff=preds-targets 
    return jnp.mean(diff*diff)

#%%
opt_init,opt_update,get_params=jax_opt.adam(1e-3)
opt_state=opt_intial(params)

@jit
def train_step(step_i,opt_state,loss_data):
    net_params=get_params(opt_state)
    loss=mse_loss(params,data)
    loss,grads=value_and_grad(mse_loss,argnums=0)(net_params,loss_data)
    return loss,opt_update(step_i,grads,opt_state)

# %%
for epoch in range(num_epochs):
    for x, y in training_generator:
        params = update(params, x, y)

# %%
loss_fn=torch.nn.MSELoss(reduction='mean')
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)

# %%
for i in range(epochs):
    order=torch.randperm(800)
    train_loss=0
    for j in range(25):
        x_batch=x_train[order[32*j:32*(j+1)-1],:]
        y_batch=y_train[order[32*j:32*(j+1)-1],:]
        y_pred=model(x_batch)
        loss=loss_fn(y_pred,y_batch)
        train_loss=train_loss+loss_fn(y_pred,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if i % 100 == 99:
        print((i+1),loss.item())
        train_loss=train_loss/25
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

# %%
import math
import jax
import numpy as np
import jax.numpy as jnp
from jax import grad,jit,vmap,random
import jax.example_libraries.optimizers as jax_opt
import matplotlib.pyplot as plt

# %%
def npy_loader(path):
    return jnp.load(path)

# %%
input_data=npy_loader("data/cosmo.npy")
x_train=jnp.stack((input_data[0:800,0],input_data[0:800,2]),axis=1)
y_train=jnp.ones((800,256))
for i in range(800):
    temp=npy_loader("data/"+str(i)+".npy")
    y_train.at[i,:].set(temp[1,:])
x_validate=jnp.stack((input_data[800:900,0],input_data[800:900,2]),axis=1)
y_validate=jnp.ones((100,256))
for i in range(100):
    temp=npy_loader("data/"+str(i+800)+".npy")
    y_validate.at[i,:].set(temp[1,:])

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
epochs=1000
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
@jit
def mse_loss(params,x,y_ref):
    preds=batched_predict(params,x)
    diff=preds-y_ref
    return jnp.mean(diff*diff)

#%%
opt_init,opt_update,get_params=jax_opt.adam(learning_rate)
opt_state=opt_init(params)

# %%
@jit
def train_step(step_i,opt_state,x,y_ref):
    params=get_params(opt_state)
    loss,grads=jax.value_and_grad(mse_loss,argnums=0)(params,x,y_ref)
    return loss,opt_update(step_i,grads,opt_state)

# %%
order=jnp.arange(800)
for i in range(epochs):
    random.permutation(random.PRNGKey(i),order)
    train_loss=0
    for j in range(25):
        x_batch=x_train[order[32*j:32*(j+1)-1],:]
        y_batch=y_train[order[32*j:32*(j+1)-1],:]
        loss,opt_state=train_step(learning_rate,opt_state,x_batch,y_batch)
        train_loss=train_loss+loss
    if i % 100 == 99:
        print((i+1),train_loss)
        train_loss=train_loss/25
        validate_loss=mse_loss(params,x_validate,y_validate)
        plt.scatter((i+1),jnp.log(train_loss),c='b')
        plt.scatter((i+1),jnp.log(validate_loss),c='g')
print("Training ended")
plt.xlabel("Epoch")
plt.ylabel("ln(loss)")
plt.title("Loss function")
plt.legend(["Training Loss","Validation Loss"])
plt.savefig("loss.png")

# %%
x_test=jnp.stack((input_data[900:1000,0],input_data[900:1000,2]),axis=1)
y_test=jnp.ones((100,256))
for i in range(100):
    temp=npy_loader("data/"+str(i+900)+".npy")
    y_test.at[i,:].set(temp[1,:])
y_pred=batched_predict(params,x_test)
print(y_pred)
error=abs(y_pred/y_test-1)
print("Max error =",jnp.max(error)*100,"%")

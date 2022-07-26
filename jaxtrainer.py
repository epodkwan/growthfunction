# %%
from typing import Sequence
import jax
import optax
import numpy as np
import jax.numpy as jnp
from jax import jit,random
from flax import linen as nn
import matplotlib.pyplot as plt

# %%
def npy_loader(path):
    return jnp.load(path)

# %%
input_data=npy_loader("data/cosmo.npy")
input_result=npy_loader("combined.npy")
x_train=jnp.stack((input_data[0:800,0],input_data[0:800,2]),axis=1)
y_train=input_result[0:800,:]
x_validate=jnp.stack((input_data[800:900,0],input_data[800:900,2]),axis=1)
y_validate=input_result[800:900,:]

# %%
class SimpleMLP(nn.Module):
    features:Sequence[int]

    @nn.compact
    def __call__(self,inputs):
        x=inputs
        for i,feat in enumerate(self.features):
            x=nn.Dense(feat)(x)
            if i != len(self.features)-1:
                x=nn.relu(x)
        return x

# %%
layer_sizes=[64,256,256,256]
learning_rate=1e1
epochs=1000
model=SimpleMLP(features=layer_sizes)
temp=jnp.ones(2)
params=model.init(random.PRNGKey(0),temp)

# %%
@jit
def mse_loss(params,x,y_ref):
    preds=model.apply(params,x)
    diff=preds-y_ref
    return jnp.mean(diff*diff)

#%%
tx=optax.adam(learning_rate=learning_rate)
opt_state=tx.init(params)

# %%
@jit
def train_step(params,opt_state,x,y_ref):
    loss,grads=jax.value_and_grad(mse_loss,argnums=0)(params,x,y_ref)
    updates,opt_state=tx.update(grads,opt_state)
    params=optax.apply_updates(params,updates)
    return loss,params,opt_state

# %%
order=jnp.arange(800)
for i in range(epochs):
    random.permutation(random.PRNGKey(i),order)
    train_loss=0
    for j in range(25):
        x_batch=x_train[order[32*j:32*(j+1)-1],:]
        y_batch=y_train[order[32*j:32*(j+1)-1],:]
        loss,params,opt_state=train_step(params,opt_state,x_batch,y_batch)
        train_loss=train_loss+loss
    if i % 10 == 9:
        train_loss=train_loss/25
        validate_loss=mse_loss(params,x_validate,y_validate)
        print((i+1),validate_loss)
        plt.scatter((i+1),jnp.log(train_loss),c='b')
        plt.scatter((i+1),jnp.log(validate_loss),c='g')
print("Training ended")
jnp.save("model.npy",params)
plt.xlabel("Epoch")
plt.ylabel("ln(loss)")
plt.title("Loss function")
plt.legend(["Training Loss","Validation Loss"])
plt.savefig("loss.png")

# %%
x_test=jnp.stack((input_data[900:1000,0],input_data[900:1000,2]),axis=1)
y_test=input_result[900:1000,:]
y_pred=model.apply(params,x_test)
print(y_pred)
error=abs(y_pred/y_test-1)
print("Max error =",jnp.max(error)*100,"%")

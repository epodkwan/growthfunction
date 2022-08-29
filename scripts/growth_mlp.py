import random
import statistics
from typing import Sequence
import jax
import numpy as np
import jax.numpy as jnp
from jax import jit,vmap
from flax import linen as nn
from flax.training import train_state,checkpoints
import matplotlib
import matplotlib.pyplot as plt


class SimpleMLP(nn.Module):
    features:Sequence[int]
    nodes:int

    @nn.compact
    def __call__(self,inputs):
        x=inputs
        for i,feat in enumerate(self.features):
            x=nn.Dense(feat)(x)
            x=jnp.sin(x)
        t=nn.Dense(nodes-1)(x)
        c=nn.Dense(nodes+1)(x)
        t=jnp.concatenate([jnp.zeros((t.shape[0],4)),jnp.cumsum(jax.nn.softmax(t),axis=1),jnp.ones((t.shape[0],3))],axis=1)
        c=jnp.concatenate([jnp.zeros((c.shape[0],1)),c],axis=1)
        return t,c


@jit
def _deBoorVectorized(x,t,c):
    p=3
    k=jnp.digitize(x,t)-1
    d=[c[j+k-p] for j in range(0,p+1)]
    for r in range(1,p+1):
        for j in range(p,r-1,-1):
            alpha=(x-t[j+k-p])/(t[j+1+k-r]-t[j+k-p])
            d[j]=(1.0-alpha)*d[j-1]+alpha*d[j]
    return d[p]
deBoor = vmap(_deBoorVectorized,in_axes=(None,0,0))


class Growth_MLP():

    def __init__(self, model, params):
        self.model = model
        self.params = params
        

    def __call__(self, cosmo, a):
        @jit 
        def _growth(cosmo, a):
            t,c = self.model.apply(self.params, cosmo)
            g = deBoor(jnp.clip(a,0,0.99999),t,c)
            return g
        return _growth(cosmo, a)



layer_sizes = [64,64]
nodes = 16
model = SimpleMLP(features=layer_sizes,nodes=nodes)
params = checkpoints.restore_checkpoint(ckpt_dir="./checkpoint_0",target=None)['params']
growth_fn = Growth_MLP(model, params)


cosmo = np.array([0.1, 0.2, 0.3]).reshape(-1, 1)
a = np.linspace(0.001, 0.999, 100)
g = growth_fn(cosmo, a)
print(a.shape, g.shape)
plt.plot(a, g)
plt.grid(which='both')
plt.savefig('tmp2.png')
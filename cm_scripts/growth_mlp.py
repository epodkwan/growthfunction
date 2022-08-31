from functools import partial
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import random
import statistics
from typing import Sequence
import jax
import numpy as np
import jax.numpy as jnp
from jax import jit,vmap
from flax import linen as nn
from flax.training import train_state,checkpoints
import time

import matplotlib
import matplotlib.pyplot as plt


#
class Simple_MLP(nn.Module):
    features:Sequence[int]
    nodes:int

    @nn.compact
    def __call__(self, inputs):
        x=inputs
        for i, feat in enumerate(self.features):
            x=nn.Dense(feat)(x)
            x=nn.elu(x)
        t=nn.Dense(self.nodes-2)(x)
        c=nn.Dense(self.nodes+1)(x)
        t=jnp.concatenate([jnp.zeros((t.shape[0], 4)), jnp.cumsum(jax.nn.softmax(t), axis=1), jnp.ones((t.shape[0], 4))], axis=1)
        c=jnp.concatenate([jnp.zeros((c.shape[0], 1)), c], axis=1)
        return t, c

# class Simple_MLP(nn.Module):
#     features:Sequence[int]
#     nodes:int

#     @nn.compact
#     def __call__(self,inputs):
#         x = inputs
#         for i,feat in enumerate(self.features):
#             x = nn.Dense(feat)(x)
#             x = jnp.sin(x)
#         t = nn.Dense(self.nodes-1)(x)
#         c = nn.Dense(self.nodes+1)(x)
#         t = jnp.concatenate([jnp.zeros((t.shape[0],4)),jnp.cumsum(jax.nn.softmax(t),axis=1),jnp.ones((t.shape[0],3))],axis=1)
#         c = jnp.concatenate([jnp.zeros((c.shape[0],1)),c],axis=1)
#         return t,c

#
@jit
def _deBoorVectorized(x,t,c):
    print("compile boor")
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

    @partial(jit, static_argnums=(0,))
    def _growth(self, cosmo, a, params):

        print('compile')
        reshape = False
        if len(cosmo.shape) == 1: 
            reshape = True
            cosmo = jnp.reshape(cosmo, (1, -1))
        t,c = model.apply(params, cosmo)
        g = deBoor(jnp.clip(a,0,0.99999),t,c)
        g1 = deBoor(jnp.array([0.99999]),t,c)
        g = g/g1
        if reshape: 
            return g[0]
        else: 
            return g

               
    def __call__(self, cosmo, a, order=1, deriv=0):

        key = "{}{}".format(order, deriv)
        params = self.params[key]
        return self._growth(cosmo, a, params)




if __name__=="__main__":


    layer_sizes = [64,64]
    nodes = 8 
    model = Simple_MLP(features=layer_sizes,nodes=nodes)
    #params = checkpoints.restore_checkpoint(ckpt_dir="./checkpoint_0",target=None)['params']
    #params = checkpoints.restore_checkpoint(ckpt_dir="./d2_1order_checkpoint_0",target=None)['params']
    params = {}
    for order in range(1, 3):
        for deriv in range(3):
            key = "{}{}".format(order, deriv)
            params[key] = checkpoints.restore_checkpoint(ckpt_dir="./d%d_%dorder_checkpoint_0"%(order, deriv),target=None)['params']
    print(len(params))
     
    growth_fn = Growth_MLP(model, params)

    cosmo = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5]).reshape(-1, 1)
    a = jnp.linspace(0.001, 0.99, 100)

    order, deriv = 1, 0
    g = growth_fn(cosmo, a, order, deriv)

    start = time.time()
    for _ in range(20): g = growth_fn(cosmo, a, order, deriv)
    print("Time taken : ", time.time() - start)


    print(a.shape, g.shape)
    for ia in range(cosmo.size):
        plt.plot(a, g[ia])
    plt.grid(which='both')
    plt.savefig('tmp2.png')


    cosmo = np.array([0.3]).reshape(-1, 1)
    a = np.linspace(0.001, 0.99, 100)

    for order in range(1, 3):
        for deriv in range(0, 3):
            print(order, deriv)
            g = growth_fn(cosmo, a, order, deriv)
            plt.plot(a, g[0])
    plt.grid(which='both')
    plt.savefig('tmp.png')

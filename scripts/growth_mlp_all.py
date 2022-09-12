from functools import partial
import random
import statistics
from typing import Sequence
import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from flax import linen as nn
from flax.training import train_state, checkpoints
import matplotlib
import matplotlib.pyplot as plt

import time
import sys, os
sys.path.append('../src/')
from growth_mlp_unnorm import Growth_MLP

#
class SimpleMLP(nn.Module):
    features:Sequence[int]
    nodes:int

    @nn.compact
    def __call__(self, inputs):
        x=inputs
        for feat in self.features[:-2]:
            x=nn.Dense(feat)(x)
            x=nn.elu(x)
        t=nn.Dense(nodes-2)(nn.elu(nn.Dense(self.features[-2])(x)))
        c=nn.Dense(nodes+1)(nn.elu(nn.Dense(self.features[-1])(x)))
        t=jnp.concatenate([jnp.zeros((t.shape[0], 4)), jnp.cumsum(jax.nn.softmax(t), axis=1), jnp.ones((t.shape[0], 4))], axis=1)
        c=jnp.concatenate([jnp.zeros((c.shape[0], 1)), c], axis=1)
        return t, c



if __name__ == "__main__":
    layer_sizes = [64, 64, 64]
    nodes = 8 
    model = Simple_MLP(features=layer_sizes,nodes=nodes)
    params = {}
    for order in range(1, 3):
        for deriv in range(3):
            key = "{}{}".format(order, deriv)
            params[key] = checkpoints.restore_checkpoint(ckpt_dir="../checkpoints//d%d_%dorder_checkpoint_0"%(order, deriv),target=None)['params']
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

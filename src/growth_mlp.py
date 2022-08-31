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

        print('compile growth')
        reshape = False
        if len(cosmo.shape) == 1: 
            reshape = True
            cosmo = jnp.reshape(cosmo, (1, -1))
        t,c = self.model.apply(params, cosmo)
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




##Test MLP growth function
##
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import time

from jax import jit, custom_vjp, ensure_compile_time_eval, grad
import jax.numpy as jnp
from flax.training import checkpoints

from conf import Configuration
from cosmology import Cosmology, SimpleLCDM, growth_integ

from skopt.sampler import Lhs

from growth_mlp import Simple_MLP, Growth_MLP

#########
### Setup
layer_sizes = [64,64]
nodes = 16
model = Simple_MLP(features=layer_sizes,nodes=nodes)
params = checkpoints.restore_checkpoint(ckpt_dir="./checkpoint_0",target=None)['params']
growth_fn = Growth_MLP(model, params)

print("Checkpoint restored")
##########################################
##Objectives
#a_test = np.arange(1e-3, 1-1e-3, 10)
a_test = np.logspace(-2, 0.)
@jit
def D(a, cosmo):
    conf = cosmo.conf 
    a = jnp.asarray(a, dtype=conf.cosmo_dtype)
    D = a * jnp.interp(a, conf.growth_a, cosmo.growth[0][0])
    D1 = 1 * jnp.interp(1., conf.growth_a, cosmo.growth[0][0])
    return D/D1


@jit
def objective(params, conf):
    omegam = params
    cosmo = SimpleLCDM(conf, Omega_m=omegam)
    cosmo = growth_integ(cosmo)
    obj = sum(D(jnp.asarray(a_test), cosmo)**2)
    return obj


@jit
def D_mlp(a, cosmo):
    omegam = jnp.array([cosmo.Omega_m])
    print(omegam)
    a = jnp.asarray(a, dtype=conf.cosmo_dtype)
    g = growth_fn(omegam, a)
    return g


@jit
def objective_mlp(params, conf):
    omegam = params
    cosmo = SimpleLCDM(conf, Omega_m=omegam)
    omegam = jnp.array([cosmo.Omega_m])
    print(omegam)
    obj = sum(growth_fn(omegam, jnp.asarray(a_test))**2)
    return obj



def calculate_gradient():
    
    omegam = 0.3
    omegak = 0.0
    w0 = -1.0
    wa = 0.0
    nc = 32
    cell_size = 8
    growth_anum = 512
    growth_a = jnp.linspace(0., 1., growth_anum) 
    conf = Configuration(cell_size=cell_size, mesh_shape=(nc,)*3, growth_anum=growth_anum) 

    #params = [omegam, omegak, w0, wa]
    params = omegam
    print("objective a: ", objective(params, conf))
    print()
    obj_grad_a = jit(grad(objective, argnums=(0)))
    print("gradient a : ", obj_grad_a(params, conf))

    print("objective mlp: ", objective_mlp(params, conf))
    print()
    obj_grad_mlp = jit(grad(objective_mlp, argnums=(0)))
    print("gradient mlp : ", obj_grad_mlp(params, conf))
    
    start = time.time()
    p, g0 = [], []
    for _ in range(100):
        params = np.random.uniform(0.15, 0.4)
        p.append(params)
        g0.append(obj_grad_a(params, conf))
    print("Time taken for ode: ", time.time() - start)

    start = time.time()
    g1 = []
    for i in range(100):
        params = np.random.uniform(0.15, 0.4)
        g1.append(obj_grad_mlp(p[i], conf))
    print("Time taken for mlp: ", time.time() - start)

    print(np.array(g0))
    print(np.array(g1))
    print(np.allclose(np.array(g0), np.array(g1), atol=1e-3, rtol=1e-2))

def test_mlp():

    cosmo = np.array([0.1, 0.2, 0.3]).reshape(-1, 1)
    a = np.linspace(0.001, 0.9999, 100)
    g = growth_fn(cosmo, a)
    print(a.shape, g.shape)
    plt.figure()
    for ia in range(a.size):
        plt.plot(a, np.log10(g[ia]))
    plt.grid(which='both')
    plt.savefig('tmp2.png')
    plt.close()

if __name__=="__main__":
    
    calculate_gradient()
    #test_mlp()

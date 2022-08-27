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

from conf import Configuration
from cosmology import Cosmology, SimpleLCDM, growth_integ

from skopt.sampler import Lhs

##########################################
##Objectives
@jit
def D(a, cosmo, order=1, deriv=0):
    """Evaluate interpolation of (LPT) growth function or derivative, the n-th
    derivatives of the m-th order growth function :math:`\mathrm{d}^n D_m /
    \mathrm{d}\ln^n a`, at given scale factors. Growth functions are normalized at the
    matter dominated era instead of today.

    Parameters
    ----------
    a : array_like
        Scale factors.
    cosmo : Cosmology
    conf : Configuration
    order : int in {1, 2}, optional
        Order of growth function.
    deriv : int in {0, 1, 2}, optional
        Order of growth function derivatives.

    Returns
    -------
    D : jax.numpy.ndarray of conf.cosmo_dtype
        Growth functions or derivatives.

    Raises
    ------
    ValueError
        If ``cosmo.growth`` table is empty.

    """
    conf = cosmo.conf 
    a = jnp.asarray(a, dtype=conf.cosmo_dtype)
    D = a * jnp.interp(a, conf.growth_a, cosmo.growth[order-1][deriv])
    return D



def sample_cosmology(nparams, seed, nsamples): 
    ''' sample cosmologies from latin hypercube in the ranges specified
    '''
    
    Omega_m = [0.1, 0.5]

    if nparams == 1:
        bounds = [Omega_m]
        cosmologies = np.array(Lhs().generate(bounds, nsamples, random_state=seed))
        return cosmologies
        
    elif nparams == 4:

        Omega_k = [0.0, 0.2]
        w_0 = [-1.1, -0.9]
        w_a = [-0.1, 0.1]
        bounds = [Omega_m, Omega_k, w_0, w_a]
        cosmologies = np.array(Lhs().generate(bounds, nsamples, random_state=seed))
        return cosmologies




def check_variations():
    '''A test code to vary cosmology parameters one at a time and see how growth function changes
    '''
    #
    nc = 32
    cell_size = 8
    growth_anum = 512
    growth_a = jnp.linspace(0., 1., growth_anum) 
    conf = Configuration(cell_size=cell_size, mesh_shape=(nc,)*3, growth_anum=growth_anum) 

    cosmo = SimpleLCDM(conf, Omega_k=0., w_0=-1., w_a=0.)
    cosmo = growth_integ(cosmo)
    a_test = np.logspace(-4, 0, 128)
    g = []

    print("Fiducial growth")
    d = D(a_test, cosmo)
    g.append(d)
    
    print("Vary cosmology")
    cosmo2 = SimpleLCDM(conf, A_s_1e9 = cosmo.A_s_1e9*1.05)
    cosmo2 = growth_integ(cosmo2)
    d = D(a_test, cosmo2)
    g.append(d)
    if np.allclose(g[0], g[-1]): pass
    else: print("vary a_s" )

    cosmo2 = SimpleLCDM(conf, n_s = cosmo.n_s*1.05)
    cosmo2 = growth_integ(cosmo2)
    d = D(a_test, cosmo2)
    g.append(d)
    if np.allclose(g[0], g[-1]): pass
    else: print("vary n_s" )

    cosmo2 = SimpleLCDM(conf, Omega_m = cosmo.Omega_m*1.05)
    cosmo2 = growth_integ(cosmo2)
    d = D(a_test, cosmo2)
    g.append(d)
    if np.allclose(g[0], g[-1]): pass
    else: print("vary Omega_m" )

    cosmo2 = SimpleLCDM(conf, Omega_b = cosmo.Omega_b*1.05)
    cosmo2 = growth_integ(cosmo2)
    d = D(a_test, cosmo2)
    g.append(d)
    if np.allclose(g[0], g[-1]): pass
    else: print("vary Omega_b" )

    cosmo2 = SimpleLCDM(conf, h = cosmo.h*1.05)
    cosmo2 = growth_integ(cosmo2)
    d = D(a_test, cosmo2)
    g.append(d)
    if np.allclose(g[0], g[-1]): pass
    else: print("vary h" )

    cosmo2 = SimpleLCDM(conf, Omega_k = cosmo.Omega_k + 0.05)
    cosmo2 = growth_integ(cosmo2)
    d = D(a_test, cosmo2)
    g.append(d)
    if np.allclose(g[0], g[-1]): pass
    else: print("vary Omega_k" )

    cosmo2 = SimpleLCDM(conf, w_0 = cosmo.w_0*1.05)
    cosmo2 = growth_integ(cosmo2)
    d = D(a_test, cosmo2)
    g.append(d)
    if np.allclose(g[0], g[-1]): pass
    else: print("vary w_0" )

    cosmo2 = SimpleLCDM(conf, w_a = cosmo.w_a + 0.05)
    cosmo2 = growth_integ(cosmo2)
    d = D(a_test, cosmo2)
    g.append(d)
    if np.allclose(g[0], g[-1]): pass
    else: print("vary w_a" )

    for i in range(len(g)):
        plt.plot(g[i]/g[0], label=i)
        
    plt.legend()
    plt.grid(which='both')
    plt.savefig('tmp.png')
    print()




def gen_data():

    nc = 32
    cell_size = 8
    growth_anum = 512
    growth_a = jnp.linspace(0., 1., growth_anum) 
    conf = Configuration(cell_size=cell_size, mesh_shape=(nc,)*3, growth_anum=growth_anum) 
    a_test = np.logspace(-4, 0, 256)

    nparams = 4
    cosmologies = sample_cosmology(nparams=nparams, seed=0, nsamples=10)
    print(cosmologies.shape)
    if nparams == 1: 
        omegak, w0, wa = 0., -1., 0.


    growth_integ_jit = jit(growth_integ)
    path = './data/growth_function/'
    for i, cc in enumerate(cosmologies):
        if i%1 == 0: print('iteration %d'%i)
        if nparams  == 1: omegam = cc[0]
        elif nparams == 4 : omegam, omegak, w0, wa = cc
        cosmo = SimpleLCDM(conf, Omega_m=omegam, Omega_k=omegak, w_0=w0, w_a=wa)
        cosmo = growth_integ_jit(cosmo)
        d = D(a_test, cosmo)

        #np.save(path + '%d'%i, np.array([a_test, d]))
    #np.save(path + 'cosmo', cosmologies)


if __name__=="__main__":
    
    check_variations()
    #gen_data()

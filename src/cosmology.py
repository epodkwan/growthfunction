from dataclasses import field, fields
from functools import partial
from operator import add, sub
from typing import Optional, Union

import numpy as np
from jax import value_and_grad, ensure_compile_time_eval

import jax.numpy as jnp
from jax.tree_util import tree_map
from jax.experimental.ode import odeint
from jax.lax import scan

from tree_util import pytree_dataclass
from conf import Configuration


FloatParam = Union[float, jnp.ndarray]


# TODO really a bad idea to add leading or trailing underscores to Omega_b, w_a, etc?
@partial(pytree_dataclass, aux_fields="conf", frozen=True)
class Cosmology:
    """Cosmological and configuration parameters, "immutable" as a frozen dataclass.

    Extension parameters have None as default values. Set them explicitly to activate
    them, e.g., for their gradients.

    Linear operators (addition, subtraction, and scalar multiplication) are defined for
    Cosmology tangent and cotangent vectors.

    Float parameters are converted to JAX arrays of conf.cosmo_dtype at instantiation,
    to avoid possible JAX weak type problems.

    Parameters
    ----------
    conf : Configuration
        Configuration parameters.
    A_s_1e9 : float or jax.numpy.ndarray
        Primordial scalar power spectrum amplitude, multiplied by 1e9.
    n_s : float or jax.numpy.ndarray
        Primordial scalar power spectrum spectral index.
    Omega_m : float or jax.numpy.ndarray
        Total matter density parameter today.
    Omega_b : float, jax.numpy.ndarray, or None
        Baryonic matter density parameter today. If None, dark matter only.
    Omega_k : None, float, or jax.numpy.ndarray, optional
        Spatial curvature density parameter today. If None (default), Omega_k is 0.
    w_0 : None, float, or jax.numpy.ndarray, optional
        Dark energy equation of state (0th order) parameter. If None (default), w is -1.
    w_a : None, float, or jax.numpy.ndarray, optional
        Dark energy equation of state (linear) parameter. If None (default), w_a is 0.
    h : float or jax.numpy.ndarray
        Hubble constant in unit of 100 [km/s/Mpc].

    """

    conf: Configuration = field(repr=False)

    A_s_1e9: FloatParam
    n_s: FloatParam
    Omega_m: FloatParam
    Omega_b: FloatParam
    h: FloatParam

    Omega_k: Optional[FloatParam] = None
    w_0: Optional[FloatParam] = None
    w_a: Optional[FloatParam] = None

    transfer: Optional[jnp.ndarray] = field(default=None, compare=False)

    growth: Optional[jnp.ndarray] = field(default=None, compare=False)

    def __post_init__(self):
        if self._is_transforming():
            return

        dtype = self.conf.cosmo_dtype
        for field in fields(self):
            value = getattr(self, field.name)
            value = tree_map(lambda x: jnp.asarray(x, dtype=dtype), value)
            object.__setattr__(self, field.name, value)

    def __add__(self, other):
        return tree_map(add, self, other)

    def __sub__(self, other):
        return tree_map(sub, self, other)

    def __mul__(self, other):
        return tree_map(lambda x: x * other, self)

    def __rmul__(self, other):
        return self.__mul__(other)

    def astype(self, dtype):
        """Cast parameters to dtype by changing conf.cosmo_dtype."""
        conf = self.conf.replace(cosmo_dtype=dtype)
        return self.replace(conf=conf)  # calls __post_init__

    @property
    def k_pivot(self):
        """Primordial scalar power spectrum pivot scale in [1/L].

        Pivot scale is defined h-less unit, so needs h to convert its unit to [1/L].

        """
        return self.conf.k_pivot_Mpc / (self.h * self.conf.Mpc_SI) * self.conf.L

    @property
    def Omega_c(self):
        """Cold dark matter density parameter today."""
        Omega_b = 0 if self.Omega_b is None else self.Omega_b
        return self.Omega_m - Omega_b

    @property
    def Omega_de(self):
        """Dark energy density parameter today."""
        Omega_mk = self.Omega_m if self.Omega_k is None else self.Omega_m + self.Omega_k
        return 1 - Omega_mk

    @property
    def ptcl_mass(self):
        """Particle mass in [M]."""
        return self.conf.rho_crit * self.Omega_m * self.conf.ptcl_cell_vol


DMO = partial(Cosmology, Omega_b=None)
DMO.__doc__ = "Dark matter only cosmology."

SimpleLCDM = partial(
    Cosmology,
    A_s_1e9=2.0,
    n_s=0.96,
    Omega_m=0.3,
    Omega_b=0.05,
    h=0.7,
)
SimpleLCDM.__doc__ = "Simple ΛCDM cosmology, for convenience and subject to change."

Planck18 = partial(
    Cosmology,
    A_s_1e9=2.105,
    n_s=0.9665,
    Omega_m=0.3111,
    Omega_b=0.04897,
    h=0.6766,
)
Planck18.__doc__ = "Planck 2018 cosmology, arXiv:1807.06209 Table 2 last column."




def odeint_rk4(fun, y0, t, *args):

    def rk4(carry, t):
        y, t_prev = carry
        h = t - t_prev
        k1 = fun(y, t_prev, *args)
        k2 = fun(y + h * k1 / 2, t_prev + h / 2, *args)
        k3 = fun(y + h * k2 / 2, t_prev + h / 2, *args)
        k4 = fun(y + h * k3, t, *args)
        y = y + 1.0 / 6.0 * h * (k1 + 2 * k2 + 2 * k3 + k4)
        return (y, t), y
    #(yf, _), y = scan(rk4, (y0, np.array(t[0])), t)
    (yf, _), y = scan(rk4, (y0, t[0]), t)
    return y


def E2(a, cosmo):
    r"""Squared Hubble parameter time scaling factors, :math:`E^2`, at given scale
    factors.

    Parameters
    ----------
    a : array_like
        Scale factors.
    cosmo : Cosmology

    Returns
    -------
    E2 : jax.numpy.ndarray of cosmo.conf.cosmo_dtype
        Squared Hubble parameter time scaling factors.

    Notes
    -----
    The squared Hubble parameter

    .. math::

        H^2(a) = H_0^2 E^2(a),

    has time scaling

    .. math::

        E^2(a) = \Omega_\mathrm{m} a^{-3} + \Omega_\mathrm{k} a^{-2}
                 + \Omega_\mathrm{de} a^{-3 (1 + w_0 + w_a)} e^{-3 w_a (1 - a)}.

    """
    a = jnp.asarray(a, dtype=cosmo.conf.cosmo_dtype)

    Oka2 = 0 if cosmo.Omega_k is None else cosmo.Omega_k * a**-2
    w_0 = -1 if cosmo.w_0 is None else cosmo.w_0
    w_a = 0 if cosmo.w_a is None else cosmo.w_a

    de_a = (a**(-3 * (1 + w_0 + w_a))
           * jnp.exp(-3 * w_a * (1 - a)))
    return cosmo.Omega_m * a**-3 + Oka2 + cosmo.Omega_de * de_a


@partial(jnp.vectorize, excluded=(1,))
def H_deriv(a, cosmo):
    r"""Hubble parameter derivatives, :math:`\mathrm{d}\ln H / \mathrm{d}\ln a`, at
    given scale factors.

    Parameters
    ----------
    a : array_like
        Scale factors.
    cosmo : Cosmology

    Returns
    -------
    dlnH_dlna : jax.numpy.ndarray of cosmo.conf.cosmo_dtype
        Hubble parameter derivatives.

    """
    a = jnp.asarray(a, dtype=cosmo.conf.cosmo_dtype)

    E2_value, E2_grad = value_and_grad(E2)(a, cosmo)
    dlnH_dlna = 0.5 * a * E2_grad / E2_value

    return dlnH_dlna


def Omega_m_a(a, cosmo):
    r"""Matter density parameters, :math:`\Omega_\mathrm{m}(a)`, at given scale factors.

    Parameters
    ----------
    a : array_like
        Scale factors.
    cosmo : Cosmology

    Returns
    -------
    Omega : jax.numpy.ndarray of cosmo.conf.cosmo_dtype
        Matter density parameters.

    Notes
    -----

    .. math::

        \Omega_\mathrm{m}(a) = \frac{\Omega_\mathrm{m} a^{-3}}{E^2(a)}

    """
    a = jnp.asarray(a, dtype=cosmo.conf.cosmo_dtype)
    return cosmo.Omega_m / (a**3 * E2(a, cosmo))



def growth_integ(cosmo):
    """Intergrate and tabulate (LPT) growth functions and derivatives at given scale
    factors.

    Parameters
    ----------
    cosmo : Cosmology

    Returns
    -------
    cosmo : Cosmology
        A new instance containing a growth table, or the input one if it already exists.
        The growth table has the shape ``(num_lpt_order, num_derivatives,
        num_scale_factors)`` and ``cosmo.conf.cosmo_dtype``.

    Notes
    -----

    TODO: ODE math

    """
    if cosmo.growth is not None:
        return cosmo

    conf = cosmo.conf

    with ensure_compile_time_eval():
        eps = jnp.finfo(conf.cosmo_dtype).eps
        growth_a_ic = 0.5 * jnp.cbrt(eps).item()  # ~ 3e-6 for float64, 2e-3 for float32
        if growth_a_ic >= conf.a_lpt_step:
            growth_a_ic = 0.1 * conf.a_lpt_step

    a = conf.growth_a
    lna = jnp.log(a.at[0].set(growth_a_ic))

    num_order, num_deriv, num_a = 2, 3, len(a)

    # TODO necessary to add lpt_order support?
    # G and lna can either be at a single time, or have leading time axes
    def ode(G, lna, cosmo):
        a = jnp.exp(lna)
        H_fac = H_deriv(a, cosmo)
        Omega_fac = 1.5 * Omega_m_a(a, cosmo)
        G_1, Gp_1, G_2, Gp_2 = jnp.split(G, num_order * (num_deriv-1), axis=-1)
        Gpp_1 = (-3 - H_fac + Omega_fac) * G_1 + (-4 - H_fac) * Gp_1
        Gpp_2 = Omega_fac * G_1**2 + (-8 - 2*H_fac + Omega_fac) * G_2 + (-6 - H_fac) * Gp_2
        return jnp.concatenate((Gp_1, Gpp_1, Gp_2, Gpp_2), axis=-1)

    G_ic = jnp.array((1, 0, 3/7, 0), dtype=conf.cosmo_dtype)
    #G = odeint(ode, G_ic, lna, cosmo, rtol=conf.growth_rtol, atol=conf.growth_atol)
    G = odeint_rk4(ode, G_ic, lna, cosmo)

    G_deriv = ode(G, lna[:, jnp.newaxis], cosmo)

    G = G.reshape(num_a, num_order, num_deriv-1)
    G_deriv = G_deriv.reshape(num_a, num_order, num_deriv-1)
    G = jnp.concatenate((G, G_deriv[..., -1:]), axis=2)
    G = jnp.moveaxis(G, 0, 2)

    m = jnp.array((1, 2), dtype=conf.cosmo_dtype)[:, jnp.newaxis]
    growth = jnp.stack((
        G[:, 0],
        m * G[:, 0] + G[:, 1],
        m**2 * G[:, 0] + 2 * m * G[:, 1] + G[:, 2],
    ), axis=1)

    return cosmo.replace(growth=growth)



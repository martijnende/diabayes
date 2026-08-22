import jax.numpy as jnp

import diabayes as db
from diabayes.typedefs import StateDict


def init_params():

    a = 0.01
    b = 0.9 * a
    Dc = 1e-5

    k = 1e3
    v0 = 1e-6
    v1 = 1e-5

    mu0 = 0.6
    theta0 = Dc / v0

    state_obj = StateDict(keys=("theta",), vals=jnp.atleast_1d(theta0))

    variables = db.Variables(mu=jnp.atleast_1d(mu0), state=state_obj)
    params = db.RSFParams(a=a, b=b, Dc=Dc)
    constants = db.RSFConstants(v0=v0, mu0=mu0)
    block_constants = db.SpringBlockConstants(k=k, v_lp=v1)

    return variables, params, constants, block_constants


def init_params_cns():

    h = 1e-3
    phi0 = 0.03

    k = 1e3
    v_lp = 1e-5

    alpha = 0.3
    phi_c = 0.4
    z = 1e-5
    mu0 = 0.6
    v0 = 1e-6
    a = 0.01

    phi_ini = 0.2

    state_obj = StateDict(keys=("phi",), vals=jnp.atleast_1d(phi_ini))
    variables = db.Variables(mu=jnp.atleast_1d(mu0), state=state_obj)
    params = db.CNSParams(alpha=alpha, phi_c=phi_c, z=z, mu0=mu0, v0=v0, a=a)
    constants = db.CNSConstants(h=h, phi0=phi0)
    block_constants = db.SpringBlockConstants(k=k, v_lp=v_lp)

    return variables, params, constants, block_constants

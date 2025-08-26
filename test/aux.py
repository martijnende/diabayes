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
    params = db.RSFParams(*jnp.array([a, b, Dc])[:, None])
    constants = db.RSFConstants(v0=v0, mu0=mu0)
    block_constants = db.SpringBlockConstants(k=k, v_lp=v1)

    return variables, params, constants, block_constants

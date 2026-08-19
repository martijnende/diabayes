import pandas as pd
import jax.numpy as jnp

from diabayes.forward_models import Forward, rsf, ageing_law, springblock
from diabayes.solver import ODESolver

from aux import init_params


def double_vstep():

    _, params, constants, block_constants = init_params()
    forward = Forward(rsf, {"theta": ageing_law}, springblock)
    solver = ODESolver(forward)

    dt = 0.01
    v0 = constants.v0
    t_steps = jnp.cumsum(jnp.array([100.0, 500.0, 50.0, 1000.0, 50.0]))
    v_steps = jnp.array([10 * v0, v0, 10 * v0, 0.0, 10 * v0])

    forward.set_initial_values(mu=1e-4, theta=1)
    y0 = forward.variables

    result, t = solver.generate_sequence(
        t_steps, v_steps, dt, y0, params, constants, block_constants
    )
    v = rsf(result, params, constants)
    v_lp = jnp.ones_like(t)
    t_start = 0.0
    for t_end, vi in zip(t_steps, v_steps):
        inds = (t >= t_start) & (t < t_end)
        v_lp = v_lp.at[inds].set(vi)
        t_start = t_end

    df = pd.DataFrame({"t": t, "mu": result.mu, "v": v, "v_lp": v_lp})
    df.to_csv("synthetic_data.csv", index=False)


if __name__ == "__main__":
    double_vstep()

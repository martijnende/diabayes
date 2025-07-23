import jax.numpy as jnp

import diabayes as db
from diabayes.forward_models import Forward, ageing_law, rsf, springblock
from diabayes.solver import ODESolver

from .aux import init_params


class TestSolvers:

    def test_forward_solver(self):

        variables, params, constants, block_constants = init_params()
        forward = Forward(rsf, ageing_law, springblock)
        solver = ODESolver(forward)

        t = jnp.linspace(0.0, 100.0, 1000)
        result = solver.solve_forward(t, variables, params, constants, block_constants)

        assert result is not None
        assert result.ys is not None

        # Steady-state tests

        mu_final = result.ys.mu[-1]
        mu_pred = constants.mu0 + (params.a - params.b) * jnp.log(
            block_constants.v_lp / constants.v0
        )
        assert jnp.allclose(mu_final, mu_pred)

        state_final = result.ys.state[-1]
        state_pred = params.Dc / block_constants.v_lp
        assert jnp.allclose(state_final, state_pred)

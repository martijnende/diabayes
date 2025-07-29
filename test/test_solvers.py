import jax
import jax.numpy as jnp

import diabayes as db
from diabayes.forward_models import Forward, ageing_law, rsf, springblock
from diabayes.solver import ODESolver

from .aux import init_params

jax.config.update("jax_enable_x64", True)


class TestSolvers:

    def test_forward_solver(self):
        """
        Test to see whether the ODE solver runs correctly,
        and whether the steady-state result is as expected.
        """

        variables, params, constants, block_constants = init_params()
        forward = Forward(rsf, ageing_law, springblock)
        solver = ODESolver(forward)

        t = jnp.linspace(0.0, 100.0, 1000)
        result = solver.solve_forward(t, variables, params, constants, block_constants)
        result_jax = solver._solve_forward(
            t, variables, params, constants, block_constants
        )

        assert result is not None
        assert result_jax is not None
        assert result_jax.ys is not None

        diff = result.to_array() - result_jax.ys.to_array()

        assert jnp.allclose(result.to_array(), result_jax.ys.to_array())

        # Steady-state tests

        mu_final = result.mu[-1]
        mu_pred = constants.mu0 + (params.a - params.b) * jnp.log(
            block_constants.v_lp / constants.v0
        )
        assert jnp.allclose(mu_final, mu_pred)

        state_final = result.state[-1]
        state_pred = params.Dc / block_constants.v_lp
        assert jnp.allclose(state_final, state_pred)

    def test_levenberg_marquardt(self):
        """
        Test for the Levenberg-Marquardt maximum-likelihood
        inversion solver. A forward simulation generates the
        friction "observation" for a given set of parameters.
        Then the true parameters are pertubed and an inversion
        is performed. The test checks whether the inverted
        parameters are identical to the initial ones, and
        whether the friction curves are identical.
        """

        variables, params, constants, block_constants = init_params()
        forward = Forward(rsf, ageing_law, springblock)
        solver = ODESolver(forward)

        # Forward simulation to generate "observations"
        t = jnp.linspace(0.0, 20.0, 1000)
        result = solver.solve_forward(t, variables, params, constants, block_constants)

        # Perturb parameters (initial guess parameters)
        params2 = db.RSFParams(a=params.a * 1.1, b=params.b * 0.9, Dc=params.Dc * 5)
        mu = result.mu

        # Invert parameters
        result_inv = solver.max_likelihood_inversion(
            t, mu, params2, constants, block_constants, verbose=False
        )
        params_inv = result_inv.value

        # Reproduce friction curve
        result2 = solver.solve_forward(
            t, variables, params_inv, constants, block_constants
        )

        assert result2 is not None

        # Check that inverted parameters and resulting friction
        # curves are identical to the original ones
        assert jnp.allclose(params.to_array(), params_inv.to_array())
        assert jnp.allclose(result.to_array(), result2.to_array())

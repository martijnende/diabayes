import jax.numpy as jnp

import diabayes as db
import diabayes.forward_models as db_models

from .aux import init_params


class TestForwardModels:

    def test_rate_and_state(self):

        variables, params, constants, block_constants = init_params()

        # Steady-state tests

        dtheta = db_models.ageing_law(constants.v0, variables, params, constants)
        assert jnp.allclose(dtheta, 0.0)

        v = db_models.rsf(variables, params, constants)
        assert jnp.allclose(v, constants.v0)

        # Asymptotic tests

        v = 1e-20 * block_constants.v_lp
        dtheta = db_models.ageing_law(v, variables, params, constants)
        assert jnp.allclose(dtheta, 1.0)

        variables2 = db.Variables(mu=0.0, state=variables.state)
        v = db_models.rsf(variables2, params, constants)
        assert jnp.allclose(v, 0.0)

        variables2 = db.Variables(mu=variables.mu, state=1e20)
        v = db_models.rsf(variables2, params, constants)
        assert jnp.allclose(v, 0.0)

    def test_spring_block(self):

        variables, _, constants, block_constants = init_params()

        # Steady-state tests

        block_constants2 = db.SpringBlockConstants(block_constants.k, v_lp=constants.v0)
        dmu = db_models.springblock(constants.v0, variables, block_constants2)
        assert jnp.allclose(dmu, 0.0)

        # Asymptotic tests

        v = 1e-20 * block_constants.v_lp
        dmu = db_models.springblock(v, variables, block_constants)
        assert jnp.allclose(dmu, block_constants.k * block_constants.v_lp)

        v = 1e20 * block_constants.v_lp
        dmu = db_models.springblock(v, variables, block_constants)
        assert jnp.allclose(dmu, -block_constants.k * v)

import jax.numpy as jnp

from diabayes.typedefs import RSFParams, StateDict, Variables


class TestTypedefs:

    def test_basic_containers(self):

        # Create container with .from_array method
        x = {"a": 1.0, "b": 2.0, "c": 3.0}
        keys = tuple(x.keys())

        state_obj = StateDict(keys=keys, vals=jnp.array(list(x.values())))
        variables = Variables(mu=jnp.array([0.6]), state=state_obj)

        vars_array = variables.to_array()
        variables2 = Variables.from_array(vars_array, keys)
        assert jnp.allclose(vars_array, variables2.to_array())

        # Update values
        x2 = {"mu": 0.2, "a": 1.0, "b": 2.0, "c": 5.0}
        variables3 = variables.set_values(**x2)
        assert jnp.allclose(jnp.array(list(x2.values())), variables3.to_array())

        # Verify that all state keys are present
        x2.pop("mu")
        assert tuple(x2.keys()) == variables3.state.keys

        # Appending scalars
        variables3 = variables.append(variables2)
        double_array = jnp.vstack([vars_array, vars_array]).T
        assert jnp.allclose(double_array, variables3.to_array())

        # Appending time-series
        N = 100
        mu = jnp.zeros(N, dtype=jnp.float32)
        x = {
            "a": jnp.ones_like(mu),
            "b": 2 * jnp.ones_like(mu),
            "c": 3 * jnp.ones_like(mu),
        }
        state_obj = StateDict(keys=tuple(x.keys()), vals=jnp.array(list(x.values())))
        variables = Variables(mu=mu, state=state_obj)
        variables2 = Variables(mu=mu, state=state_obj)
        variables3 = variables.append(variables2)
        array = jnp.array([mu, *x.values()])
        double_array = jnp.hstack([array, array])
        assert jnp.allclose(double_array, variables3.to_array())

    def test_SVI_containers(self):

        import jax.random as jr

        key = jr.PRNGKey(42)

        # Test Particles

        key, split_key = jr.split(key)
        Nparticles = 100
        loc = jnp.array([1.0, 2.0, 3.0])
        scale = jnp.ones(3)
        particles = RSFParams.generate(Nparticles, loc, scale, split_key)

        assert len(particles) == Nparticles

        x = particles.to_array()
        assert x.shape[0] == len(loc)

        rtol = 2 / jnp.sqrt(Nparticles)
        assert jnp.allclose(x.mean(axis=1), loc, rtol=rtol)
        assert jnp.allclose(x.std(axis=1), scale, rtol=rtol)

        assert jnp.allclose(x, RSFParams.from_array(x).to_array())

        # Test Chains

        pass

    # def test_Bayesian_statistics(self): ...

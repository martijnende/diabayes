import pytest
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
        double_array = jnp.vstack([vars_array, vars_array])
        assert jnp.allclose(double_array, variables3.to_array())

        # Appending time-series
        N = 100
        mu = jnp.arange(N, dtype=jnp.float32)
        x = {
            "a": jnp.ones_like(mu),
            "b": 2 * jnp.ones_like(mu),
            "c": 3 * jnp.ones_like(mu),
        }
        state_obj = StateDict(keys=tuple(x.keys()), vals=jnp.vstack(list(x.values())).T)
        variables = Variables(mu=mu, state=state_obj)
        variables2 = Variables(mu=mu, state=state_obj)
        variables3 = variables.append(variables2)
        array = jnp.column_stack([mu, *x.values()])
        double_array = jnp.vstack([array, array])
        assert jnp.allclose(double_array, variables3.to_array())

    def test_container_slicing(self):

        N = 100
        mu = jnp.arange(N, dtype=jnp.float32)
        x = {
            "a": jnp.ones_like(mu),
            "b": 2 * jnp.ones_like(mu),
            "c": 3 * jnp.ones_like(mu),
        }
        state_obj = StateDict(keys=tuple(x.keys()), vals=jnp.vstack(list(x.values())).T)
        variables = Variables(mu=mu, state=state_obj)

        # Get item by name (returns a specific variable)
        mu2 = variables["mu"]
        assert jnp.allclose(mu, mu2)
        for key, val in x.items():
            assert jnp.allclose(variables[key], val)

        # Get item by index (returns a specific time value)
        for i in (0, 1, -1):
            variables2 = variables[i]
            assert jnp.isclose(variables2.mu, mu[i])
            for key, val in x.items():
                assert jnp.isclose(variables2[key], val[i])

        # Check that an index selection of scalars yields identity
        # regardless of the index value
        variables2 = variables[0]
        variables3 = variables2[999]
        assert jnp.isclose(variables2.mu, variables3.mu)
        for key, val in x.items():
            assert jnp.allclose(variables2[key], variables3[key])

        # Get items by slice
        slc = slice(0, min(10, N))
        variables2 = variables[slc]
        assert jnp.allclose(variables2.mu, mu[slc])
        for key, val in x.items():
            assert jnp.allclose(variables2[key], val[slc])

        # Verify that anything else triggers an IndexError
        with pytest.raises(IndexError):
            variables[[0]]  # type: ignore
            variables[jnp.arange(3)]  # type: ignore
            variables[True]  # type: ignore
            variables[None]  # type: ignore

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

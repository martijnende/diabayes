import jax.numpy as jnp

from diabayes.typedefs import RSFParams, RSFParticles, Variables


class TestTypedefs:

    def test_basic_containers(self):

        a = 1.0
        b = 2.0
        c = 3.0

        assert (
            Variables(a, b)
            == Variables(mu=a, state=b)
            == Variables.from_array(jnp.array([a, b]))
            == Variables.tree_unflatten(None, jnp.array([a, b]))
        )
        variables = Variables(mu=a, state=b)
        assert jnp.allclose(
            variables.to_array(), jnp.array(variables.tree_flatten()[0])
        )

        assert (
            RSFParams(a, b, c)
            == RSFParams(a=a, b=b, Dc=c)
            == RSFParams.from_array(jnp.array([a, b, c]))
            == RSFParams.tree_unflatten(None, jnp.array([a, b, c]))
        )
        params = RSFParams(a=a, b=b, Dc=c)
        assert jnp.allclose(params.to_array(), jnp.array(params.tree_flatten()[0]))

    def test_SVI_containers(self):

        import jax.random as jr

        key = jr.PRNGKey(42)

        # Test Particles

        key, split_key = jr.split(key)
        Nparticles = 100
        loc = jnp.array([1.0, 2.0, 3.0])
        scale = jnp.ones(3)
        particles = RSFParticles.generate(Nparticles, loc, scale, split_key)

        assert len(particles) == Nparticles

        x = particles.to_array()
        assert x.shape[1] == len(loc)

        rtol = 2 / jnp.sqrt(Nparticles)
        assert jnp.allclose(x.mean(axis=0), loc, rtol=rtol)
        assert jnp.allclose(x.std(axis=0), scale, rtol=rtol)

        assert jnp.allclose(x, RSFParticles.from_array(x).to_array())

        # Test Chains

        pass

    def test_Bayesian_statistics(self): ...

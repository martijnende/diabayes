# Maximum-likelihood inversion

For a given measured friction curve ($\mu_{obs}$), it is likely that a single set of parameters $\mathcal{P}$ produces the best possible agreement between a forward simulation $\mu = \mathrm{Forward}[t, \mathcal{P}]$ and the (noisy) measurements. This single optimal estimate of $\mathcal{P}$ is called the _maximum-likelihood estimate_, and it can be obtained through standard inversion schemes.

For the least-squares objective function discussed in [the previous section](index), the [Levenberg-Marquardt algorithm](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm) is one of the fastest converging methods on the market. DiaBayes utilises the [`LevenbergMarquardt`](https://docs.kidger.site/optimistix/api/least_squares/#optimistix.LevenbergMarquardt) implementation of the [Optimistix](https://docs.kidger.site/optimistix/) library, which interfaces directly with the [`diffrax.diffeqsolve`](https://docs.kidger.site/diffrax/api/diffeqsolve/) routine that is used to solve the forward problem.

Concretely, performing a maximum-likelihood inversion in DiaBayes is fairly straightforward. Assuming that a solver has already been instantiated with an appropriate forward model (see the [Getting Started guide](../../getting-started)), and given some measured friction curve `mu_measured` and initial guess parameters `params_guess`, the inversion can be performed with a single function call:
```python
inv_result = solver.max_likelihood_inversion(
    t, mu_measured,
    params=params_guess,
    friction_constants=constants,
    block_constants=block_constants,
    verbose=True
)

params_inverted = inv_result.value
```
Depending on the number of data samples and the complexity of the forward model, this inversion should take anywhere from a few seconds to at most one minute.

In many cases, the maximum-likelihood result is sufficient for the analysis of an experiment. However, this approach provides no information on the uncertainty in the inverted parameters, and the potential trade-offs between them. For a complete evaluation of the admissible range of parameters, consider performing a Bayesian inversion (hint: see next section).
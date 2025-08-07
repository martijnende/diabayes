# Bayesian inversion

Bayesian inversion refers to a family of inversion methods that aim to estimate the _probability_ that a given set of parameters $\mathcal{P}$ can explain the measured data $\mu_{obs}$. This probability is expressed in terms of the _posterior distribution_ over the parameters, which comprises a likelihood term and a prior probability term, usually written as:
```{math}
:label: eq:Bayes
p\left(\mathcal{P} | \mu_{obs} \right) \propto p\left(\mu_{obs} | \mathcal{P} \right) p\left(\mathcal{P}\right)
```
For those new to Bayesian statistics: $p\left(a|b \right)$ is read as "the probability that $a$ is true given that $b$ is true" (loosely speaking). The first term on the right-hand side is called the _likelihood_ term, and the second term is the _prior_ probability (or simply "prior"). The likelihood term can be estimated by running a forward simulation (with a given $\mathcal{P}$) and calculating the mismatch with the observations; the maximum-likelihood inversion discussed in [the previous section](max-likelihood) aims to find $\mathcal{P}$ that maximises this term. However, Bayesian inversion also takes into account that one might have a-priori information about $\mathcal{P}$ itself, before observing any data (the prior). For example, we might expect $\mathcal{P}$ to be positive, or be constrained by independent measurements, etc. Lastly, the proportionality constant, which quantifies the probability of observing the data, is unknown but irrelevant for the inversion.

## Gradients of the posterior

In practice, most Bayesian inversion methods sample the logaritm of $p$, hence Eq. {eq}`eq:Bayes` becomes:
```{math}
:label: eq:logBayes
\ln p\left(\mathcal{P} | \mu_{obs} \right) \propto \ln p\left(\mu_{obs} | \mathcal{P} \right) + \ln p\left(\mathcal{P}\right)
```
To permit an informed exploration of the posterior distribution, gradient information is used to guide the search direction within the parameter space. To see how our forward simulations fit into this, let's consider a normal (Gaussian) likelihood distribution:
```{math}
:label: eq:Gaussian
p\left(\mu_{obs} | \mathcal{P} \right) \propto \prod_t \exp \left( - \left[ \frac{\mu_{obs}(t) - \mathrm{Forward}\left[t, \mathcal{P}\right]}{\sqrt{2} \nu} \right]^2 \right)
```
To avoid confusion with the normal stress (often denoted $\sigma$), the standard deviation (also often denoted by $\sigma$) is here denoted by $\nu$. This standard deviation is expected to be proportional to the measurement noise amplitude in $\mu_{obs}$. The logarithm of Eq. {eq}`eq:Gaussian` turns the product into a summation:
```{math}
:label: eq:log_Gaussian
\ln p\left(\mu_{obs} | \mathcal{P} \right) \propto - \sum_t \left[ \frac{\mu_{obs}(t) - \mathrm{Forward}\left[t, \mathcal{P}\right]}{\sqrt{2} \nu} \right]^2
```
Lastly, the derivative of the log-likelihood with respect to $\mathcal{P}$ can then be found by virtue of the chain rule, and differentiation through the ODE solution:
```{math}
:label: eq:grad_log_Gaussian
\frac{\partial \ln p\left(\mu_{obs} | \mathcal{P} \right)}{\partial \mathcal{P}} \propto \sum_t \frac{\mu_{obs}(t) - \mathrm{Forward}\left[t, \mathcal{P}\right]}{2 \nu^2} \frac{\partial \mathrm{Forward}}{\partial \mathcal{P}}
```
Because of the differentiability of the DiaBayes forward models, the gradient term on the right-hand side can be precisely calculated. A similar procedure is applied to obtain the gradients of the prior distribution.

By doing gradient descent on $-\partial \ln p\left(\mu_{obs} | \mathcal{P} \right) / \partial \mathcal{P}$, one arrives as the maximum a-posteriori estimate, which strikes a balance between the likelihood and prior terms. But this is not what we want: the point of doing Bayesian inversion, is to gain insight into the _distribution_ of parameter probabilities, not just a single estimate thereof. To estimate the posterior distribution, one needs to perform some kind of sampling of the neighbourhood around the maximum posterior. There are numerous methods that can do this, many of which have been implemented into the [Blackjax](https://blackjax-devs.github.io/blackjax/) library. However, DiaBayes implements its own Bayesian sampler: _Stein Variational Inference_.


## Stein Variational Inference

Stein Variational Inference (SVI) through _Stein Variational Gradient Descent_{footcite}`liu2016` is a particle-swarm optimisation method in which each particle represents one sample from the posterior distribution. The particles can move around in a space with the same number of dimensions as the number of invertible parameters. They are attracted towards the maximum in the likelihood and the prior propabilities (through gradient descent on the negative log-likelihood and log-prior, see Eq. {eq}`eq:grad_log_Gaussian`), and they repel one another based on their mutual distance. Eventually the particle swarm will find an equilibrium, after which the particle positions are fixed. This final distribution of the particles approximates the posterior distribution.

To get a statistically significant sample of the posterior distribution, we need $N \gg 1$ particles, which scales with the number of invertible parameters. For an inversion of a classical rate-and-state friction model with 3 parameters, a sample size of $N = 1000$ is a good starting point. And thanks to the `jax.vmap` functionality we can run $N$ forward simulations in parallel on a GPU, so that doing the inversion with $N=100$ is not necessarily much faster than with $N=1000$.

To find the equilibrium state, we randomly initialise the swarm of $N$ particles in a ball around the maximum-likelihood estimate (see [the previous section](max-likelihood)), and allow for $T$ update steps to converge towards the equilibrium state. During each step, we first calculate an "effective gradient" as defined by the SVI method:
```{math}
:label: eq:SVI
\begin{split}
\phi\left(\mathcal{P}_n \right) = \frac{1}{N} \sum_{i=1}^N \underbrace{\kappa \left(\mathcal{P}_n, \mathcal{P}_i \right) \frac{\partial \ln p \left(\mu_{obs} | \mathcal{P}_i \right)}{\partial \mathcal{P}_i}}_{\text{attraction to likelihood}} \\
+ \underbrace{\kappa \left(\mathcal{P}_n, \mathcal{P}_i \right) \frac{\partial \ln p \left(\mathcal{P}_i \right)}{\partial \mathcal{P}_i}}_{\text{attraction to prior}} \\
+ \underbrace{\kappa \left(\mathcal{P}_n, \mathcal{P}_i \right)}_{\text{repulsion}}
\end{split}
```
Here, $\kappa(a, b)$ denotes a function (the "kernel") that equals to 1 when $a = b$, and decays to 0 when $a$ and $b$ get farther removed from each other. DiaBayes uses a standard radial basis function combined with a Euclidean distance metric:
```{math}
:label: eq:kernel
\kappa(a, b) = \exp \left(- \frac{1}{h} ||a - b||_2^2 \right)
```
where $||x||_2$ denotes the Euclidean norm of a vector, and $h$ is calculated for the entire particle swarm using [the median trick](https://en.wikipedia.org/wiki/Median_trick). Then, each particle position is updated using a gradient-descent approach. For classical gradient descent, the update step looks like this:
```{math}
:label: eq:SVI_update
\mathcal{P}_n^{k+1} = \mathcal{P}_n^{k} - \eta \phi\left(\mathcal{P}_n^k \right)
```
However, DiaBayes employs the [Adam](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adam){footcite}`kingma2014` implementation of the [Optax](https://optax.readthedocs.io/en/latest/index.html) library for improved convergence, which employs a slightly different formulation.

After having converged over $T$ steps, the final positions of the set of particles $\left\{ \mathcal{P} \right\}_{n=1}^N$ represent the posterior probability distribution over the parameters.

## Bayesian analysis in DiaBayes

Concretely, if one wants to perform a Bayesian inversion of the forward model parameters, it only requires a single function call, e.g.:
```python
bayesian_result = solver.bayesian_inversion(
    t=t, mu=mu_measured, noise_std=noise_std,
    params=params_init, friction_constants=constants, 
    block_constants=block_constants, 
    Nparticles=1500, Nsteps=100, rng=42
)
```
The function call signature is very similar to that of the `diabayes.ODEsolver.max_likelihood_solver` routine, with a few additional arguments specific to the SVI method. The `examples/simple_inversion.ipynb` example notebook describes these arguments in detail, which will not be repeated here. This notebook also demonstrates how to use the various analysis tools that accompany the `bayesian_inversion` method, such as convergence checking and uncertainty visualisation.

What is important to clarify here, is that the prior distribution is currently not "properly" defined, at least not in the strict meaning of the term. In many (or even the majority) of Bayesian geophysics studies, the prior distribution is assumed to be uniform (also known as a "flat prior"). The corresponding gradients of this prior hence vanish ($\partial \ln p\left(\mathcal{P} \right) / \partial \mathcal{P} = 0$), causing the particles to settle around the likelihood distribution. This is not necessarily a bad strategy, because often there are no independent constraints on the parameters of the friction model, and a uniform prior accurately reflects this notion of "we don't know what to expect". On the other hand, we actually do know a bit more that: both empirically and theoretically, the rate-and-state friction parameters $a$ and $D_c$ are known to be strictly positive, and we expect them to exhibit a certain amplitude (e.g., $10^{-4} < a < 10^{-1}$). To somewhat account for this additional knowledge, DiaBayes opts to define a prior conditioned on the maximum-likelihood estimate $\mathcal{P}^*$, i.e.:
```{math}
p\left( \mathcal{P} \right) \approx q \left( \mathcal{P} | \mathcal{P}^* \right)
```
Because of this conditioning on $\mathcal{P}^*$, which itself in conditioned on the data $\mu_{obs}$, this choice of a prior cannot be considered a _true_ prior. Nonetheless, it is a convenient choice to somewhat restrict the posterior distribution to parameter values that one would consider reasonable (assuming that the maximum-likelihood inversion found reasonable parameter values).

A second important clarification, is that currently DiaBayes solves the Bayesian problem in log-parameter space, i.e. $\mathcal{P} = \ln \mathcal{Q}$, with $\mathcal{Q}$ representing the forward model parameters like $a$, $b$, $D_c$ (in the case of rate-and-state friction). This strategy ensures that all the parameters $\mathcal{Q}$ are strictly positive, which greatly improves the stability of the inversion; forward simulations with $a < 0$ are not only non-physical, they are also highly unstable. And simulations with $D_c < 0$ are not defined at all (logarithm of a negative number). Moreover, some parameters can vary by orders of magnitude, and hence solving the problem in log-space is a sensible choice. The flip-side of this strategy, is that some parameters that are normally not restricted to be positive (like the rate-and-state $b$ parameter) cannot become negative, even if it is demanded by the observed data.

```{warning}
TL;DR of the above:
- The DiaBayes prior distribution is conditioned on the maximum-likelihood solution.
- The inversion is done on log-transformed parameters, hence the resulting values are strictly positive.
```

```{admonition} Future work
Future versions of DiaBayes will address the above restrictions by allowing for user-defined prior distribution functions.
```

```{rubric} References
```
```{footbibliography}
```
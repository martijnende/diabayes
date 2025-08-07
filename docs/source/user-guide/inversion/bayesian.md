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

After having converged over $T$ steps, the final positions of $\left\{ \mathcal{P} \right\}_{n=1}^N$ represent the posterior probability distribution over the parameters.

## Usage in DiaBayes

...

```{rubric} References
```
```{footbibliography}
```
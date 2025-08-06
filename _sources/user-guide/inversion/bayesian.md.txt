# Bayesian inversion

Bayesian inversion refers to a family of inversion methods that aim to estimate the _probability_ that a given set of parameters $\mathcal{P}$ can explain the measured data $\mu_{obs}$. This probability is expressed in terms of the _posterior distribution_ over the parameters, which comprises a likelihood term and a prior probability term, usually written as:
```{math}
:label: Bayes
p\left(\mathcal{P} | \mu_{obs} \right) \propto p\left(\mu_{obs} | \mathcal{P} \right) p\left(\mathcal{P}\right)
```
For those new to Bayesian statistics: $p\left(a|b \right)$ is read as "the probability that $a$ is true given that $b$ is true" (loosely speaking). The first term on the right-hand side is called the _likelihood_ term, and the second term is the _prior_ probability (or simply "prior"). The likelihood term can be estimated by running a forward simulation (with a given $\mathcal{P}$) and calculating the mismatch with the observations; the maximum-likelihood inversion discussed in [the previous section](max-likelihood) aims to find $\mathcal{P}$ that maximises this term. However, Bayesian inversion also takes into account that one might have a-priori information about $\mathcal{P}$ itself, before observing any data (the prior). For example, we might expect $\mathcal{P}$ to be positive, or be constrained by independent measurements, etc. Lastly, the proportionality constant, which quantifies the probability of observing the data, is unknown but irrelevant for the inversion.

## Gradients of the posterior

In practice, most Bayesian inversion methods sample the logaritm of $p$, hence Eq. {eq}`Bayes` becomes:
```{math}
:label: logBayes
\ln p\left(\mathcal{P} | \mu_{obs} \right) \propto \ln p\left(\mu_{obs} | \mathcal{P} \right) + \ln p\left(\mathcal{P}\right)
```
To permit an informed exploration of the posterior distribution, gradient information is used to guide the search direction within the parameter space. To see how our forward simulations fit into this, let's consider a normal (Gaussian) likelihood distribution:
```{math}
:label: Gaussian
p\left(\mu_{obs} | \mathcal{P} \right) \propto \prod_t \exp \left( - \left[ \frac{\mu_{obs}(t) - \mathrm{Forward}\left[t, \mathcal{P}\right]}{\sqrt{2} \nu} \right]^2 \right)
```
To avoid confusion with the normal stress (often denoted $\sigma$), the standard deviation (also often denoted by $\sigma$) is here denoted by $\nu$. This standard deviation is expected to be proportional to the measurement noise amplitude in $\mu_{obs}$. The logarithm of Eq. {eq}`Gaussian` turns the product into a summation:
```{math}
:label: log_Gaussian
\ln p\left(\mu_{obs} | \mathcal{P} \right) \propto - \sum_t \left[ \frac{\mu_{obs}(t) - \mathrm{Forward}\left[t, \mathcal{P}\right]}{\sqrt{2} \nu} \right]^2
```
Lastly, the derivative of the log-likelihood with respect to $\mathcal{P}$ can then be found by virtue of the chain rule, and differentiation through the ODE solution:
```{math}
:label: grad_log_Gaussian
\frac{\partial \ln p\left(\mu_{obs} | \mathcal{P} \right)}{\partial \mathcal{P}} \propto \sum_t \frac{\mu_{obs}(t) - \mathrm{Forward}\left[t, \mathcal{P}\right]}{2 \nu^2} \frac{\partial \mathrm{Forward}}{\partial \mathcal{P}}
```
Because of the differentiability of the DiaBayes forward models, the gradient term on the right-hand side can be precisely calculated. A similar procedure is applied to obtain the gradients of the prior distribution.

By doing gradient descent on $-\partial \ln p\left(\mathcal{P} | \mu_{obs} \right) / \partial \mathcal{P}$, one arrives as the maximum a-posteriori estimate, which strikes a balance between the likelihood and prior terms. But this is not what we want: the point of doing Bayesian inversion, is to gain insight into the _distribution_ of parameter probabilities, not just a single estimate thereof. To estimate the posterior distribution, one needs to perform some kind of sampling of the neighbourhood around the maximum posterior. There are numerous methods that can do this, many of which have been implemented into the [Blackjax](https://blackjax-devs.github.io/blackjax/) library. However, DiaBayes implements its own Bayesian sampler: _Stein Variational Inference_.


## Stein Variational Inference


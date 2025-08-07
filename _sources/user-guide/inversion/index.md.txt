# Inverse modelling

Given a set of observations $\mu_{obs}(t)$ (i.e., a _friction curve_), and a suitable [forward model](../forward/index) with parameters $\mathcal{P}$, what are the parameter values that can explain the observations? We could choose some random $\mathcal{P}$, run a forward simulation, and see if the forward solution $\mu(t) = \mathrm{Forward}[t, \mathcal{P}]$ matches $\mu_{obs}(t)$. If not, change $\mathcal{P}$ a little bit and try again, until an acceptable agreement between $\mu(t)$ and $\mu_{obs}(t)$ is found.

Of course, we can do this more systematically. Define an objective function (or "loss" function in Machine/Deep Learning jargon):
```{math}
:label: loss_fn
\mathcal{L} = \frac{1}{T} \int_0^T \left(\mu_{obs} - \mathrm{Forward}[t, \mathcal{P}] \right)^2 \mathrm{d} t
```
In practice, the integral over $t$ is replaced by a summation (since $\mu_{obs}$ is sampled at discrete time intervals), but for the theory it makes sense to keep the integral for now. The "best" parameter values are those that result in the smallest value of $\mathcal{L}$, i.e.:
```{math}
\mathcal{P}^* = \underset{\mathcal{P}}{\arg \min} \, \left\{ \frac{1}{T} \int_0^T \left(\mu_{obs} - \mathrm{Forward}[t, \mathcal{P}] \right)^2 \mathrm{d} t \right\}
```

A classical approach to finding $\mathcal{P}^*$ is performing gradient descent on Eq. {eq}`loss_fn`: we compute the gradient of $\mathcal{L}$ with respect to $\mathcal{P}$, and then take a small step down the gradient. By repeating this procedure multiple times, we expect to eventually arrive at the (local) minimum, where the gradients are zero (and so we automatically stop there). Quantitatively, this reads:
```{math}
\mathcal{P}^{i+1} = \mathcal{P}^i - \eta \left.\frac{\mathrm{d} \mathcal{L}}{\mathrm{d} \mathcal{P}} \right\vert_{\mathcal{P}^i}
```
There are more advanced gradient-descent algorithms, like Adam{footcite}`kingma2014` (as used by DiaBayes), but they are all just different flavours of this general concept.

```{caution}
When the number of invertible parameter exceeds 1, $\mathcal{P}$ becomes a vector. A more precise notation would be to use a vector arrow $\vec{\mathcal{P}}$ to distinguish it from a scalar, but for ease of reading (and writing...) I will drop the arrow everywhere in these sections.
```

The challenge now is to find the derivative term on the right-hand side. By virtue of the [Leibniz integral rule](https://en.wikipedia.org/wiki/Leibniz_integral_rule), the total derivative of $\mathcal{L}$ becomes a partial derivative of $\mathrm{Forward}[t, \mathcal{P}]$ with respect to $\mathcal{P}$ inside the integral. But how does one differentiate the solution of an ODE with respect to its parameters? The solution at any point $t_i$ depends on the entire history from $t = t_0$ up to $t_i$, and so the derivatives at time $t_i$ depend on the derivatives at $t_{i-1}, t_{i-2}, \dots$. Fortunately, there is an elegant approach to solve this recursive problem, which is called the [_adjoint state method_](https://en.wikipedia.org/wiki/Adjoint_state_method) (or simply "adjoint method").

It is beyond the scope of the user guide to explain this method here, but there are numerous lecture notes and blog posts out there that describe this approach in the context of continuous-time ODEs (and also Neural ODEs{footcite}`chen2018_neuralode`). Conveniently, this method is built into the solvers of the [Diffrax](https://docs.kidger.site/diffrax/) library{footcite}`kidger2021`, and so in the spirit of the underlying JAX library, any operations involving an ODE solution are differentiable with respect to its parameters. DiaBayes builds further on this, integrating the gradients coming out of the adjoint back-propagation into gradient-based inversion algorithms. DiaBayes implements two flavours of inversion methods, which are the [maximum-likelihood inversion](max-likelihood) and [Bayesian inversion](bayesian) that are discussed in the next sections.

```{toctree}
:hidden:
:maxdepth: 1

max-likelihood
bayesian
```

```{rubric} References
```
```{footbibliography}
```
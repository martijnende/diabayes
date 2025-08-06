# Spring-blocks

While the friction law (and state evolution law) describes how a fault responds to a particular state of stress, an additional descriptor is needed to connect the fault response back to the state of stress (i.e., mechanical feedback). For natural faults it is often necessary to come up with an elastic model that includes complexities like a free surface and fault non-planarities. For laboratory experiments, however, it is often sufficient to represent the experimental set-up with a _spring-block_ mechanical analogue. A frictional interface is formed between a rigid "block" placed on top of a surface, and the block is pulled horizontally by a spring, the other end of which is being translated at a constant rate ($v_{lp}$). For $t > 0$, stretching of the spring transmits a force to the block. This force is resisted by friction ($\mu$) at the interface. The friction law then describes the rate of translation of the block ($v = f(\mu, \dots)$), which results in shortening of the spring, resulting in a reduction in friction, etc.

The most basic spring-block formulation is effectively zero-dimensional, meaning that there is no internal deformation of the block or variation of stress/slip at the interface. More advanced formulations can account for such internal deformation, which can lead to e.g. rupture propagation across the interface. DiaBayes currently only implements two types of zero-dimensional spring-block analogues.

## Basic spring-block

The force balance describing the most basic spring-block analogue reads as follows:
```{math}
:label: spring-block
\frac{\mathrm{d} \mu}{\mathrm{d} t} = k \left( v_{lp} - v \right)
```
where $k$ is the stiffness of the spring (with units of $[m^{-1}]$), and $v_{lp}$ is the rate of translation of the spring's endpoint (the "load-point velocity"). The response of the fault to a given state of stress is given by the friction law, i.e. $v = f(\mu, \dots)$. The difference $v_{lp} - v$ is the rate of stretching of the spring, which, combined with the stiffness $k$, gives the rate of stress increase on the interface.

The force balance (Eq. {eq}`spring-block`) is then combined with the friction law $f$ and the state evolution law $g$ to furnish a complete description of the forward model, i.e.:
```{math}
:label: forward
\frac{\mathrm{d} \vec{X}}{\mathrm{d} t} = \begin{cases}
\dot{\mu} = k \left(v_{lp} - f(\mu, \dots) \right) \\
\dot{\theta} = g(v, \theta, \dots)
\end{cases}
```

The basic spring-block model is available through `diabayes.forward_models.springblock`.

```{note}
For certain values of $k$ below a given threshold (which is dictated by the friction law), a _frictional instability_ can develop{footcite}`ruina1983`, also known as _stick-slip_. This instability results from a force imbalance that can potentially increase indefinitely for a simple spring-block analogue, meaning that the stick-slip cycles will continue to grow until the numerical solver fails to resolve the (near-infinite) acceleration. To correctly model stick-slip cycles in DiaBayes, select either an _inertial spring-block_ (see below) or pick an appropriate friction law like the [_Chen-Niemeijer-Spiers_ model](chen-niemeijer-spiers).
```

## Inertial spring-block

For a basic spring-block analogue, a normal stress is imposed to the block to act as a certain "weight" pressing down on the frictional interface. However, the block itself is assumed to be massless (non-inertial), leading to a simplified force balance. For small accelerations, like one would encounter for small velocity-steps or slide-hold-slide experiments, this simplification is sufficiently accurate (i.e., the inertial effects are negligible). For more rapid accelerations, like one would encounter in stick-slip sequences, inertia becomes a significant contributor to the force balance. When accounting for a finite (and dimensionless) mass $M$ of the block, the governing force balance becomes:
```{math}
:label: inertial_spring-block
M \frac{\mathrm{d} v}{\mathrm{d} t} = k \left(v_{lp}t - x \right) - \mu(\dots)
```
Since DiaBayes defines a friction law as $v = f(\mu, \dots)$ and not $\mu = f'(v, \dots)$, the above expression is in principle incompatible with the DiaBayes modelling strategy. However, it is possible to rewrite it using the partial derivatives of $v$ with respect to the variables of the friction law (usually friction $\mu$ and state $\theta$):
```{math}
:label: inertial_spring-block_partials
M \left( \frac{\partial v}{\partial \mu} \frac{\mathrm{d} \mu}{\mathrm{d} t} +  \frac{\partial v}{\partial \theta} \frac{\mathrm{d} \theta}{\mathrm{d} t} + \dots \right) = k \left(v_{lp}t - x \right) - \mu
```
The $\dots$ denote any other variables that might be associated with a particular friction law (normal stress, temperature, ...), though this would represent a much more [advanced physics scenario](../advanced_usage). Rewriting Eq. {eq}`inertial_spring-block_partials` in terms of $\dot{\mu}$ yields a more familiar form of the force balance:
```{math}
:label: inertial_spring-block_partials2
\frac{\mathrm{d} \mu}{\mathrm{d} t} = \left[ \frac{\partial v}{\partial \mu} \right]^{-1} \left( \frac{1}{M} \left[ k \left( v_{lp} t - x \right) - \mu \right] - \frac{\partial v}{\partial \theta} \frac{\mathrm{d} \theta}{\mathrm{d} t} - \dots \right)
```
To the forward model is completed with the somewhat trivial expression for the position of the block, $\mathrm{d}x / \mathrm{d}t = v(\mu, \theta, \dots)$, which, all together, gives:
```{math}
\frac{\mathrm{d} \vec{X}}{\mathrm{d} t} = \begin{cases}
\dot{\mu} = \dots \quad \text{(inertial force balance)} \\
\dot{\theta} = g(v, \theta, \dots) \quad \text{(state evolution)} \\
\dot{x} = v(\mu, \theta, \dots) \quad \text{(friction law)}
\end{cases}
```
Correspondingly, the solution vector gains an extra component, $\vec{X}(t) = \left[ \mu(t), \theta(t), x(t) \right]^\intercal$.

In order for a friction model to be compatible with an inertial spring-block, it must implement a `_partials` method to compute the partial derivatives of $v$ with respect to the relevant variables. This method takes as an argument the vector $\vec{X}$ and returns $\partial v / \partial \mu$ and the sum of the remaining partial derivative terms ($(\partial v / \partial \theta) \dot{\theta} + \dots$).

The inertial spring-block model is available through `diabayes.forward_models.inertial_springblock`.

```{hint}
To calculate $M$ for your experimental set-up, estimate the mass of the forcing block that is moving (i.e., driven by a shear piston), multiply this by the gravitational acceleration (9.8 m/s²) and divide this by the product of the interface contact area times applies normal stress:

$$
M = \frac{m g}{A \sigma_n}
$$

```


```{rubric} References
```
```{footbibliography}
```
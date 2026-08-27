# Chen-Niemeijer-Spiers

One major limitation of the rate-and-state friction (RSF) model, is that it is a purely empirical framework.
There have been numerous efforts to explain the origins of the (logarithmic) dependencies on time, slip rate, and slip as originally proposed, and yet it has proven difficult to come up with a physical mechanism that correctly describes the wide range of frictional phenomena.
For example, the RSF parameters $a$, $b$, and $D_c$ are presumed to be independent of the instantaneous slip rate, which would give a constant value of the rate dependence $(a - b)$, even though steady-state experiments show that this is far from true (see e.g. Fig. 1 in the section on [friction experiments](../friction_experiments)).

Since this kind of behaviour is essential for the dynamics of earthquake nucleation and rupture, a model based on physical principles would give us much more confidence when trying to extrapolate laboratory observations to natural faults.
This is, in a nutshell, the motivation for the models proposed by _Niemeijer & Spiers_{footcite}`niemeijer2006` and _Chen & Spiers_{footcite}`chen2016`, denoted collectively as the _Chen-Niemeijer-Spiers_ (CNS) model. The remainder of this section will focus on the mathematical formulations rather than the physical processes. For an extensive summary of these models and the experimental observations that preceded them, see _Verberne et al._{footcite}`verberne2020`.

## Original formulation

Just like for the RSF model, the original CNS formulation comprises a friction law (i.e., fault slip rate as a function of stress and other variables) and a state evolution law. While for the RSF model the "state" is a somewhat nebulous concept, in the CNS framework the state is taken as the fault gouge porosity.
This is one of the main advantages of the CNS model: since porosity is a well-defined observable, we can directly see and measure the "state" of a fault in the lab and in rock samples, and we have a clear sense for what controls the evolution of it.

Without going into the derivations, the friction law of the CNS model is written as resulting from the added (parallel) shear rate contributions of granular flow ($\dot{\gamma}_{\text{gr}}$) and a creep process ($\dot{\gamma}_{\text{creep}}$) operating within a gouge layer of thickness $h$:
```{math}
\begin{split}
v(\tau, \phi) = h \left( \dot{\gamma}_{\text{gr}} + \dot{\gamma}_{\text{creep}} \right) \\
\dot{\gamma}_{\text{gr}} = \dot{\gamma}_0 \exp \left( \frac{\tau \left[ 1 - \mu_0 \tan \psi \right] - \sigma \left[ \mu_0 + \tan \psi \right]}{\alpha \left[\sigma + \tau \tan \psi \right]} \right) \\
\dot{\gamma}_{\text{creep}} = Z \tau f(\phi)
\end{split}
```
In these equations, $\tau$ denotes the instantaneous shear stress, $\sigma$ the effective normal stress, $\phi$ the gouge porosity, $\alpha$ a parameter that is similar to the RSF parameter $a$, $\dot{\gamma}_0$ and $\mu_0$ co-dependent parameters similarly defined as the RSF $v_0$ and $\mu_0$, and $Z$ the rate parameter of the creep process.
The two remaining terms are defined as:
```{math}
\begin{split}
\tan \psi = 2 \beta \left( \phi_c - \phi \right) \\
f(\phi) = \frac{\phi - \phi_0}{\phi_c - \phi}
\end{split}
```
The first equation is known in the soil mechanics literature as the _dilatancy angle_, and it controls how much an over-consolidated gouge dilates for an increment of shear strain. It introduces two new parameters, $\beta$ being a geometric constant of order 1, and $\phi_c$ the _critical state porosity_ (the highest possible porosity that a granular gouge can support).
The second equation describes how the nominally imposed stress ($\tau, \sigma$) translates into the average stress supported by individual grain contacts, and can therefore be seen as a porosity-dependent stress amplification factor.
The parameter $\phi_0$ determines the lowest possible porosity (typically ~3%), such that $\phi_0 < \phi < \phi_c$ at all times.
There exist variations to these equations, but these are the ones that are quite convenient for numerical modelling.

Lastly, the state (porosity) evolution law captures the competition of slip-dependent dilatancy caused by granular flow versus time-dependent compaction caused by creep, resulting in both slip and time-dependent state evolution:
```{math}
\begin{split}
\frac{\mathrm{d} \phi}{\mathrm{d} t} = -\left( 1 - \phi \right) \left( \dot{\varepsilon}_{\text{gr}} + \dot{\varepsilon}_{\text{creep}} \right) \\
\dot{\varepsilon}_{\text{gr}} = - \tan \psi \dot{\gamma}_{\text{gr}} \\
\dot{\varepsilon}_{\text{creep}} = Z \sigma f(\phi)
\end{split}
```
Admittedly, these equations seem a lot more complicated than the original RSF formulation, but in return, the CNS model produces a wealth of frictional sliding behaviours that RSF cannot reproduce, which makes the extra effort worth it.
Moreover, there are a few more tweaks that can be applied to obtain a more digestible set of equations and parameters.

## DiaBayes modifications

Without loss of generality, DiaBayes makes several symbolic adjustments to simplify the numerical implementation and to reduce the number of unconstrained parameters for inversion.
Firstly, all stresses are normalised by $\sigma$, distances by $h$, and time is normalised as $t' = \dot{\gamma}_0 t = t v_0 h^{-1}$, which gives:
```{math}
\begin{split}
v'(\mu, \phi) = \exp \left( \frac{\mu \left[ 1 - \mu_0 \tan \psi \right] - \left[ \mu_0 + \tan \psi \right]}{\alpha \left[1 + \mu \tan \psi \right]} \right) + \xi \mu f(\phi) \\
\frac{\mathrm{d} \phi}{\mathrm{d} t'} = -\left( 1 - \phi \right) \left(- \tan \psi \left[ v' - \xi \mu f(\phi) \right] + \xi f(\phi) \right)
\end{split}
```
with $\xi = h Z \left(v_0 \sigma \right)^{-1}$.

From this normalisation exercise, we can easily count the number of invertible (non-dimensional) parameters: $\alpha$, $\beta$, $\mu_0$, $\xi$, and $\phi_c$, assuming that $\phi_0$ is a known constant that is sufficiently small (a few per cent) to not matter much.
Comparing this to RSF, which has the governing parameters $a$, $b$, and $D_c$, the symbolic complexity of the CNS model doesn't seem excessive, especially considering that RSF also requires $\mu_0$ and $v_0$ to be determined for absolute friction values.
However, in contrast to typical RSF inversion practice, $\mu_0$ cannot generally be interpreted as the "initial" friction, e.g. at the start of a velocity step.
This is because the CNS model considers two independent physical processes (granular flow and viscous creep) that both contribute to the slip rate, and correspondingly friction, while $\mu_0$ is exclusively a property of granular flow.
However, for the range of fault slip rates in which granular flow dominates ($\dot{\gamma}_{\text{gr}} \gg \dot{\gamma}_{\text{creep}}$), $\mu_0$ can reasonably be interpreted as an initial or reference friction value.

## Numerical solution strategy

In addition to the normalisation described above, DiaBayes has another trick up its sleeve to improve numerical stability of the numerical integration of the ODE.
Due to the functional form of $f(\phi) \propto \left(\phi_c - \phi \right)^{-1}$, having a singularity at $\phi \rightarrow \phi_c$, numerical integrators tend to struggle to correctly resolve the ODE when $\phi$ approaches $\phi_c$.
In other words, the conventional CNS formulation is [numerically stiff](https://en.wikipedia.org/wiki/Stiff_equation).
Fortunately, a simple change of variables eliminates the singularity; define $t' = \left(\phi_c - \phi \right) r$ giving:
```{math}
\begin{split}
\frac{\mathrm{d} \phi}{\mathrm{d} r} &= \frac{\mathrm{d} t'}{\mathrm{d} r} \frac{\mathrm{d} \phi}{\mathrm{d} t'} \\
&= -\left( 1 - \phi \right) \left(- \tan \psi \left[ v' \left(\phi_c - \phi \right) - \xi \mu \left( \phi - \phi_0 \right) \right] + \xi \left[ \phi - \phi_0 \right] \right)
\end{split}
```
Let it be clear that this expression no longer has any singularities.
Such a manipulation is known as a _Sundman transformation_, and it is commonly used when dealing with planetary orbits to avoid the gravity singularity.
The flip-side is that the ODE is now expressed as a function of $r$ and not $t$, and so it is not immediately obvious what the integration bounds are (_which range of_ $r = t \left( \phi_c - \phi(t) \right)$ _corresponds with_ $t \in [t_0, t_1 )$?).
DiaBayes solves this conundrum by adding an additional equation to the ODE:
```{math}
\frac{\mathrm{d} t'}{\mathrm{d} r} = \left( \phi_c - \phi \right)
```
As the integration marches forward, it continuously tracks $t'$ integrated over $r$ until it exceeds $t_1$, at which point the integrator stops.
The final results of the integration are then interpolated onto the user-requested time grid, so that the results can be correctly interpreted as varying with (physical) time and compared with measurements.
Both the normalisation logic and Sundman transformations are abstracted away from the user, so that the overall user experience is no different between the RSF and CNS formulations.

## Relationship with RSF

Since both the RSF and CNS models aim to describe the same phenomenon (fault friction), there should exist approximate or asymptotic mappings between the two.
In the limit of an infinitesimal slip rate (or stress) perturbation, it has been shown that the two formulations converge.
In this limit, the RSF parameters can be written in terms of CNS parameters, and the RSF state evolution laws (ageing or slip laws) can be expressed as the evolution of porosity.
While the translation between RSF and CNS parameters takes up quite a bit of space on paper (see Table 1 in _Chen et al._{footcite}`chen2017`), the following statements generally hold:

1. The RSF parameter $a$ is proportional to the CNS parameter $\alpha$, and both are typically of the same magnitude.
2. The RSF parameter $b$ is proportional to $\beta \sqrt{\xi}$ (for the definitions of $\tan \psi$ and $f(\phi)$ as implemented in DiaBayes).
3. The RSF parameter $D_c$ is proportional to $h \beta^{-1}$.

It is, of course, not recommended to try to force CNS-like behaviour into a RSF formulation or vice versa.
Previous works listed at the bottom of this page clearly explain the interpretation of the CNS parameters and how to calculate them from e.g. temperature, grain size, and geometry, so there is no reason to compute them using RSF parameter values as an input.

## Example usage

See `examples/cns_model.ipynb` for a hands-on tutorial of forward modelling using the CNS model.

The example code below represents a minimalistic and self-contained case of running a CNS forward model:
```python
import jax.numpy as jnp
import diabyayes as db
from diabayes.forward_models import Forward, cns, cns_porosity, springblock, steady_state_porosity
from diabayes.solver import ODESolver

# Constants
h = 1e-3      # Gouge layer thickness [m]
phi0 = 0.03   # Minimum porosity (not initial porosity!) [-]
v0 = 1e-6     # Reference velocity [m/s]
constants = db.CNSConstants(h=h, phi0=phi0, v0=v0)

# Spring-block constants
k = 1e2       # Stiffness [1/m]
v_lp = 1e-5   # Load-point velocity [m/s]
block_constants = db.SpringBlockConstants(k=k, v_lp=v_lp)

# Parameters (all dimensionless)
alpha = 0.01   # Rate parameter
beta = 0.3     # Dilatancy geometric factor
phi_c = 0.4    # Critical-state porosity
xi = 1e-1      # Creep process rate parameter
mu0 = 0.6      # Reference friction
params = db.CNSParams(alpha=alpha, beta=beta, phi_c=phi_c, xi=xi, mu0=mu0)

# Calculate the steady-state porosity from these parameters
phi_ini = steady_state_porosity(v0, mu0, params, constants)

# Assemble forward model
state_dict = {"phi": cns_porosity}
forward = Forward(
    friction_model=cns,
    state_evolution=state_dict,
    stress_transfer=springblock
)
solver = ODESolver(forward_model=forward)

# Set initial values
forward.set_initial_values(mu=mu0, phi=phi_ini)
y0 = forward.variables

# Run the forward simulation
t = jnp.linspace(0, 1000., 1000)
result = solver.solve_forward(
    t, y0, params=params,
    friction_constants=constants,
    block_constants=block_constants
)
```

```{rubric} References
```
```{footbibliography}
```
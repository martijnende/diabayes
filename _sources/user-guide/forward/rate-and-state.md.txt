# Rate-and-state friction

Rate-and-state friction (RSF) is a well-established formulation that describes a wide range of laboratory and natural phenomena. It is therefore commonly used to model laboratory friction experiments and natural earthquakes{footcite}`marone1998`. The formulation itself is empirical, and the governing parameters have long been subject of debate over their physical interpretation. Nonetheless, there is sufficient consistency between the parameters for different types of experiments and conditions to warrant a comparison, and analysis of their various dependencies (most notably temperature, but also stress, slip rate, chemistry, etc.). The RSF formulation consists of a friction law with a characteristic exponential dependency, and is accompanied by a state evolution law that prescribes the time-evolution of a state parameter (usually denoted by $\theta$) that keeps track of the slip history.

DiaBayes currently implements the two most common flavours of rate-and-state friction, as well as two state evolution laws.

## Classical rate-and-state friction

The classical RSF law is often written as:
```{math}
:label: RSF
\mu(v, \theta) = \mu_0 + a \ln \left( \frac{v}{v_0} \right) + b \ln \left( \frac{v_0 \theta}{D_c} \right)
```
with $\mu$ the instantaneous friction coefficient, which is a function of the instantaneous slip rate $v$ and "state" $\theta$. The governing parameters are the reference value of friction $\mu_0$ at the reference slip rate $v_0$, the direct effect parameter $a$, the evolution effect parameter $b$, and the characteristic slip distance $D_c$. In a laboratory setting, $mu_0$ and $v_0$ are often taken directly from a steady-state measurement, leaving only $a$, $b$, and $D_c$ as the parameters of interest.

In DiaBayes, Eq. {eq}`RSF` is re-written in terms of $v$:
```{math}
:label: RSF2
v = v_0 \exp \left( \frac{1}{a} \left[ \mu - \mu_0 - b \ln \left( \frac{v_0 \theta}{D_c} \right) \right] \right)
```

The implementation of the classical RSF formulation is available through `diabayes.forward_models.rsf`.

## Regularised rate-and-state friction

Owing to the logarithms used in the classical RSF formulation, the slip rate $v$ is required to be strictly positive. This requirement can be problematic in simulations where the shear stress (and corresonding slip rate) can change sign or become zero. To alleviate this, a regularised form{footcite}`lapusta2000` can be adopted with the same asymptotic behaviour as Eq. {eq}`RSF2`:
```{math}
:label: RSF_reg
v = 2 v_0 \sinh \left\{ \frac{\mu}{a} \exp \left( - \frac{1}{a} \left[\mu_0 + b \ln \left( \frac{v_0 \theta}{D_c} \right) \right] \right) \right\}
```

The implementation of the regularised RSF formulation is available through `diabayes.forward_models.rsf_regular`.

## Ageing law state evolution

The _ageing law_ derives its name from the fact that the incremental growth of state is proportional to time ("age"), and decays with slip:
```{math}
:label: ageing_law
\frac{\mathrm{d} \theta}{\mathrm{d} t} = 1 - \frac{v \theta}{D_c}
```
At steady-state $\dot{\theta} = 0$, hence correspondingly $\theta_{ss} = D_c v^{-1}$. Since the state variable cannot be directly observed in laboratory experiment, it is often assumed that prior to a velocity-step, the initial value of state is $\theta(t=0) = \theta_{ss}$.

The implementation of the ageing law is available through `diabayes.forward_models.ageing_law`.

## Slip law state evolution

An alternative formulation to the ageing law is the _slip law_, which derives its name from the fact that it exclusively evolves with slip:
```{math}
:label: slip_law
\frac{\mathrm{d} \theta}{\mathrm{d} t} = - \frac{v \theta}{D_c} \ln \left(\frac{v \theta}{D_c} \right)
```
Like for the ageing law, the slip law is characterised by a steady-state value of $\theta_{ss} = D_c v^{-1}$.

The implementation of the slip law is available through `diabayes.forward_models.slip_law`.

```{rubric} References
```
```{footbibliography}
```
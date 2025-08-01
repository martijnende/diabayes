# Getting started

## Installation guide

It is recommended to create a dedicated Python environment using either `conda` or `venv`. DiaBayes will install numerous JAX-related packages that might conflict with your existing Python environments, especially when GPU support is enabled (which installs CUDA binaries). To create a new environment using `conda`, use:
```bash
conda create -n diabayes python=3.11 pip
conda activate diabayes
```
You can then install the latest version using `pip` directly from GitHub:
```bash
pip install "git+https://github.com/martijnende/diabayes.git#egg=diabayes"
```
To enable Nvidia GPU support (CUDA version 12):
```bash
pip install "git+https://github.com/martijnende/diabayes.git#egg=diabayes[gpu]"
```
If you plan to make direct changes to the code base, you can clone the repository and install from the local repository:
```bash
git clone https://github.com/martijnende/diabayes.git
cd diabayes
pip install -e .[gpu]
```
Note that you'll need to repeat `pip install` every time you make changes. If you also plan to push these changes to the original DiaBayes project, please include the development and documentation tools:
```bash
pip install .[gpu,docs,dev]
```
The `dev` packages include [`black`](https://github.com/psf/black) and [`isort`](https://github.com/PyCQA/isort), which (re-)structure your code to a consistent format whenever you execute these in the repository root directory.

## Basic usage

### Variables, parameters, constants

DiaBayes can accommodate several physical models that each come with their own set of parameters. To assist in the bookkeeping of the variables, parameters, and constants, DiaBayes uses special containers that allow access to their contents by name. For example, variables are declared as:
```python
import diabayes as db
y = db.Variables(mu=0.6, state=10.5)
```
The container items can then be accessed through `y.mu` and `y.state`. If you'we used Python's `namedtuple` collection before then this should feel familiar. Invertible parameters and constants are declared in a similar way:
```python
# RSF parameters
params = db.RSFParams(a=a, b=b, Dc=Dc)
# RSF constants
constants = db.RSFConstants(v0=v0, mu0=mu0)
# Spring-block constants
block_constants = db.SpringBlockConstants(k=k, v_lp=v1)
```

### Forward models

The physics that underlies your experiment is encoded by a _forward model_. A DiaBayes `Forward` model is composed of three components:

1. A friction law, which is an equation that takes the friction and "state" as an input, and returns the instanteneous slip rate as an output, i.e. $v(t) = f(\mu(t), \theta(t))$.
2. A state evolution law, which returns the time derivative of the "state" variable.
3. A stress transfer model, which describes how the local slip rate results in a change in shear stress on the fault interface.

These three components can be mixed and matched according to the user's needs. For example, one could compile a forward model with regularised rate-and-state friction, the slip law for state evolution, and a non-inertial spring-block analogue for the stress transfer. Or one could pick the _Chen-Niemeijer-Spiers_ friction model with its corresponding state (porosity) evolution law coupled with an inertial spring-block to simulate stick-slip motion. See the [User Guide on forward modelling](user-guide/forward/index) for more information about the various forward model components that have been implemented.

Here is an example of a classical rate-and-state friction model combined with the ageing law and a non-inertial spring-block:
```python
from diabayes.forward_models import Forward, rsf, ageing_law, springblock
forward = Forward(friction_model=rsf, state_evolution=ageing_law, block_type=springblock)
```

This forward model can then be handed over to a solver that takes care of the forward (and inverse) modelling:
```python
from diabayes.solver import ODESolver
solver = ODESolver(forward_model=forward)
```

The `ODESolver` class contains a number of public and private methods that take care of unpacking the various containers (`Variables`, `RSFConstants`, etc.) and casting the forward models in a common format to make them compatible for an ODE solver. Getting a forward solution from DiaBayes is a single function call:
```python
result = solver.solve_forward(
    t=t, y0=y0, params=params,
    friction_constants=constants,
    block_constants=block_constants
)
```
This `result` is a `Variables` container that contains the time series of the relevant variables (e.g. friction and state for rate-and-state friction) that can be accessed e.g. through `result.mu`, `result.state`, etc.

### Inversion

DiaBayes currently has two flavours if inversion implemented. The _maximum-likelihood_ inversion uses the Levenberg-Marquardt method with adjoint back-propagation through the ODE to search for a single set of physical parameters that minimise the misfit between the predicted friction curve and the measured one. The second flavour is a form of _Bayesian inversion_, which estimates the probability that a given set of parameters describes the observed friction within the measurement uncertainties. See the [User Guide on inverse modelling](user-guide/inversion/index) for an in-depth discussion of these two inversion approaches.

Performing a maximum-likelihood inversion involves a single function call that has a similar signature as the forward solver:
```python
mu_measured = ...
initial_guess_params = ...
inv_result = solver.max_likelihood_inversion(
    t=t, mu=mu_measured,
    params=initial_guess_params,
    friction_constants=constants,
    block_constants=block_constants
)
```

The return value `inv_result` is an [`optimistix.Solution`](https://docs.kidger.site/optimistix/api/solution/#optimistix.Solution) object that contains the maximum-likelihood solution of the parameters, as well as diagnostic information. This maximum-likelihood estimate can then be used for subsequent Bayesian inversion to estimate the uncertainties and trade-offs in the parameters:
```python
params_inv = inv_result.value
noise_amplitude = 1e-3  # Estimate of the measurement uncertainty
bayesian_result = solver.bayesian_inversion(
    t=t, mu=mu_measured, noise_std=noise_amplitude, 
    params=params_inv, friction_constants=constants, 
    block_constants=block_constants, 
    Nparticles=1500, Nsteps=100, rng=42
)
```
In this example, the posterior distribution is estimated by means of a particle swarm consisting of 1500 particles, which converge to an equilibrium over 100 steps.

The return value of `bayesian_inversion` is a `BayesianSolution` object, which not only contains the posterior samples, but also several diagnostic and visualisation routines. To evaluate whether the inversion converged to an equilibrium state, use:
```python
bayesian_result.plot_convergence();
```
To visualise the posterior distributions over the inverted parameters, use:
```python
bayesian_result.cornerplot();
```
To draw samples from the posterior distribution (`samples`), and their corresponding friction time series (`sample_results`), use:
```python
samples, sample_results = bayesian_result.sample(
    solver, y0, t,
    constants, 
    block_constants,
    nsamples=100
)
```

## Examples

See `examples/simple_example.ipynb` for a basic tutorial on how to use DiaBayes in practice.

## Getting help

If it is unclear on how to use the DiaBayes API, if you encountered a bug, or if you want to contribute a new feature, head over to the [DiaBayes Community page](community). There you will find useful links and resources, frequently encountered issues, and suggestions.
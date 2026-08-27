# Getting started

## Installation guide

It is recommended to create a dedicated Python environment using either `conda` or `venv`. DiaBayes will install numerous JAX-related packages that might conflict with your existing Python environments, especially when GPU support is enabled (which installs CUDA binaries).

### Pip / conda

If you use [conda](https://conda.io), create or switch to the desired environment:
```bash
conda create -n diabayes python=3.14 pip
conda activate diabayes
```
Install the latest version with CPU support from GitHub:
```bash
pip install git@github.com:martijnende/diabayes.git  # Base version
pip install "diabayes[gpu,gui] @ git+github.com:martijnende/diabayes.git"  # Base version + GPU support + GUI
```
You can also install from a local directory after cloning the repository
```bash
git clone git@github.com:martijnende/diabayes.git && cd diabayes
pip install .           # Base version
pip install .[gpu,gui]  # Base version + GPU support + GUI
```
If you plan to contribute to the development of this package, please include the development tools:
```bash
pip install .[dev]
```

### UV

Using [Astral UV](https://docs.astral.sh/uv/), you create a dedicated environment with:
```bash
git clone git@github.com:martijnende/diabayes.git && cd diabayes
uv sync                          # Install just the base environment
uv sync --extra gui --extra gpu  # Equivalent to pip install .[gui,gpu]
uv sync --extra-all              # Install everything
```
When working from a different workspace directory, you can instruct UV to use a specific virtual environment by setting an environment variable:
```bash
export VIRTUAL_ENV="/path/to/diabayes/.venv/"   # For bash, zsh, ...
set -gx VIRTUAL_ENV "/path/to/diabayes/.venv/"  # For fish
```
You can then run the DiaBayes GUI from a different workspace directory as:
```bash
cd /path/to/workspace
uv run diabayes init .
uv run diabayes run
```

### A note on GPU support

Currently DiaBayes only supports Nvidia GPUs with Cuda version 12, simply because that is what I have on my local machine.
Feel free to change the `gpu` option in `pyproject.toml` to change the JAX version that is supported by your hardware.
See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for detailed instructions.
Note that the GPU is only used for Bayesian inference (SVI); all other components of the software use the CPU.

## Basic usage

```{note}
To clarify the notation used throughout the documentation: 
- A _variable_ is a quantity that potentially varies over time.
- A _parameter_ is a constant quantity that one might want to invert for.
- A _constant_ is a user-defined constant quantity that remains untouched throughout inversion procedures.
```

### Forward models

The physics that underlies your experiment is encoded by a _forward model_. A DiaBayes `Forward` model is composed of three components:

1. A friction law, which is an equation that takes the friction and one or more "state" variables as an input, and returns the instantaneous slip rate as an output, i.e. $v(t) = f(\mu(t), \theta(t), \phi(t), \dots)$.
2. A state evolution law, which returns the time derivative of the "state" variable(s).
3. A stress transfer model, which describes how the local slip rate results in a change in shear stress on the fault interface.

These three components can be mixed and matched according to the user's needs. For example, one could compile a forward model with regularised rate-and-state friction, the slip law for state evolution, and a non-inertial spring-block analogue for the stress transfer. Or one could pick the _Chen-Niemeijer-Spiers_ friction model with its corresponding state (porosity) evolution law coupled with an inertial spring-block to simulate stick-slip motion. See the [User Guide on forward modelling](user-guide/forward/index) for more information about the various forward model components that have been implemented.

Here is an example of a classical rate-and-state friction model combined with the ageing law and a non-inertial spring-block:
```python
from diabayes.forward_models import Forward, rsf, ageing_law, springblock
state_dict = {"theta": ageing_law}
forward = Forward(friction_model=rsf, state_evolution=state_dict, stress_transfer=springblock)
```
The `rsf` friction model uses only a single state variable (`theta`), but other friction models could accept several. The structure of the `state_dict` is `{"variable_name1": variable_evolution1, "variable_name2": variable_evolution2, ...}`. In other words, every state variable has a unique name and exactly one function that describes the evolution of this variable. The variable names need to correspond with what is expected by the friction law; see the documentation of each friction law for a description.

This forward model can then be handed over to a solver that takes care of the forward (and inverse) modelling:
```python
from diabayes.solver import ODESolver
solver = ODESolver(forward_model=forward)
```

Since the forward/inverse models need to accommodate any kind of physics that come with their own quantities, DiaBayes implements special containers that allows access to these quantities by name (rather than by position inside an array, which is not known ahead of time). For constant quantities (parameters and constants), the user can import the relevant containers and set these quantities directly:
```python
import diabayes as db

# RSF parameters
params = db.RSFParams(a=a, b=b, Dc=Dc)
# RSF constants
constants = db.RSFConstants(v0=v0, mu0=mu0)
# Spring-block constants
block_constants = db.SpringBlockConstants(k=k, v_lp=v1)
```

Time-dependent variables are a bit special, and so they need to be set through the `Forward` class:
```python
theta0 = Dc / v0
forward.set_initial_values(mu=mu0, theta=theta0)
y0 = forward.variables
```
Since we initially provided just one state evolution equation, there is only one state variable to specify in addition to the friction coefficient. In this example `forward.variables` points to a `Variables` class, and we can access the various quantities like `params.Dc`, `block_constants.k`, `y0.theta`, etc.

The last step is to get a forward solution from DiaBayes, which is a single function call:
```python
result = solver.solve_forward(
    t=t, y0=y0, params=params,
    friction_constants=constants,
    block_constants=block_constants
)
```
This `result` is a `Variables` container that contains the time series of the relevant variables (e.g. friction and state for rate-and-state friction) that can be accessed e.g. through `result.mu`, `result.theta`, etc.

### Inversion

DiaBayes currently has two flavours if inversion implemented. The _maximum-likelihood_ inversion uses the Levenberg-Marquardt method with adjoint back-propagation through the ODE to search for a single set of physical parameters that minimise the misfit between the predicted friction curve and the measured one. The second flavour is a form of _Bayesian inversion_, which estimates the probability that a given set of parameters describes the observed friction within the measurement uncertainties. See the [User Guide on inverse modelling](user-guide/inversion/index) for an in-depth discussion of these two inversion approaches.

Performing a maximum-likelihood inversion involves a single function call that has a similar signature as the forward solver:
```python
mu_measured = ...
initial_guess_params = ...
inv_result = solver.max_likelihood_inversion(
    t=t, mu=mu_measured,
    y0=y0, params=initial_guess_params,
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
    y0=y0, params=params_inv, friction_constants=constants, 
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

### Using the GUI

DiaBayes comes with a web-based GUI that exposes a number of basic features, like data visualisation, forward modelling, and inversion, using the more conventional friction models.
This GUI persistently stores its environment state so that you can resume the analysis over consecutive sessions.
You can also export and share this environment state, for example as a supplementary material to a publication.

To enable the GUI, make sure to install it first (see the next section).
Then, to initialise a workspace, execute:
```bash
cd /path/to/workspace
diabayes init .  # Initialise the environment (run only once per environment)
diabayes run     # Start the GUI server
```
Point your browser at the URL printed after the `run` command (default: `http://127.0.0.1:5000`).

The `init` command will produce a default `workspace.toml` file with settings that can be adjusted by the user (requires relaunching the server to take effect).

## Examples

See `examples/simple_example.ipynb` for a basic tutorial on how to use DiaBayes in practice.

## Getting help

If it is unclear on how to use the DiaBayes API, if you encountered a bug, or if you want to contribute a new feature, head over to the [DiaBayes Community page](community). There you will find useful links and resources, frequently encountered issues, and suggestions.

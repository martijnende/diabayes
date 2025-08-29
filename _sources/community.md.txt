# DiaBayes Community

## Frequently Asked Questions

### The maximum-likelihood inversion crashes before converging

To ensure that everything is wired-up correctly, check that you are able to run a forward simulation with your specified forward model. If that works well, make sure that the initial guess of the parameters produces a friction curve that is sufficiently close to the observed friction. For parameter estimates that are too far off, the Gauss-Newton step of the Levenberg-Marquardt algorithm may fail.

### I am getting a mysterious error

Under the hood, most of the computations are handled by JAX, which compiles and ports the code to a GPU or multi-core CPU. One disadvantage of this automatic optimisation and acceleration, is that the error messages that are being produced tend to be rather cryptic, and often the final ``Exception`` is not at all informative of the code segment that causes trouble. To facilitate debugging, you can turn off the _Just-in-Time_ (JIT) compiler by including the following at the top of your script:
```python
import jax
jax.config.update("jax_disable_jit", True)
```
If NaNs are produced, it may also help to turn on NaN debugging:
```python
jax.config.update("jax_debug_nans", True)
```
Note that ``jax.config`` should be set only at the start of your script before any JAX routines are called. The configuration will stay enabled until the script terminates.

## How to cite?

A publication describing this software package is underway. Until then, feel free to refer to the GitHub repository: https://github.com/martijnende/diabayes

## How to contribute?

If you have ideas for new features to include, you can submit an issue on [GitHub](https://github.com/martijnende/diabayes) to discuss it. See [the User Guide](user-guide/adding_features.md) for general guidelines on how to extend DiaBayes with new physics. Contributed example Jupyter Notebooks and documentation improvements are also welcome.

## Useful resources

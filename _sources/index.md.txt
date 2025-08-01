# DiaBayes documentation

DiaBayes (pronounced like the mafic rock type *diabase*) is a collection of Python tools for forward and inverse modelling of rock friction experiments. It is designed to be modular in terms of the physical components that define a forward problem, so that the user can mix and match different types of physics, e.g. classical rate-and-state friction combined with a more exotic state evolution law, or combining the *Chen-Niemeijer-Spiers* model with an inertial spring-block. These models can be solved in a forward sense (generating a friction time series), or in the inverse sense (estimating the parameters that generated your measured friction). To greatly accelerate the maximum-likelihood and Bayesian inversion, DiaBayes is optimised for GPU execution using JAX and several high-performance libraries (Diffrax, Optax, and Optimistix, to name a few).

See the getting started section for a quick overview of the installation options and basic usage. A self-contained example Jupyter Notebook is included in the GitHub repo (`examples/simple_inversion.ipynb`). Keep reading for more detailed background on the modelling procedures and implementation.

````{grid} 1 2 2 2
:gutter: 4
:padding: 2 2 0 0

```{grid-item-card} Getting Started
:link: getting-started
:link-type: doc
The _Getting Started_ guide covers the installation options and a few practical example usage cases. Start here if you're new to DiaBayes.
```

```{grid-item-card} User Guide
:link: user-guide/index
:link-type: doc
The _User Guide_ explains a number of core concepts in depth, with detailed descriptions of commonly-used functionalities.
```

```{grid-item-card} API Reference
:link: api/index
:link-type: doc
The _API Reference Guide_ is auto-generated based on the docstrings of all the DiaBayes classes and routines. 
```

```{grid-item-card} DiaBayes Community
:link: community
:link-type: doc
Found a bug? Want to contribute? How to cite? Go here to find out more.
```
````

```{toctree}
:maxdepth: 1
:hidden:

getting-started
user-guide/index
api/index
community
```

# DiaBayes: rock friction inversion tools

[![GitHub Release](https://img.shields.io/github/release/martijnende/diabase.svg?style=flat)]() 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/martijnende/diabayes/blob/main/examples/google_colab.ipynb)
[![tests](https://github.com/martijnende/diabayes/actions/workflows/python-test.yml/badge.svg)](https://github.com/martijnende/diabayes/actions/workflows/python-test.yml)
[![documentation](https://github.com/martijnende/diabayes/actions/workflows/build-docs.yml/badge.svg)](https://martijnende.github.io/diabayes)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- [![PyPi Version](https://img.shields.io/pypi/v/diabayes.svg)](https://pypi.python.org/pypi/diabayes/) -->

[Documentation](https://martijnende.github.io/diabayes) | [Example usage](#example-usage) | [Installation](#installation) | [How to cite?](#how-to-cite)

## Example usage

See `examples/simple_example.ipynb` for a self-contained Jupyter notebook that illustrates forward and inverse modelling.

## Installation

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

### A note on GPU support

Currently DiaBayes only supports Nvidia GPUs with Cuda version 12, simply because that is what I have on my local machine.
Feel free to change the `gpu` option in `pyproject.toml` to change the JAX version that is supported by your hardware.
See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for detailed instructions.
Note that the GPU is only used for Bayesian inference (SVI); all other components of the software use the CPU.

## How to cite?

A publication describing this software package is underway. Until then, feel free to refer to this repository.

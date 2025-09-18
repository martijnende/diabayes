# DiaBayes: rock friction inversion tools

[![GitHub Release](https://img.shields.io/github/release/martijnende/diabase.svg?style=flat)]() 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/martijnende/diabayes/blob/docs/colab/examples/google_colab.ipynb)
[![tests](https://github.com/martijnende/diabayes/actions/workflows/python-test.yml/badge.svg)](https://github.com/martijnende/diabayes/actions/workflows/python-test.yml)
[![documentation](https://github.com/martijnende/diabayes/actions/workflows/build-docs.yml/badge.svg)](https://martijnende.github.io/diabayes)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- [![PyPi Version](https://img.shields.io/pypi/v/diabayes.svg)](https://pypi.python.org/pypi/diabayes/) -->

[Documentation](https://martijnende.github.io/diabayes) | [Example usage](#example-usage) | [Installation](#installation) | [How to cite?](#how-to-cite)

## Example usage

See `examples/simple_example.ipynb` for a self-contained Jupyter notebook that illustrates forward and inverse modelling.

## Installation

Installing the latest version with CPU support from GitHub:
```bash
pip install git@github.com:martijnende/diabayes.git
```
Installing from a local repository with Nvidia GPU support (CUDA version 12):
```bash
cd /path/to/diabase && pip install .[gpu]
```
If you plan to contribute to the development of this package, please include the development tools:
```bash
pip install .[gpu,dev]
```

## How to cite?

A publication describing this software package is underway. Until then, feel free to refer to this repository.

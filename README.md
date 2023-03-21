# Collective dynamics in simple active matter
### Yann-Edwin Keta — L2C, Université de Montpellier — 2020

## Introduction

This repository contains scripts, for simulation and analysis purposes, developed for my PhD project (see my [website](https://yketa.xyz)), concerned with the emergence of large collective motion in simple model of active matter. It is derived from the repository **[yketa/active_work](https://github.com/yketa/active_work)**.

Simulation and cloning scripts are written in C++. Wrapper scripts to launch the latter are written in Python, and other Python classes and functions are available to read and analyse the generated data.

A library of C++ functions ([`pycpp.cpp`](https://github.com/yketa/coll_dyn_activem/blob/master/pycpp.cpp)) is implemented in [`pycpp.py`](https://github.com/yketa/coll_dyn_activem/blob/master/pycpp.py), given that `_pycpp.so` is compiled (this necessitates the [`GNU Science Library`](https://www.gnu.org/software/gsl/) and [`pybind11`](https://pybind11.readthedocs.io)).

While C++ files can be quite cumbersome, Python wrappers are hopefully more readable and commented enough so that their functioning can be easily understood.

## Requirements

All code was developed and tested on 64-bit linux. C++ cloning scripts necessitate `OpenMP`. Python scripts are written for `python3.*`, import the `coll_dyn_activem` package which necessitates the directory containing this repository to be added to the `$PYTHONPATH`, e.g. by executing
```
echo "export PYTHONPATH=\$PYTHONPATH:${PWD}/.." >> ~/.bashrc
```
from this directory, and rely on the following packages:

- `matplotlib`: plotting,
- `seaborn`: color palettes,
- `numpy`: mathematical functions and array manipulation,
- `scipy`: various methods and special functions,
- `freud-analysis`: analysis of MD simulations,
- `fastkde`: kernel density estimation ([`scde.py`](https://github.com/yketa/coll_dyn_activem/blob/master/scde.py)),

which can be installed by running [`pip.sh`](https://github.com/yketa/coll_dyn_activem/blob/master/pip.sh).

Production of movies, via [`frame.py`](https://github.com/yketa/coll_dyn_activem/blob/master/frame.py), necessitates `ffmpeg` — though other functionalities of the former can be used without the latter.

Memory error detection and profiling, using `make memcheck` and `make massif` (see [`Makefile`](https://github.com/yketa/coll_dyn_activem/blob/master/Makefile)), necessitates `valgrind`.

Minimisation procedures from [`ALGLIB`](https://www.alglib.net/download.php) are used in [`alglib.hpp`]((https://github.com/yketa/coll_dyn_activem/blob/master/alglib.hpp) — the library must be downloaded and added as `alglib` to the [`g++` path](https://commandlinefanatic.com/cgi-bin/showarticle.cgi?article=art026).

# Collective dynamics in simple active matter
### Yann-Edwin Keta — L2C, Université de Montpellier — 2020

## Introduction

This repository contains scripts, for simulation and analysis purposes, developed for a PhD project, detailed in **[this wiki](https://yketa.github.io/PhD_Wiki)**, concerned with the emergence of large collective motion in simple model of active matter. It is derived from the repository **[yketa/active_work](https://github.com/yketa/active_work)**.

Simulation and cloning scripts are written in C++. Wrapper scripts to launch the latter are written in Python, and other Python classes and functions are available to read and analyse the generated data.

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

## Execution

Compilation of all relevant executables, using `g++`, is possible by running [`compile.sh`](https://github.com/yketa/coll_dyn_activem/blob/master/compile.sh) — which essentially performs all relevant `make` commands (see [`Makefile`](https://github.com/yketa/coll_dyn_activem/blob/master/Makefile)).

Given these have been compiled, they can be executed with the Python scripts listed below.

### Simulations of ABPs

ABP model and simulation procedure is detailed in [this tiddler](https://yketa.github.io/DAMTP_MSC_2019_Wiki/#Active%20Brownian%20particles).

- Simulations with custom relations between parameters are launched using [`launch.py`](https://github.com/yketa/coll_dyn_activem/blob/master/launch.py).
- Simulations with custom relations between parameters and for different values of the torque parameter are launched using [`launchG.py`](https://github.com/yketa/coll_dyn_activem/blob/master/launchG.py).
- Simulations of general ABPs are launched using [`launch0.py`](https://github.com/yketa/coll_dyn_activem/blob/master/launch0.py).

### Simulations of interacting Brownian rotors

Interacting Brownian rotors model is detailed in [this tiddler](https://yketa.github.io/DAMTP_MSC_2019_Wiki/#N-interacting%20Brownian%20rotors).

- Simulations are launched using [`launchR.py`](https://github.com/yketa/coll_dyn_activem/blob/master/launchR.py).

### Cloning of ABPs

Principle and computation scheme of the scaled cumulant generating function (SCGF) of the active work and corresponding averages in the biased ensemble are detailed in [this tiddler](https://yketa.github.io/DAMTP_MSC_2019_Wiki/#ABP%20cloning%20algorithm).

- Cloning of trajectories of ABPs systems with custom relations between parameters, and biased with respect to either the polarisation or the active work, are launched using [`cloning.py`](https://github.com/yketa/coll_dyn_activem/blob/master/cloning.py).

### Cloning of non-interacting Brownian rotors

Principle and computation scheme of the scaled cumulant generating function (SCGF) of the (squared) polarisation and corresponding averages in the biased ensemble are detailed in [this tiddler](https://yketa.github.io/DAMTP_MSC_2019_Wiki/#Brownian%20rotors%20cloning%20algorithm).

- Cloning of trajectories of Brownian rotors are launched using [`cloningR.py`](https://github.com/yketa/coll_dyn_activem/blob/master/cloningR.py).

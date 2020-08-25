#!/bin/bash -e

# Compile all executables.

# move to script directory
cd "$( dirname "${BASH_SOURCE[0]}" )"

# use Heun's integrations scheme
USE_HEUN=no

# ROTORS
make clean && ROTORS=yes HEUN=$USE_HEUN make;

# ROTORS CLONING
make clean && CLONINGR=yes BIAS=0 HEUN=$USE_HEUN make;                          # biasing with polarisation
make clean && CLONINGR=yes BIAS=1 CONTROLLED_DYNAMICS=no HEUN=$USE_HEUN make;   # biasing with squared polarisation without controlled dynamics
make clean && CLONINGR=yes BIAS=1 CONTROLLED_DYNAMICS=yes HEUN=$USE_HEUN make;  # biasing by squared polarisation with controlled dynamics

for CL in yes no; do  # with and without cell lists

  # SIMULATIONS
  for S0 in yes no; do  # general ABPs and custom model
    make clean && SIM0=$S0 CELLLIST=$CL HEUN=$USE_HEUN make;
  done

  # ABP CLONING
  # POLARISATION
  for CD in yes no; do  # with and without controlled dynamics
    make clean && CLONING=yes BIAS_POLARISATION=yes CONTROLLED_DYNAMICS=$CD CELLLIST=$CL HEUN=$USE_HEUN make;
  done
  # ACTIVE WORK
  for CD in {0..3}; do  # different controlled dynamics
    make clean && CLONING=yes BIAS_POLARISATION=no CONTROLLED_DYNAMICS=$CD CELLLIST=$CL HEUN=$USE_HEUN make;
  done

done

# CLEAN
make clean

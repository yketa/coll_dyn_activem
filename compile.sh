#! /bin/bash -e

# Compile all executables.

# move to script directory
cd "$( dirname "${BASH_SOURCE[0]}" )"

# use Heun's integrations scheme
USE_HEUN=yes

# python/C++ interface
make clean && make _pycpp.so

# ADD
for MD in yes no; do  # using MD or CG
  for LIMIT in yes no; do # limit on displacements (CG) or iterations (MD)
    for NEXT in yes no; do  # next frame propulsion
      make clean && ADD=yes ADD_MD=$MD ADD_NO_LIMIT=$LIMIT ADD_NEXT_PROPULSION=$NEXT CELLLIST=yes make;
    done
  done
done

# ROTORS
make clean && ROTORS=yes HEUN=$USE_HEUN make;

# ROTORS CLONING
make clean && CLONINGR=yes BIAS=0 HEUN=$USE_HEUN make;                          # biasing with polarisation
make clean && CLONINGR=yes BIAS=1 CONTROLLED_DYNAMICS=no HEUN=$USE_HEUN make;   # biasing with squared polarisation without controlled dynamics
make clean && CLONINGR=yes BIAS=1 CONTROLLED_DYNAMICS=yes HEUN=$USE_HEUN make;  # biasing by squared polarisation with controlled dynamics

for CL in yes no; do  # with and without cell lists

  # CUSTOM ABP MODEL SIMULATIONS
  make clean && SIM=dat CELLLIST=$CL HEUN=$USE_HEUN make;

  # GENERAL SIMULATIONS
  for TY in ABP AOUP; do  # types of active particles
    for SI in dat0 datN; do # types of simulation
      make clean && SIM=$SI TYPE=$TY CELLLIST=$CL HEUN=$USE_HEUN make;
    done
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

"""
Module launch launches simulations with custom relations between parameters.
"""

from coll_dyn_activem.exponents import float_to_letters
from coll_dyn_activem.init import get_env, Time
from coll_dyn_activem.read import Dat

from numpy.random import randint

from os import path
from subprocess import Popen, DEVNULL
from sys import stderr

# FUNCTIONS AND CLASSES

def filename(N, phi, lp, g, launch):
    """
    Name of simulation output files.

    Parameters
    ----------
    N : int
        Number of particles in the system.
    phi : float
        Packing fraction.
    lp : float
        Dimensionless persistence length.
    g : float
        Torque parameter.
    launch : int
        Launch identifier.

    Returns
    -------
    name : str
        File name.
    """

    return 'N%s_D%s_L%s_G%s_E%s.dat' % tuple(map(float_to_letters,
        (N, phi, lp, g, launch)))

# DEFAULT VARIABLES

_N = 10     # default number of particles in the system
_lp = 40    # default dimensionless persistence length
_phi = 0.65 # default packing fraction

_seed = randint(1e7)    # default random seed
_dt = 1e-3              # default time step
_Niter = 5e4            # default number of iterations

_launch = 0 # default launch identifier

_nWork = 0  # default number of frames on which to sum the active work before dumping (0 => nWork = lp/dt)
_dump = 1   # default boolean to indicate to dump positions and orientations to output file
_period = 1 # default period of dumping of positions and orientations in number of frames

_N_cell = 100                                                           # number of particles above which simulations should be launched with a cell list
_exec_dir = path.join(path.dirname(path.realpath(__file__)), 'build')   # default executable directory
_exec_name = ['simulation', 'simulation_cell_list']                     # default executable name without and with a cell list

_out_dir = _exec_dir    # default simulation output directory

# SCRIPT

if __name__ == '__main__':

    # VARIABLE DEFINITIONS

    # INPUT FILE PARAMETERS
    inputFilename = get_env('INPUT_FILENAME', default='', vartype=str)  # input file from which to copy data
    inputFrame = get_env('INPUT_FRAME', default=0, vartype=int)         # frame to copy as initial frame

    if inputFilename == '':

        # SYSTEM PARAMETERS
        N = get_env('N', default=_N, vartype=int)                   # number of particles in the system
        lp = get_env('LP', default=_lp, vartype=float)              # dimensionless persistence length
        phi = get_env('PHI', default=_phi, vartype=float)           # packing fraction
        g = get_env('TORQUE_PARAMETER', default=0, vartype=float)   # torque parameter

    else:

        # SYSTEM PARAMETERS
        with Dat(inputFilename, loadWork=False) as dat: # data object
            N = dat.N                                   # number of particles in the system
            lp = dat.lp                                 # dimensionless persistence length
            phi = dat.phi                               # packing fraction
            g = dat.g                                   # torque parameter

    # SIMULATION PARAMETERS
    seed = get_env('SEED', default=_seed, vartype=int)      # random seed
    dt = get_env('DT', default=_dt, vartype=float)          # time step
    Niter = get_env('NITER', default=_Niter, vartype=int)   # number of iterations

    # NAMING PARAMETERS
    launch = get_env('LAUNCH', default=_launch, vartype=float)  # launch identifier

    # OUTPUT PARAMETERS
    nWork = get_env('NWORK', default=_nWork, vartype=int)       # number of frames on which to sum the active work before dumping (0 => nWork = lp/dt)
    dump = get_env('DUMP', default=_dump, vartype=int)          # boolean to indicate to dump positions and orientations to output file
    period = get_env('PERIOD', default=_period, vartype=int)    # period of dumping of positions and orientations in number of frames

    # EXECUTABLE PARAMETERS
    exec_dir = get_env('EXEC_DIR', default=_exec_dir, vartype=str)      # executable directory
    exec_name = get_env('EXEC_NAME', default=_exec_name[N >= _N_cell],  # executable name
        vartype=str)

    # OUTPUT FILE PARAMETERS
    out_dir = get_env('OUT_DIR', default=_out_dir, vartype=str) # simulation output directory
    out_file = filename(N, phi, lp, g, launch)                  # simulation output file name

    # LAUNCH

    time = Time()   # time object to time simulation
    stderr.write(
        "[start] %s\n\n"
        % time.getInitial())

    proc = Popen(
        ['{ %s; }' % str(' ').join(['setsid', path.join(exec_dir, exec_name)])],
        stdout=DEVNULL, shell=True, env={
            'N': str(N), 'LP': str(lp), 'PHI': str(phi),
            'TORQUE_PARAMETER': str(g),
            'INPUT_FILENAME': str(inputFilename),
            'INPUT_FRAME': str(inputFrame),
            'SEED': str(seed),
            'FILE': path.join(out_dir, out_file),
            'DT': str(dt), 'NITER': str(Niter),
            'NWORK': str(nWork),
            'DUMP': str(dump), 'PERIOD': str(period)})
    proc.wait()

    stderr.write(
        "[end] %s (elapsed: %s)\n\n"
        % (time.getFinal(), time.getElapsed()))

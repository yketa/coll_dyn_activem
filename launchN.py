"""
Module launchN launches simulations with all different parameters and save
logarithmically spaced frames.
"""

from coll_dyn_activem.exponents import float_to_letters
from coll_dyn_activem.init import get_env, Time
from coll_dyn_activem.read import Dat
from coll_dyn_activem.launch0 import v0_AOUP

from numpy.random import randint
from numpy import sqrt, pi

from os import path
from subprocess import Popen, DEVNULL
from sys import stderr

# FUNCTIONS AND CLASSES

def filename(N, epsilon, v0, D, Dr, phi, launch):
    """
    Name of simulation output files.

    Parameters
    ----------
    N : int
        Number of particles in the system.
    epsilon : float
        Coefficient parameter of potential.
    v0 : float
        Self-propulsion velocity.
    D : float
        Translational diffusivity.
    Dr : float
        Rotational diffusivity.
    phi : float
        Packing fraction.
    launch : int
        Launch identifier.

    Returns
    -------
    name : str
        File name.
    """

    return 'N%s_F%s_V%s_T%s_R%s_D%s_E%s.datN' % tuple(map(float_to_letters,
        (N, epsilon, v0, D, Dr, phi, launch)))

# DEFAULT VARIABLES

_N = 10                 # default number of particles in the system
_Dr = 1./40.            # default rotational diffusivity
_v0 = 1                 # default self-propulsion velocity
_phi = 0.65             # default packing fraction

_I = 0  # polydispersity index

_seed = randint(1e7)    # default random seed
_dt = 1e-3              # default time step
_init = 1000            # default initialisation number of iterations
_Niter = 10000          # default number of production iterations
_dtMin = 1              # default minimum lag time
_dtMax = 100            # default maximum lag time
_nMax = 100             # default maxium number of lag times
_intMax = 10            # default maximum number of initial times

_Emin = 1   # default minimum energy at which to stop the minimisation

_launch = 0 # default launch identifier

_nWork = 0  # default number of frames on which to sum the active work before dumping
_dump = 1   # default boolean to indicate to dump positions and orientations to output file
_period = 1 # default period of dumping of positions and orientations in number of frames

_N_cell = 100                                                           # number of particles above which simulations should be launched with a cell list
_exec_dir = path.join(path.dirname(path.realpath(__file__)), 'build')   # default executable directory
_exec_name = ['simulationN%s', 'simulationN%s_cell_list']               # default executable name without and with a cell list
_exec_type = {'ABP': '', 'AOUP': 'OU'}                                  # default suffixes associtated to active particles' types

_out_dir = _exec_dir    # default simulation output directory

# SCRIPT

if __name__ == '__main__':

    # VARIABLE DEFINITIONS

    # INPUT FILE PARAMETERS
    inputFilename = get_env('INPUT_FILENAME', default='', vartype=str)  # input file from which to copy data
    inputFrame = get_env('INPUT_FRAME', default=-1, vartype=int)        # frame to copy as initial frame

    if inputFilename == '':

        # SYSTEM PARAMETERS
        N = get_env('N', default=_N, vartype=int)                       # number of particles in the system
        Dr = get_env('DR', default=_Dr, vartype=float)                  # rotational diffusivity
        epsilon = get_env('EPSILON', default=Dr/3., vartype=float)      # coefficient parameter of potential
        D = get_env('D', default=epsilon, vartype=float)                # translational diffusivity
        v0 = get_env('V0', default=_v0, vartype=float)                  # self-propulsion velocity
        phi = get_env('PHI', default=_phi, vartype=float)               # packing fraction
        I = get_env('I', default=_I, vartype=float)                     # polydispersity index

    else:

        # SYSTEM PARAMETERS
        with Dat(inputFilename, loadWork=False) as dat:                         # data object
            N = get_env('N', default=dat.N, vartype=float)                      # number of particles in the system
            ratioN = N/dat.N                                                    # ratio of number of particles
            Dr = get_env('DR', default=dat.Dr, vartype=float)                   # rotational diffusivity
            epsilon = get_env('EPSILON', default=dat.epsilon, vartype=float)    # coefficient parameter of potential
            D = get_env('D', default=dat.D, vartype=float)                      # translational diffusivity
            v0 = get_env('V0', default=dat.v0, vartype=float)                   # self-propulsion velocity
            phi = get_env('PHI', default=dat.phi, vartype=float)                # packing fraction
            I = get_env('I', default=-1, vartype=float)                         # polydispersity index
            if inputFrame < 0:
                try:
                    inputFrame = dat.frameIndices.max()
                except AttributeError:
                    inputFrame = dat.frames - 1
            del dat
        if ratioN != round(sqrt(ratioN))**2:
            raise ValueError(
                "Ratio of number of particles is not a perfect square.")

    # TYPE
    type = get_env('TYPE', default='ABP', vartype=str)  # type of active particles
    if type == 'AOUP': v0 = 0

    # SIMULATION PARAMETERS
    seed = get_env('SEED', default=_seed, vartype=int)          # random seed
    dt = get_env('DT', default=_dt, vartype=float)              # time step
    init = get_env('INIT', default=_init, vartype=int)          # initialisation number of iterations
    Niter = get_env('NITER', default=_Niter, vartype=int)       # number of production iterations
    dtMin = get_env('LAGMIN', default=_dtMin, vartype=int)      # minimum lag time
    dtMax = get_env('LAGMAX', default=_dtMax, vartype=int)      # maximum lag time
    nMax = get_env('NMAX', default=_nMax, vartype=int)          # maximum number of lag times
    intMax = get_env('INTMAX', default=_intMax, vartype=int)    # maximum number of initial times
    period = get_env('LINEAR', default=None, vartype=int)       # save linearly spaced frames
    if period != None:
        if Niter % period:
            raise ValueError(
                "No integer multiple of %i frames in a total of %i."
                    % (period, Niter))
        else:
            dtMin = 0
            dtMax = period
            nMax = 0
            intMax = Niter//period

    # FIRE ALGORITHM PARAMETERS
    Emin = get_env('EMIN', default=_Emin, vartype=float)            # minimum energy at which to stop the minimisation
    iterMax = get_env('ITERMAX', default=int(100/dt), vartype=int)  # maximum number of iterations of the algorithm
    dtmin = get_env('DTMIN', default=dt*1e-3, vartype=float)        # minimum time step at which to stop the algorithm
    dt0 = get_env('DT0', default=dt*1e-1, vartype=float)            # initial time step of the algorith
    dtmax = get_env('DTMAX', default=dt*1e1, vartype=float)         # maximum time step of the algorithm

    # NAMING PARAMETERS
    launch = get_env('LAUNCH', default=_launch, vartype=float)  # launch identifier

    # EXECUTABLE PARAMETERS
    exec_dir = get_env('EXEC_DIR', default=_exec_dir, vartype=str)      # executable directory
    exec_name = get_env('EXEC_NAME',                                    # executable name
        default=(_exec_name[N >= _N_cell] % _exec_type[type]),
        vartype=str)

    # OUTPUT FILE PARAMETERS
    out_dir = get_env('OUT_DIR', default=_out_dir, vartype=str) # simulation output directory
    out_file = filename(N, epsilon, v0, D, Dr, phi, launch)     # simulation output file name
    if period != None: out_file += '.linear'

    # LAUNCH

    time = Time()   # time object to time simulation
    stderr.write(
        "[start] %s\n\n"
        % time.getInitial())

    proc = Popen(
        ['{ %s; }' % str(' ').join(['setsid', path.join(exec_dir, exec_name)])],
        stdout=DEVNULL, shell=True, env={
            'N': str(N), 'EPSILON': str(epsilon), 'V0': str(v0), 'D': str(D),
                'DR': str(Dr), 'PHI': str(phi), 'I': str(I),
            'INPUT_FILENAME': str(inputFilename),
            'INPUT_FRAME': str(inputFrame),
            'SEED': str(seed),
            'FILE': path.join(out_dir, out_file),
            'DT': str(dt), 'INIT': str(init), 'NITER': str(Niter),
                'LAGMIN': str(dtMin), 'LAGMAX': str(dtMax),
                'NMAX': str(nMax), 'INTMAX': str(intMax),
            'EMIN': str(Emin), 'ITERMAX': str(iterMax), 'DTMIN': str(dtmin),
                'DT0': str(dt0), 'DTMAX': str(dtmax)})
    proc.wait()

    stderr.write(
        "[end] %s (elapsed: %s)\n\n"
        % (time.getFinal(), time.getElapsed()))

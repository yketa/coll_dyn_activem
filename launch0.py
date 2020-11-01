"""
Module launch0 launches simulations with all different parameters.
"""

from coll_dyn_activem.exponents import float_to_letters
from coll_dyn_activem.init import get_env
from coll_dyn_activem.read import Dat

from numpy.random import randint
from numpy import sqrt, pi

from os import path
from subprocess import Popen, DEVNULL

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

    return 'N%s_F%s_V%s_T%s_R%s_D%s_E%s.dat0' % tuple(map(float_to_letters,
        (N, epsilon, v0, D, Dr, phi, launch)))

def v0_AOUP(D, Dr):
    """
    Returns mean self-propulsion vector norm for an OU process with rotational
    diffusivity Dr and translational diffusivity D.

    (see https://yketa.github.io/PhD_Wiki/#Active%20Ornstein-Uhlenbeck%20particles)

    Parameters
    ----------
    D : float
        Translational diffusivity.
    Dr : float
        Rotational diffusivity.

    Returns
    -------
    v0 : float
        Mean self-propulsion vector norm.
    """

    return sqrt(pi*D*Dr/2.)

# DEFAULT VARIABLES

_N = 10                 # default number of particles in the system
_Dr = 1./40.            # default rotational diffusivity
_v0 = 1                 # default self-propulsion velocity
_phi = 0.65             # default packing fraction

_I = 0  # polydispersity index

_seed = randint(1e7)    # default random seed
_dt = 1e-3              # default time step
_Niter = 5e4            # default number of iterations

_Emin = 1   # default minimum energy at which to stop the minimisation

_launch = 0 # default launch identifier

_nWork = 0  # default number of frames on which to sum the active work before dumping
_dump = 1   # default boolean to indicate to dump positions and orientations to output file
_period = 1 # default period of dumping of positions and orientations in number of frames

_N_cell = 100                                                           # number of particles above which simulations should be launched with a cell list
_exec_dir = path.join(path.dirname(path.realpath(__file__)), 'build')   # default executable directory
_exec_name = ['simulation0%s', 'simulation0%s_cell_list']               # default executable name without and with a cell list
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
        if inputFrame < 0:
            try:
                inputFrame = dat.frameIndices.max()
            except AttributeError:
                inputFrame = dat.frames - 1

    else:

        # SYSTEM PARAMETERS
        with Dat(inputFilename, loadWork=False) as dat: # data object
            N = dat.N                                   # number of particles in the system
            Dr = dat.Dr                                 # rotational diffusivity
            epsilon = dat.epsilon                       # coefficient parameter of potential
            D = dat.D                                   # translational diffusivity
            v0 = dat.v0                                 # self-propulsion velocity
            phi = dat.phi                               # packing fraction
            I = -1                                      # polydispersity index
            del dat

    # TYPE
    type = get_env('TYPE', default='ABP', vartype=str)  # type of active particles
    if type == 'AOUP': v0 = 0

    # SIMULATION PARAMETERS
    seed = get_env('SEED', default=_seed, vartype=int)      # random seed
    dt = get_env('DT', default=_dt, vartype=float)          # time step
    Niter = get_env('NITER', default=_Niter, vartype=int)   # number of iterations

    # FIRE ALGORITHM PARAMETERS
    Emin = get_env('EMIN', default=_Emin, vartype=float)            # minimum energy at which to stop the minimisation
    iterMax = get_env('ITERMAX', default=int(100/dt), vartype=int)  # maximum number of iterations of the algorithm
    dtmin = get_env('DTMIN', default=dt*1e-3, vartype=float)        # minimum time step at which to stop the algorithm
    dt0 = get_env('DT0', default=dt*1e-1, vartype=float)            # initial time step of the algorith
    dtmax = get_env('DTMAX', default=dt*1e1, vartype=float)         # maximum time step of the algorithm

    # NAMING PARAMETERS
    launch = get_env('LAUNCH', default=_launch, vartype=float)  # launch identifier

    # OUTPUT PARAMETERS
    nWork = get_env('NWORK', default=_nWork, vartype=int)       # number of frames on which to sum the active work before dumping
    dump = get_env('DUMP', default=_dump, vartype=int)          # boolean to indicate to dump positions and orientations to output file
    period = get_env('PERIOD', default=_period, vartype=int)    # period of dumping of positions and orientations in number of frames

    # EXECUTABLE PARAMETERS
    exec_dir = get_env('EXEC_DIR', default=_exec_dir, vartype=str)      # executable directory
    exec_name = get_env('EXEC_NAME',                                    # executable name
        default=(_exec_name[N >= _N_cell] % _exec_type[type]),
        vartype=str)

    # OUTPUT FILE PARAMETERS
    out_dir = get_env('OUT_DIR', default=_out_dir, vartype=str) # simulation output directory
    out_file = filename(N, epsilon, v0, D, Dr, phi, launch)     # simulation output file name

    # LAUNCH

    proc = Popen(
        ['{ %s; }' % str(' ').join(['setsid', path.join(exec_dir, exec_name)])],
        stdout=DEVNULL, shell=True, env={
            'N': str(N), 'EPSILON': str(epsilon), 'V0': str(v0), 'D': str(D),
                'DR': str(Dr), 'PHI': str(phi), 'I': str(I),
            'INPUT_FILENAME': str(inputFilename),
            'INPUT_FRAME': str(inputFrame),
            'SEED': str(seed),
            'FILE': path.join(out_dir, out_file),
            'DT': str(dt), 'NITER': str(Niter),
            'NWORK': str(nWork),
            'EMIN': str(Emin), 'ITERMAX': str(iterMax), 'DTMIN': str(dtmin),
                'DT0': str(dt0), 'DTMAX': str(dtmax),
            'DUMP': str(dump), 'PERIOD': str(period)})
    proc.wait()

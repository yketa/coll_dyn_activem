"""
Module mixture_pa launches simulations of mixtures of active particles, and
provides functions and classes to analyse them.
"""

from coll_dyn_activem.exponents import float_to_letters
from coll_dyn_activem.init import get_env, Time
from coll_dyn_activem.read import Dat

from numpy.random import randint

from os import path
from distutils.spawn import find_executable
from subprocess import Popen, DEVNULL
from sys import stderr

# FUNCTIONS AND CLASSES

def filename(N1, N2, phi, D1, Dr1, D2, Dr2, launch):
    """
    Name of simulation output files.

    Parameters
    ----------
    N1 : int
        Number of particles of first group.
    N2 : int
        Number of particles of second group.
    phi : float
        Packing fraction.
    D1 : float
        Translational diffusivity of first group.
    Dr1 : float
        Rotational diffusivity of first group.
    D2 : float
        Translational diffusivity of second group.
    Dr2 : float
        Rotational diffusivity of second group.
    launch : int
        Launch identifier.

    Returns
    -------
    name : str
        File name.
    """

    return 'mixture_N%s_D%s_T1%s_R1%s_T2%s_R2%s_E%s.datM' % tuple(map(
        float_to_letters,
        (N1 + N2, phi, D1, Dr1, D2, Dr2, launch)))

# DEFAULT VARIABLES

_N = 1000               # default number of particles per group
_D = 1                  # default translational diffusivity
_Dr = 1                 # default rotational diffusivity
_epsilon = 1            # default interaction potential coefficient
_phi = 0.65             # default packing fraction

_I = 0  # polydispersity index

_seed = randint(1e7)    # default random seed
_dt = 5e-5              # default time step
_init = 1000            # default initialisation number of iterations
_Niter = 10000          # default number of production iterations
_dtMin = 1              # default minimum lag time
_dtMax = 100            # default maximum lag time
_nMax = 100             # default maxium number of lag times
_intMax = 10            # default maximum number of initial times

_launch = 0 # default launch identifier

_exec_dir = path.join(path.dirname(path.realpath(__file__)), 'build')   # default executable directory
_exec_name = 'mixture_pa'                                               # default executable name without and with a cell list

_out_dir = _exec_dir    # default simulation output directory

# SCRIPT

if __name__ == '__main__':

    # VARIABLE DEFINITIONS

    # INPUT FILE PARAMETERS
    inputFilename = get_env('INPUT_FILENAME', default='', vartype=str)  # input file from which to copy data
    inputFrame = get_env('INPUT_FRAME', default=-1, vartype=int)        # frame to copy as initial frame

    if True:
    # if inputFilename == '':

        # SYSTEM PARAMETERS
        N1 = get_env('N1', default=_N, vartype=int)                     # number of particles in the first group
        N2 = get_env('N2', default=N1, vartype=int)                     # number of particles in the first group
        # Dr1 = get_env('DR1', default=_Dr, vartype=float)                # rotational diffusivity of the first group
        # Dr2 = get_env('DR2', default=_Dr, vartype=float)                # rotational diffusivity of the second group
        # D1 = get_env('D1', default=_D, vartype=float)                   # translational diffusivity of the first group
        # D2 = get_env('D2', default=_D, vartype=float)                   # translational diffusivity of the second group
        # epsilon = get_env('EPSILON', default=_epsilon, vartype=float)   # interaction potential coefficient
        # phi1 = get_env('PHI1', default=_phi, vartype=float)             # packing fraction of the first group
        # phi2 = get_env('PHI2', default=phi1, vartype=float)             # packing fraction of the second group
        # I1 = get_env('I1', default=_I, vartype=float)                   # polydispersity index of the first group
        # I2 = get_env('I2', default=I1, vartype=float)                   # polydispersity index of the second group
        ratioL = get_env('RATIOL', default=1, vartype=float)            # aspect ratio of the system box

        #####################################
        # TEST (EQUIVALENT SYSTEM, XU ET AL.)
        Fa = get_env('FA', default=24, vartype=float)       # self-propulsion velocity
        Pe = get_env('PE', default=300, vartype=float)      # Péclet number
        D1 = Fa*Pe/6.
        Dr1 = 3.*Fa/Pe
        D2 = Fa/Pe
        Dr2 = Fa*Pe/2.
        epsilon = 1
        rho1 = get_env('RHO1', default=0.7, vartype=float)  # area fraction
        rho2 = get_env('RHO2', default=rho1, vartype=float)
        phi1 = (2**(1./3.))*rho1
        phi2 = (2**(1./3.))*rho2
        # phi1 = (2**(1./3.))*rho
        # phi2 = phi1
        I1 = get_env('I', default=0, vartype=float)         # polydispersity index
        I2 = get_env('I2', default=I1, vartype=float)
        #####################################

    else:

        raise ValueError

    # SIMULATION PARAMETERS
    seed = get_env('SEED', default=_seed, vartype=int)          # random seed
    dt0 = get_env('DT0', default=5e-5, vartype=float)           # initialisation time step
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

    # NAMING PARAMETERS
    launch = get_env('LAUNCH', default=_launch, vartype=float)  # launch identifier

    # EXECUTABLE PARAMETERS
    exec_dir = get_env('EXEC_DIR', default=_exec_dir, vartype=str)      # executable directory
    exec_name = get_env('EXEC_NAME', default=_exec_name, vartype=str)   # executable name

    # OUTPUT FILE PARAMETERS
    out_dir = get_env('OUT_DIR', default=_out_dir, vartype=str) # simulation output directory
    out_file = filename(N1, N2, phi1, D1, Dr1, D2, Dr2, launch) # simulation output file name
    if period != None: out_file += '.linear'

    # LAUNCH

    time = Time()   # time object to time simulation
    stderr.write(
        "[start] %s\n\n"
        % time.getInitial())

    env = {
        'N1': str(N1), 'N2': str(N2), 'DR1': str(Dr1), 'DR2': str(Dr2),
            'D1': str(D1), 'D2': str(D2), 'EPSILON': str(epsilon),
            'PHI1': str(phi1), 'PHI2': str(phi2), 'I1': str(I1), 'I2': str(I2),
            'RATIOL': str(ratioL),
        'INPUT_FILENAME': str(inputFilename),
        'INPUT_FRAME': str(inputFrame),
        'SEED': str(seed),
        'FILE': path.join(out_dir, out_file),
        'DT0': str(dt0), 'DT': str(dt), 'INIT': str(init), 'NITER': str(Niter),
            'LAGMIN': str(dtMin), 'LAGMAX': str(dtMax),
            'NMAX': str(nMax), 'INTMAX': str(intMax)}
    # proc = Popen(
    #     ['{ %s; }' % str(' ').join(['setsid', path.join(exec_dir, exec_name)])],
    #     stdout=DEVNULL, shell=True, env=env)
    singularity_path = find_executable('singularity')
    assert singularity_path
    proc = Popen(
        ['%s run %s' %
            (singularity_path,
            path.join(
                path.dirname(path.realpath(__file__)),
                'mixture_pa.sif'))],
            # path.join(path.dirname(path.realpath(__file__)), 'mixture_pa.test.sif')],
        stdout=DEVNULL, shell=True,
        env={'PATH': get_env('PATH'),
            **{'SINGULARITYENV_%s' % var: env[var] for var in env}})
    proc.wait()

    stderr.write(
        "[end] %s (elapsed: %s)\n\n"
        % (time.getFinal(), time.getElapsed()))

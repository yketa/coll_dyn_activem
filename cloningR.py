"""
Module cloningR launches cloning simulations of rotors and provides classes to
read output data files from these simulations.

Bias is chosen with environment variable `CLONING_BIAS':
    (0) order parameter,
    (1) squared order parameter.
(see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#Brownian%20rotors%20cloning%20algorithm)
"""

import numpy as np
from numpy import random

from os import path
from shutil import rmtree as rmr
from shutil import move
import sys

from subprocess import Popen, DEVNULL, PIPE

import pickle

from coll_dyn_activem.read import _Read
from coll_dyn_activem.init import get_env, get_env_list, mkdir
from coll_dyn_activem.exponents import float_to_letters
from coll_dyn_activem.maths import mean_sterr

# FUNCTIONS AND CLASSES

class CloningOutput:
    """
    Read and analyse aggregated data from cloning simulations launched with
    coll_dyn_activem.cloningR.
    """

    def __init__(self, filename):
        """
        Get data.

        Parameters
        ----------
        filename : string
            Path to data file.
        """

        self.filename = filename

        with open(self.filename, 'rb') as input:
            (self.exec_path,        # executable path (can help discriminate controlled dynamics method)
            self.tmax,              # dimensionless time simulated
            self.nc,                # number of clones
            self.nRuns,             # number of different runs
            self.initSim,           # number of initial elementary number of iterations to "randomise"
            self.bias,              # cloning bias
            self.sValues,           # biasing parameters
            self.seed,              # master random seed of master random seeds
            self.seeds,             # master random seeds
            self.N,                 # number of rotors in the system
            self.Dr,                # rotational diffusivity
            self._tau,              # elementary number of steps
            self.dt,                # time step
            self.tSCGF,             # array of different measurements of the time scaled CGF per value of the biasing parameter
            self.orderParameter,    # array of different measurements of the order parameter per value of the biasing parameter
            self.orderParameterSq,  # array of different measurements of the squared order parameter per value of the biasing parameter
            self.walltime           # array of different running time per value of the biasing parameter
            ) = pickle.load(input)

        self.order = [self.orderParameter, self.orderParameterSq][self.bias]    # order parameter which bias the trajectories
        self.SCGF = self.tSCGF/self.N                                           # scaled cumulant generating function
        self.tau = self._tau*self.dt                                            # dimensionless elementary time
        self.tinit = self.tau*self.initSim                                      # dimensionless initial simulation time

        self.I = np.empty((self.sValues.size, self.nRuns, 2))   # rate function
        for i in range(self.sValues.size):
            sValue = self.sValues[i]
            for j, SCGF, order in zip(
                range(self.nRuns), self.SCGF[i], self.order[i]):
                self.I[i, j, 0] = order
                self.I[i, j, 1] = -sValue*order - SCGF

    def meanSterr(self, remove=False, max=None):
        """
        Returns array of mean and standard error of measured data.

        Parameters
        ----------
        remove : bool
            Remove inf and -inf as well as nan. (default: False)
            NOTE: A warning will be issued if remove == False and such objects
                  are encountered.
        max : float or None
            Remove data which is strictly above max in absolute value.
            (default: None)
            NOTE: max != None will trigger remove = True.

        Returns
        -------
        SCGF : (self.sValues.size, 3) float Numpy array
            Scaled cumulant generating function.
        orderParameter : (self.sValues.size, 3) float Numpy array.
            Order parameter.
        orderParameterSq : (self.sValues.size, 3) float Numpy array.
            Squared order parameter.

        NOTE: (0) Biasing parameter.
              (1) Mean.
              (2) Standard error.

        I : (self.sValues.size, 4) float Numpy array
            Rate function.

        NOTE: (0) (Squared) order parameter.
              (1) Standard error on (squared) order parameter.
              (2) Rate function.
              (3) Standard error on rate function.
        """

        SCGF = np.empty((self.sValues.size, 3))
        orderParameter = np.empty((self.sValues.size, 3))
        orderParameterSq = np.empty((self.sValues.size, 3))
        I = np.empty((self.sValues.size, 4))
        for i in range(self.sValues.size):
            SCGF[i] = [
                self.sValues[i],
                *mean_sterr(self.SCGF[i], remove=remove, max=max)]
            orderParameter[i] = [
                self.sValues[i],
                *mean_sterr(self.orderParameter[i], remove=remove, max=max)]
            orderParameterSq[i] = [
                self.sValues[i],
                *mean_sterr(self.orderParameterSq[i], remove=remove, max=max)]
            I[i] = [
                *mean_sterr(self.I[i, :, 0], remove=remove, max=max),
                *mean_sterr(self.I[i, :, 1], remove=remove, max=max)]

        return SCGF, orderParameter, orderParameterSq, I

class _CloningOutput(_Read):
    """
    Read data from a single cloning simulation.
    """

    def __init__(self, filename):
        """
        Get data.

        Parameters
        ----------
        filename : string
            Path to data file.
        """

        # FILE
        super().__init__(filename)

        # HEADER INFORMATION
        self.tmax = self._read('d')         # dimensionless time simulated
        self.nc = self._read('i')           # number of clones
        self.sValue = self._read('d')       # biasing parameter
        self.seed = self._read('i')         # master random seed
        self.nRuns = self._read('i')        # number of different runs
        self.cloneMethod = self._read('i')  # cloning method
        self.initSim = self._read('i')      # number of initial elementary number of iterations to "randomise" the systems
        self.N = self._read('i')            # number of rotors in the system
        self.Dr = self._read('d')           # rotational diffusivity
        self.tau = self._read('i')          # elementary number of steps
        self.dt = self._read('d')           # time step
        self.bias = self._read('i')         # cloning bias

        # FILE PARTS LENGTHS
        self.headerLength = self.file.tell()    # length of header in bytes
        self.runLength = 4*self._bpe('d')       # length the data of a run takes

        # FILE CORRUPTION CHECK
        if self.fileSize != self.headerLength + self.nRuns*self.runLength:
            raise ValueError("Invalid data file size.")

        # MEASUREMENTS
        self.tSCGF = np.empty((self.nRuns,))            # time scaled cumulant generating function
        self.orderParameter = np.empty((self.nRuns,))   # order parameter
        self.orderParameterSq = np.empty((self.nRuns,)) # squared order parameter
        self.walltime = np.empty((self.nRuns,))         # time taken for each run
        for i in range(self.nRuns):
            self.tSCGF[i] = self._read('d')
            self.orderParameter[i] = self._read('d')
            self.orderParameterSq[i] = self._read('d')
            self.walltime[i] = self._read('d')

def filename(N, Dr, nc, bias, launch):
    """
    Name of simulation output directory.

    Parameters
    ----------
    N : int
        Number of rotors in the system.
    Dr : float
        Rotational diffusivity.
    nc : int
        Number of clones.
    bias : int
        Cloning bias.
    launch : int
        Launch identifier.

    Returns
    -------
    name : str
        File name.
    """

    return 'N%s_R%s_NC%s_B%s_E%s' % tuple(map(float_to_letters,
        (N, Dr, nc, bias, launch)))

# DEFAULT PARAMETERS

_tmax = 1                   # default dimensionless time to simulate
_nc = 10                    # default number of clones
_seed = random.randint(1e7) # default master random seed
_nRuns = 1                  # default number of different runs
_initSim = 1                # default number of initial elementary number of iterations to "randomise" the systems
_bias = 0                   # default cloning bias

_sMin = -0.1    # default minimum value of the biasing parameter
_sMax = 0.1     # default maximum value of the biasing parameter
_sNum = 10      # default number of values of the biasing parameter

_threads = -1           # [openMP] default number of threads

_N = 100    # default number of rotors in the system
_Dr = 1./2. # default rotational diffusivity

_tau = 100  # default elementary number of steps
_dt = 0.001 # default time step

_launch = 0 # default launch identifier

_exec_dir = path.join(path.dirname(path.realpath(__file__)), 'build')   # default executable directory
_exec_name = {                                                          # default executable name
    0: ('cloningR_B0', 'cloningR_B0'),                                  # cloning bias `0' without and with control
    1: ('cloningR_B1', 'cloningR_B1_C')}                                # cloning bias `1' without and with control

_slurm_path = path.join(path.dirname(path.realpath(__file__)), 'slurm.sh')  # Slurm submitting script

_out_dir = _exec_dir    # default simulation output directory

# SCRIPT

if __name__ == '__main__':

    # VARIABLE DEFINITIONS

    # CLONING PARAMETERS
    tmax = get_env('TMAX', default=_tmax, vartype=float)        # dimensionless time to simulate
    nc = get_env('NC', default=_nc, vartype=int)                # number of clones
    nRuns = get_env('NRUNS', default=_nRuns, vartype=int)       # number of different runs
    initSim = get_env('INITSIM', default=_initSim, vartype=int) # number of initial elementary number of iterations to "randomise" the systems
    bias = get_env('CLONING_BIAS', default=_bias, vartype=int)  # cloning bias

    # BIASING PARAMETERS
    sMin = get_env('SMIN', default=_sMin, vartype=float)    # minimum value of the biasing parameter
    sMax = get_env('SMAX', default=_sMax, vartype=float)    # maximum value of the biasing parameter
    sNum = get_env('SNUM', default=_sNum, vartype=int)      # number of values of the biasing parameter
    sValues = np.linspace(sMin, sMax, sNum, endpoint=True)  # array of values of the biasing parameter

    # RANDOM SEEDS
    seed = get_env('SEED', default=_seed, vartype=int)  # master random seed of master random seeds
    random.seed(seed)                                   # set seed
    seeds = random.randint(1e7, size=(sNum,))           # master random seeds

    # OPENMP PARAMETERS
    threads = get_env('THREADS', default=_threads, vartype=int) # number of threads

    # SLURM PARAMETERS
    slurm = get_env('SLURM', default=False, vartype=bool)       # use Slurm job scheduler (see coll_dyn_activem/slurm.sh)
    slurm_partition = get_env('SLURM_PARTITION', vartype=str)   # partition for the ressource allocation
    slurm_ntasks = get_env('SLURM_NTASKS', vartype=int)         # number of MPI ranks running per node
    slurm_time = get_env('SLURM_TIME', vartype=str)             # required time
    slurm_chain = get_env_list('SLURM_CHAIN', vartype=int)      # execute after these jobs ID have completed (order has to be the same as sValues)

    # PHYSICAL PARAMETERS
    N = get_env('N', default=_N, vartype=int)       # number of rotors in the system
    Dr = get_env('DR', default=_Dr, vartype=float)  # rotational diffusivity

    # SIMULATION PARAMETERS
    tau = get_env('TAU', default=_tau, vartype=int) # elementary number of steps
    dt = get_env('DT', default=_dt, vartype=float)  # time step

    # EXECUTABLE PARAMETERS
    exec_dir = get_env('EXEC_DIR', default=_exec_dir, vartype=str)  # executable directory
    exec_name = get_env('EXEC_NAME',                                # executable name
        default=_exec_name[bias][
            get_env('CONTROLLED_DYNAMICS', default=False, vartype=bool)],
        vartype=str)
    exec_path = path.join(exec_dir, exec_name)                      # executable path

    # OUTPUT FILES PARAMETERS
    launch = get_env('LAUNCH', default=_launch, vartype=float)      # launch identifier
    out_dir = get_env('OUT_DIR', default=_out_dir, vartype=str)     # output directory
    sim_name = filename(N, Dr, nc, bias, launch)                    # simulation output name
    sim_dir = path.join(out_dir, sim_name)                          # simulation output directory name
    mkdir(sim_dir, replace=True)
    tmp_dir = path.join(sim_dir, 'tmp')                             # temporary files directory
    mkdir(tmp_dir, replace=True)
    tmp_template = '%010d.cloning.out'                              # template of temporary files
    out_file = path.join(sim_dir, sim_name + '.cloR')               # simulation output file name

    # LAUNCH

    env = lambda i: {   # environment variables for cloning executables as function of sValues index
        'TMAX': str(tmax), 'NC': str(nc), 'SVALUE': str(sValues[i]),
            'SEED': str(seeds[i]), 'NRUNS': str(nRuns),
            'INITSIM': str(initSim),
        'THREADS': str(threads),
        'N': str(N), 'DR': str(Dr),
        'TAU': str(tau), 'DT': str(dt),
        'FILE': path.join(tmp_dir,
            tmp_template % i)}

    if slurm:   # using Slurm job scheduler

        slurm_launch = ['bash', _slurm_path, '-w']  # commands to submit Slurm job
        if slurm_partition != None: slurm_launch += ['-p', slurm_partition]
        if slurm_ntasks != None: slurm_launch += ['-r', str(slurm_ntasks)]
        if slurm_time != None: slurm_launch += ['-t', slurm_time]

        # LAUNCH
        procs, jobsID = [], []
        for i in range(sNum):

            procs += [
                Popen(
                    ['%s \"{ %s %s; }\"' %
                        (str(' ').join(slurm_launch                 # Slurm submitting script
                            + ['-j', '\'' +  exec_path.split('/')[-1]
                                + ' %04i %s\'' % (i, env(i)['SVALUE'])]
                            + ([] if slurm_chain == []
                                else ['-c', str(slurm_chain[i])])),
                        str(' ').join(['%s=%s' % (key, env(i)[key]) # environment variables
                            for key in env(i)]),
                        exec_path)],                                # cloning executable
                    stdout=PIPE, shell=True)]

            getJobID = Popen(                               # get submitted job ID
                ['head', '-n1'],
                stdin=procs[-1].stdout, stdout=PIPE)
            getJobID.wait()
            jobID = getJobID.communicate()[0].decode().split()
            if jobID[:-1] == ['Submitted', 'batch', 'job']: # job has been succesfully submitted to Slurm
                jobsID += [jobID[-1]]
            else:                                           # failed to submit job
                raise ValueError("Job ID not returned.")

        sys.stdout.write(':'.join(jobsID) + '\n')   # print jobs ID to stdout with syntax compatible with coll_dyn_activem.init.get_env_list
        sys.stdout.flush()

    else:   # not using Slurm job scheduler

        # LAUNCH
        procs = [
            Popen(['{ %s; }' % exec_path],
                stdout=DEVNULL, shell=True, env=env(i))
            for i in range(sNum)]

    for proc in procs: proc.wait()  # wait for them to finish

    # CLONING OUTPUT FILE

    # LOAD TEMPORARY FILES
    tmp_out = []
    for i in range(sNum):
        tmp_out += [_CloningOutput(
            path.join(tmp_dir, tmp_template % i))]

    # ARRAYS OF DATA
    tSCGF = np.array(
        [tmp_out[i].tSCGF for i in range(sNum)])
    orderParameter = np.array(
        [tmp_out[i].orderParameter for i in range(sNum)])
    orderParameterSq = np.array(
        [tmp_out[i].orderParameterSq for i in range(sNum)])
    walltime = np.array(
        [tmp_out[i].walltime for i in range(sNum)])

    # OUT
    with open(out_file, 'wb') as output:
        pickle.dump([
            exec_path,
            tmax, nc, nRuns, initSim, bias, sValues,
            seed, seeds,
            N, Dr,
            tau, dt,
            tSCGF, orderParameter, orderParameterSq, walltime],
            output)

    # CLEAN
    if get_env('CLEAN', default=True, vartype=bool):
        move(out_file, path.join(out_dir, sim_name + '.cloR'))  # move output file to output directory
        rmr(sim_dir, ignore_errors=True)                        # delete simulation directory

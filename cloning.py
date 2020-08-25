"""
Module cloning launches cloning simulations of ABPs biased with respect to
either the polarisation or the active work and provides classes to read output
data files from these simulations.

Controlled dynamics is chosen with environment variable `CONTROLLED_DYNAMICS':
    [POLARISATION]
    (0) Unmodified dynamics,
    (1) Modified rotational EOM, choosing free parameter as function of the
        order parameter norm.
    [ACTIVE WORK]
    (0) Unmodified dynamics,
    (1) Modified translational EOM,
    (2) Modified translational and rotational EOM, choosing free parameter as
        function of the order parameter norm,
    (3) Modified translational and rotational EOM, choosing free parameter as
        to minimise the SCGF.
(see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#ABP%20cloning%20algorithm%20%E2%80%94%20polarisation
and https://yketa.github.io/DAMTP_MSC_2019_Wiki/#ABP%20cloning%20algorithm)
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
    coll_dyn_activem.cloning.
    """

    def __init__(self, filename, bias_polarisation=False):
        """
        Get data.

        Parameters
        ----------
        filename : string
            Path to data file.
        bias_polarisation : bool
            Cloning data corresponds to simulation biased with respect to the
            polarisation. (default: False)
        """

        self.filename = filename
        self.bias_polarisation = bias_polarisation

        with open(self.filename, 'rb') as input:
            (self.exec_path,        # executable path (can help discriminate controlled dynamics method)
            self.tmax,              # dimensionless time simulated
            self.nc,                # number of clones
            self.nRuns,             # number of different runs
            self.initSim,           # number of initial elementary number of iterations to "randomise"
            self.sValues,           # biasing parameters
            self.seed,              # master random seed of master random seeds
            self.seeds,             # master random seeds
            self.N,                 # number of particles in the system
            self.lp,                # dimensionless persistence length
            self.phi,               # packing fraction
            self._tau,              # elementary number of steps
            self.dt,                # time step
            self.tSCGF,             # array of different measurements of the time scaled CGF per value of the biasing parameter
            self.activeWork,        # array of different measurements of the active work per value of the biasing parameter
            self.activeWorkForce,   # array of different measurements of the force part of the active work per value of the biasing parameter
            self.activeWorkOri,     # array of different measurements of the orientation part of the active work per value of the biasing parameter
            self.orderParameter,    # array of different measurements of the order parameter per value of the biasing parameter
            self.walltime           # array of different running time per value of the biasing parameter
            ) = pickle.load(input)

        self.SCGF = self.tSCGF/self.N       # scaled cumulant generating function
        self.tau = self._tau*self.dt        # dimensionless elementary time
        self.tinit = self.tau*self.initSim  # dimensionless initial simulation time

        self.order = [self.activeWork, self.orderParameter][
            self.bias_polarisation]
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
        activeWork : (self.sValues.size, 3) float Numpy array
            Normalised rate of active work.
        activeWorkForce : (self.sValues.size, 3) float Numpy array
            Force part of the normalised rate of active work.
        activeWorkOri : (self.sValues.size, 3) float Numpy array
            Orientation part of the normalised rate of active work.
        orderParameter : (self.sValues.size, 3) float Numpy array
            Order parameter.

        NOTE: (0) Biasing parameter.
              (1) Mean.
              (2) Standard error.

        I : (self.sValues.size, 4) float Numpy array
            Rate function.

        NOTE: (0) Active work.
              (1) Standard error on active work.
              (2) Rate function.
              (3) Standard error on rate function.
        """

        if max != None: remove = True

        SCGF = np.empty((self.sValues.size, 3))
        activeWork = np.empty((self.sValues.size, 3))
        activeWorkForce = np.empty((self.sValues.size, 3))
        activeWorkOri = np.empty((self.sValues.size, 3))
        orderParameter = np.empty((self.sValues.size, 3))
        I = np.empty((self.sValues.size, 4))
        for i in range(self.sValues.size):
            SCGF[i] = [
                self.sValues[i],
                *mean_sterr(self.SCGF[i], remove=remove, max=max)]
            activeWork[i] = [
                self.sValues[i],
                *mean_sterr(self.activeWork[i], remove=remove, max=max)]
            activeWorkForce[i] = [
                self.sValues[i],
                *mean_sterr(self.activeWorkForce[i], remove=remove, max=max)]
            activeWorkOri[i] = [
                self.sValues[i],
                *mean_sterr(self.activeWorkOri[i], remove=remove, max=max)]
            orderParameter[i] = [
                self.sValues[i],
                *mean_sterr(self.orderParameter[i], remove=remove, max=max)]
            I[i] = [
                *mean_sterr(self.I[i, :, 0], remove=remove, max=max),
                *mean_sterr(self.I[i, :, 1], remove=remove, max=max)]

        return (SCGF,
            activeWork, activeWorkForce, activeWorkOri, orderParameter,
            I)

class _CloningOutput(_Read):
    """
    Read data from a single cloning simulation.
    """

    def __init__(self, filename, repair=False):
        """
        Get data.

        Parameters
        ----------
        filename : string
            Path to data file.
        repair : bool
            Extract data even when the number of runs is lesser than expected.
            (default: False)
            NOTE: Replaces missing data with None.
        """

        # FILE
        super().__init__(filename)

        # REPAIR
        self.repair = repair

        # HEADER INFORMATION
        self.tmax = self._read('d')         # dimensionless time simulated
        self.nc = self._read('i')           # number of clones
        self.sValue = self._read('d')       # biasing parameter
        self.seed = self._read('i')         # master random seed
        self.nRuns = self._read('i')        # number of different runs
        self.cloneMethod = self._read('i')  # cloning method
        self.initSim = self._read('i')      # number of initial elementary number of iterations to "randomise" the systems
        self.N = self._read('i')            # number of particles in the system
        self.lp = self._read('d')           # dimensionless persistence length
        self.phi = self._read('d')          # packing fraction
        self.tau = self._read('i')          # elementary number of steps
        self.dt = self._read('d')           # time step

        # FILE PARTS LENGTHS
        self.headerLength = self.file.tell()    # length of header in bytes
        self.runLength = 6*self._bpe('d')       # length the data of a run takes

        # FILE CORRUPTION CHECK
        if (
            (self.fileSize != self.headerLength + self.nRuns*self.runLength
                and not(self.repair))                                       # not the correct number of runs in non-repair mode
            or (self.fileSize - self.headerLength)%self.runLength != 0):    # not an integer number of runs
            raise ValueError("Invalid data file size.")

        # MEASUREMENTS
        self.tSCGF = np.empty((self.nRuns,))            # time scaled cumulant generating function
        self.activeWork = np.empty((self.nRuns,))       # normalised rate of active work
        self.activeWorkForce = np.empty((self.nRuns,))  # force part of the normalised rate of active work
        self.activeWorkOri = np.empty((self.nRuns,))    # orientation part of the normalised rate of active work
        self.orderParameter = np.empty((self.nRuns,))   # order parameter
        self.walltime = np.empty((self.nRuns,))         # time taken for each run
        for i in range(self.nRuns):
            if (self.repair and
                i >= (self.fileSize - self.headerLength)//self.runLength == 0):
                self.tSCGF[i] = None
                self.activeWork[i] = None
                self.activeWorkForce[i] = None
                self.activeWorkOri[i] = None
                self.orderParameter[i] = None
                self.walltime[i] = None
            else:
                self.tSCGF[i] = self._read('d')
                self.activeWork[i] = self._read('d')
                self.activeWorkForce[i] = self._read('d')
                self.activeWorkOri[i] = self._read('d')
                self.orderParameter[i] = self._read('d')
                self.walltime[i] = self._read('d')

class TorqueDump:
    """
    Read torque parameter dump file.

    NOTE: Cloning simulations have to be run with CONTROLLED_DYNAMICS set to 2
          or 3 and witg TORQUE_DUMP defined. (see
          coll_dyn_activem/cloningserial.hpp)
    """

    def __init__(self, filename):
        """
        Get data.

        Parameters
        ----------
        filename : string
            Path to data file. (default: torque.dump)
        """

        self.filename = filename

        with open(self.filename, 'rb') as input:
            (self.exec_path,        # executable path (can help discriminate controlled dynamics method)
            self.tmax,              # dimensionless time simulated
            self.nc,                # number of clones
            self.nRuns,             # number of different runs
            self.initSim,           # number of initial elementary number of iterations to "randomise"
            self.sValues,           # biasing parameters
            self.seed,              # master random seed of master random seeds
            self.seeds,             # master random seeds
            self.N,                 # number of particles in the system
            self.lp,                # dimensionless persistence length
            self.phi,               # packing fraction
            self._tau,              # elementary number of steps
            self.dt,                # time step
            self.torqueParameter    # array of torque parameters along the simulation
            ) = pickle.load(input)

        self.tau = self._tau*self.dt        # dimensionless elementary time
        self.tinit = self.tau*self.initSim  # dimensionless initial simulation time

class _TorqueDump(_Read):
    """
    Read torque parameter from a single cloning simulation.

    NOTE: Cloning simulations have to be run with CONTROLLED_DYNAMICS set to 2
          or 3 and witg TORQUE_DUMP defined. (see
          coll_dyn_activem/cloningserial.hpp)
    """

    def __init__(self, filename='torque.dump'):
        """
        Get data.

        Parameters
        ----------
        filename : string
            Path to data file. (default: torque.dump)
        """

        # FILE
        super().__init__(filename)

        # FILE CORRUPTION CHECK
        if self.fileSize % self._bpe('d') != 0:
            raise ValueError("Invalid data file size.")

        # GET TORQUE PARAMETERS
        self.torqueParameter = []
        for i in range(self.fileSize//self._bpe('d')):
            self.torqueParameter += [self._read('d')]
        self.torqueParameter = np.array(self.torqueParameter)

def filename(N, phi, lp, nc, launch):
    """
    Name of simulation output directory.

    Parameters
    ----------
    N : int
        Number of particles in the system.
    phi : float
        Packing fraction.
    lp : float
        Dimensionless persistence length.
    nc : int
        Number of clones.
    launch : int
        Launch identifier.

    Returns
    -------
    name : str
        File name.
    """

    return 'N%s_D%s_L%s_NC%s_E%s' % tuple(map(float_to_letters,
        (N, phi, lp, nc, launch)))

# DEFAULT PARAMETERS

_tmax = 1                   # default dimensionless time to simulate
_nc = 10                    # default number of clones
_seed = random.randint(1e7) # default master random seed
_nRuns = 1                  # default number of different runs
_initSim = 1                # default number of initial elementary number of iterations to "randomise" the systems

_sMin = -0.1    # default minimum value of the biasing parameter
_sMax = 0.1     # default maximum value of the biasing parameter
_sNum = 10      # default number of values of the biasing parameter

_threads = -1           # [openMP] default number of threads

_N = 100    # default number of particles in the system
_lp = 5     # default dimension persistence length
_phi = 0.65 # default packing fraction

_tau = 100  # default elementary number of steps
_dt = 0.001 # default time step

_launch = 0 # default launch identifier

_N_cell = 100                                                               # number of particles above which simulations should be launched with a cell list
_exec_dir = path.join(path.dirname(path.realpath(__file__)), 'build')       # default executable directory
_exec_name = {                                                              # default executable name
    False: {0: 'cloning', **{i: 'cloning_C%i' % i for i in range(1, 4)}},   # biasing with respect to active work
    True: {0: 'cloningP', 1: 'cloningP_C'}}                                 # biasing with respect to polarisation

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
    N = get_env('N', default=_N, vartype=int)           # number of particles in the system
    lp = get_env('LP', default=_lp, vartype=float)      # dimensionless persistence length
    phi = get_env('PHI', default=_phi, vartype=float)   # packing fraction

    # SIMULATION PARAMETERS
    tau = get_env('TAU', default=_tau, vartype=int) # elementary number of steps
    dt = get_env('DT', default=_dt, vartype=float)  # time step

    # SAVE/LOAD CONFIGURATIONS
    # these files have to be given with a format specifier for the index in the list of biasing parameters
    load_file = get_env('LOAD_FILE', default='', vartype=str)   # load cloning configurations from input file (overrides physical and simulation parameters)
    save_file = get_env('SAVE_FILE', default='', vartype=str)   # save cloning configurations to output file

    # EXECUTABLE PARAMETERS
    exec_dir = get_env('EXEC_DIR', default=_exec_dir, vartype=str)      # executable directory
    exec_name = get_env('EXEC_NAME',                                    # executable name
        default=_exec_name[
            get_env('BIAS_POLARISATION', default=False, vartype=bool)][
            get_env('CONTROLLED_DYNAMICS', default=0, vartype=int)]
            + ('_cell_list' if N >= _N_cell else ''),
        vartype=str)
    exec_path = path.join(exec_dir, exec_name)                          # executable path

    # OUTPUT FILES PARAMETERS
    launch = get_env('LAUNCH', default=_launch, vartype=float)              # launch identifier
    out_dir = get_env('OUT_DIR', default=_out_dir, vartype=str)             # output directory
    sim_name = filename(N, phi, lp, nc, launch)                             # simulation output name
    sim_dir = path.join(out_dir, sim_name)                                  # simulation output directory name
    mkdir(sim_dir, replace=True)
    tmp_dir = path.join(sim_dir, 'tmp')                                     # temporary files directory
    mkdir(tmp_dir, replace=True)
    tmp_template = '%04d.cloning.out'                                      # template of temporary files
    out_file = path.join(sim_dir, sim_name + '.clo')                        # simulation output file name
    torque_tmp_template = tmp_template.replace('cloning', 'torque')         # template of temporary torque parameter dump files
    torque_file = out_file.replace('.clo', '.torque.dump')                  # torque parameter dump file name
    save_traj = get_env('SAVE_TRAJECTORIES', default=False, vartype=bool)   # save individual clone trajectories
    if save_traj:
        for i in range(sNum):
            mkdir(path.join(tmp_dir, '%04d.traj' % i))

    # LAUNCH

    env = lambda i: {   # environment variables for cloning executables as function of sValues index
        'TMAX': str(tmax), 'NC': str(nc), 'SVALUE': str(sValues[i]),
            'SEED': str(seeds[i]), 'NRUNS': str(nRuns),
            'INITSIM': str(initSim),
        'THREADS': str(threads),
        'N': str(N), 'LP': str(lp), 'PHI': str(phi),
        'TAU': str(tau), 'DT': str(dt),
        'LOAD_FILE': load_file.format(i), 'SAVE_FILE': save_file.format(i),
        'FILE': path.join(tmp_dir, tmp_template % i),
        'TORQUE_DUMP_FILE': path.join(tmp_dir, torque_tmp_template % i),
        'CLONES_DIRECTORY': (path.join(tmp_dir, '%04d.traj' % i)
            if save_traj else '')}


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
    activeWork = np.array(
        [tmp_out[i].activeWork for i in range(sNum)])
    activeWorkForce = np.array(
        [tmp_out[i].activeWorkForce for i in range(sNum)])
    activeWorkOri = np.array(
        [tmp_out[i].activeWorkOri for i in range(sNum)])
    orderParameter = np.array(
        [tmp_out[i].orderParameter for i in range(sNum)])
    walltime = np.array(
        [tmp_out[i].walltime for i in range(sNum)])

    # OUT
    with open(out_file, 'wb') as output:
        pickle.dump([
            exec_path,
            tmax, nc, nRuns, initSim, sValues,
            seed, seeds,
            N, lp, phi,
            tau, dt,
            tSCGF, activeWork, activeWorkForce, activeWorkOri, orderParameter,
                walltime],
            output)

    # TRAJECTORY FILES

    if save_traj:
        for i in range(sNum):
            for j in range(nRuns):
                for k in range(nc):
                    traj_file = '%04d.%06d.dat' % (j, k)
                    move(path.join(tmp_dir, '%04d.traj' % i, traj_file),
                        path.join(sim_dir, (sim_name + '_S%s.' + traj_file)
                            % float_to_letters(sValues[i])))

    # TORQUE PARAMETER OUTPUT FILE

    # LOAD TEMPORARY FILES
    torque_tmp_out = []
    for i in range(sNum):
        try:
            torque_tmp_out += [_TorqueDump(
                path.join(tmp_dir, torque_tmp_template % i))]
        except (FileNotFoundError, ValueError):
            del torque_tmp_out
            break

    try: torque_tmp_out
    except NameError: torque_tmp_out = None
    if torque_tmp_out != None:

        # ARRAYS OF DATA
        torqueParameter = np.array(
            [torque_tmp_out[i].torqueParameter for i in range(sNum)])

        # OUT
        with open(torque_file, 'wb') as output:
            pickle.dump([
                exec_path,
                tmax, nc, nRuns, initSim, sValues,
                seed, seeds,
                N, lp, phi,
                tau, dt,
                torqueParameter],
                output)

    # CLEAN
    if get_env('CLEAN', default=True, vartype=bool):
        move(out_file, path.join(out_dir, sim_name + '.clo'))                   # move output file to output directory
        if torque_tmp_out:
            move(torque_file, path.join(out_dir, sim_name + '.torque.dump'))    # move output file to output directory
        if save_traj:
            for i in range(sNum):
                for j in range(nRuns):
                    for k in range(nc):
                        traj_file = (sim_name + '_S%s.%04d.%06d.dat' %
                            (float_to_letters(sValues[i]), j, k))
                        move(path.join(sim_dir, traj_file),
                            path.join(out_dir, traj_file))
        rmr(sim_dir, ignore_errors=True)                                        # delete simulation directory

"""
Module launch launches simulations with custom relations between parameters at
different values of the torque parameter.
"""

import numpy as np
from numpy import random, gcd

from os import path
from shutil import rmtree as rmr
from shutil import move
import sys

from subprocess import Popen, DEVNULL, PIPE

import pickle

from coll_dyn_activem.read import _Dat as Dat
from coll_dyn_activem.init import get_env, get_env_list, mkdir
from coll_dyn_activem.exponents import float_to_letters
from coll_dyn_activem.maths import mean_sterr

# FUNCTIONS AND CLASSES

class DatG:
    """
    Read and analyse aggregated data from simulations at different values of the
    torque parameter.
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
            (self.exec_path,        # executable path
            self.initSim,           # number of initial number of iterations to "randomise" the systems
            self.Niter,             # number of iterations
            self.dt,                # time step
            self.nRuns,             # number of different runs
            self.gValues,           # array of values of the torque parameter
            self.seed,              # master random seed of master random seeds
            self.seeds,             # master random seeds
            self.N,                 # number of particles in the system
            self.lp,                # dimensionless persistence length
            self.phi,               # packing fraction
            self.activeWork,        # array of different measurements of the active work per value of the torque parameter
            self.activeWorkForce,   # array of different measurements of the force part of the active work per value of the torque parameter
            self.activeWorkOri,     # array of different measurements of the orientation part of the active work per value of the torque parameter
            self.orderParameter,    # array of different measurements of the order parameter per value of the torque parameter
            self.torqueIntegral1,   # array of different measurements of the first torque integral per value of the torque parameter
            self.torqueIntegral2    # array of different measurements of the second torque integral per value of the torque parameter
            ) = pickle.load(input)

        self.tinit = self.dt*self.initSim           # dimensionless initial simulation time
        self.tmax = self.dt*self.Niter - self.tinit # dimensionless total simulation time

        self.Lambda = np.empty((self.gValues.size, self.nRuns, 2))  # bound to the rate function
        for i in range(self.gValues.size):
            gValue = self.gValues[i]
            for j, activeWork, torqueIntegral1, torqueIntegral2 in zip(
                range(self.nRuns), self.activeWork[i],
                self.torqueIntegral1[i], self.torqueIntegral2[i]):
                self.Lambda[i, j, 0] = activeWork
                self.Lambda[i, j, 1] = gValue*(
                    1./self.N - torqueIntegral1
                    - gValue*self.lp*torqueIntegral2)

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
        activeWork : (self.gValues.size, 3) float Numpy array
            Normalised rate of active work.
        activeWorkForce : (self.gValues.size, 3) float Numpy array
            Force part of the normalised rate of active work.
        activeWorkOri : (self.gValues.size, 3) float Numpy array
            Orientation part of the normalised rate of active work.
        orderParameter : (self.gValues.size, 3) float Numpy array
            Order parameter.
        torqueIntegral1 : (self.gValues.size, 3) float Numpy array
            First torque integral.
        torqueIntegral2 : (self.gValues.size, 3) float Numpy array
            Second torque integral.

        NOTE: (0) Torque parameter.
              (1) Mean.
              (2) Standard error.

        Lambda : (self.gValues.size, 4) float Numpy array
            Bound to the rate function.

        NOTE: (0) Active work.
              (1) Standard error on active work.
              (2) Bound to the rate function.
              (3) Standard error on the boudn to the function.
        """

        activeWork = np.empty((self.gValues.size, 3))
        activeWorkForce = np.empty((self.gValues.size, 3))
        activeWorkOri = np.empty((self.gValues.size, 3))
        orderParameter = np.empty((self.gValues.size, 3))
        torqueIntegral1 = np.empty((self.gValues.size, 3))
        torqueIntegral2 = np.empty((self.gValues.size, 3))
        Lambda = np.empty((self.gValues.size, 4))
        for i in range(self.gValues.size):
            activeWork[i] = [
                self.gValues[i],
                *mean_sterr(self.activeWork[i], remove=remove, max=max)]
            activeWorkForce[i] = [
                self.gValues[i],
                *mean_sterr(self.activeWorkForce[i], remove=remove, max=max)]
            activeWorkOri[i] = [
                self.gValues[i],
                *mean_sterr(self.activeWorkOri[i], remove=remove, max=max)]
            orderParameter[i] = [
                self.gValues[i],
                *mean_sterr(self.orderParameter[i], remove=remove, max=max)]
            torqueIntegral1[i] = [
                self.gValues[i],
                *mean_sterr(self.torqueIntegral1[i], remove=remove, max=max)]
            torqueIntegral2[i] = [
                self.gValues[i],
                *mean_sterr(self.torqueIntegral2[i], remove=remove, max=max)]
            Lambda[i] = [
                *mean_sterr(self.Lambda[i, :, 0], remove=remove, max=max),
                *mean_sterr(self.Lambda[i, :, 1], remove=remove, max=max)]

        return (activeWork, activeWorkForce, activeWorkOri, orderParameter,
            torqueIntegral1, torqueIntegral2, Lambda)

def filename(N, phi, lp, launch):
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
    launch : int
        Launch identifier.

    Returns
    -------
    name : str
        File name.
    """

    return 'N%s_D%s_L%s_E%s' % tuple(map(float_to_letters,
        (N, phi, lp, launch)))

# DEFAULT PARAMETERS

_seed = random.randint(1e7) # default master random seed
_nRuns = 1                  # default number of different runs
_initSim = 1                # default number of initial elementary number of iterations to "randomise" the systems

_gMin = -1  # default minimum value of the torque parameter
_gMax = 0   # default maximum value of the torque parameter
_gNum = 11  # default number of values of the torque parameter

_N = 100    # default number of particles in the system
_lp = 5     # default dimension persistence length
_phi = 0.65 # default packing fraction

_Niter = 5e4    # default number of iterations
_dt = 0.001     # default time step

_launch = 0 # default launch identifier

_N_cell = 100                                                               # number of particles above which simulations should be launched with a cell list
_exec_dir = path.join(path.dirname(path.realpath(__file__)), 'build')       # default executable directory
_exec_name = ['simulation', 'simulation_cell_list']                         # default executable name without and with a cell list

_slurm_path = path.join(path.dirname(path.realpath(__file__)), 'slurm.sh')  # Slurm submitting script

_out_dir = _exec_dir    # default simulation output directory

# SCRIPT

if __name__ == '__main__':

    # VARIABLE DEFINITIONS

    # CLONING PARAMETERS
    nRuns = get_env('NRUNS', default=_nRuns, vartype=int)       # number of different runs
    initSim = get_env('INITSIM', default=_initSim, vartype=int) # number of initial number of iterations to "randomise" the systems

    # BIASING PARAMETERS
    gMin = get_env('GMIN', default=_gMin, vartype=float)    # minimum value of the torque parameter
    gMax = get_env('GMAX', default=_gMax, vartype=float)    # maximum value of the torque parameter
    gNum = get_env('GNUM', default=_gNum, vartype=int)      # number of values of the torque parameter
    gValues = np.linspace(gMin, gMax, gNum, endpoint=True)  # array of values of the torque parameter

    # RANDOM SEEDS
    seed = get_env('SEED', default=_seed, vartype=int)  # master random seed of master random seeds
    random.seed(seed)                                   # set seed
    seeds = random.randint(1e7, size=(gNum, nRuns))     # master random seeds

    # SLURM PARAMETERS
    slurm = get_env('SLURM', default=False, vartype=bool)       # use Slurm job scheduler (see coll_dyn_activem/slurm.sh)
    slurm_partition = get_env('SLURM_PARTITION', vartype=str)   # partition for the ressource allocation
    slurm_time = get_env('SLURM_TIME', vartype=str)             # required time
    slurm_chain = get_env_list('SLURM_CHAIN', vartype=int)      # execute after these jobs ID have completed (order has to be the same as gValues x nRuns)

    # PHYSICAL PARAMETERS
    N = get_env('N', default=_N, vartype=int)           # number of particles in the system
    lp = get_env('LP', default=_lp, vartype=float)      # dimensionless persistence length
    phi = get_env('PHI', default=_phi, vartype=float)   # packing fraction

    # SIMULATION PARAMETERS
    Niter = get_env('NITER', default=_Niter, vartype=int)   # number of iterations
    dt = get_env('DT', default=_dt, vartype=float)          # time step
    nWork = gcd(int(initSim), int(Niter))                   # number of frames on which to sum the active work before dumping
    skipDump = int(initSim/nWork)                           # number of work dumps corresponding to the initial iterations

    # EXECUTABLE PARAMETERS
    exec_dir = get_env('EXEC_DIR', default=_exec_dir, vartype=str)      # executable directory
    exec_name = get_env('EXEC_NAME', default=_exec_name[N >= _N_cell],  # executable name
        vartype=str)
    exec_path = path.join(exec_dir, exec_name)                          # executable path

    # OUTPUT FILES PARAMETERS
    launch = get_env('LAUNCH', default=_launch, vartype=float)  # launch identifier
    out_dir = get_env('OUT_DIR', default=_out_dir, vartype=str) # output directory
    sim_name = filename(N, phi, lp, launch)                     # simulation output name
    sim_dir = path.join(out_dir, sim_name)                      # simulation output directory name
    mkdir(sim_dir, replace=True)
    tmp_dir = path.join(sim_dir, 'tmp')                         # temporary files directory
    mkdir(tmp_dir, replace=True)
    tmp_template = '%010d.torque.dat'                           # template of temporary files
    out_file = path.join(sim_dir, sim_name + '.datG')           # simulation output file name

    # LAUNCH

    env = lambda i: {   # environment variables for simulation executables as function of gValues x nRuns index
        'N': str(N), 'LP': str(lp), 'PHI': str(phi),
        'TORQUE_PARAMETER': str(gValues[i//nRuns]),
        'SEED': str(seeds[i//nRuns, i%nRuns]),
        'FILE': path.join(tmp_dir, tmp_template % i),
        'DT': str(dt), 'NITER': str(Niter),
        'NWORK': str(nWork),
        'DUMP': str(0), 'PERIOD': str(1)}

    if slurm:   # using Slurm job scheduler

        slurm_launch = ['bash', _slurm_path, '-w']  # commands to submit Slurm job
        if slurm_partition != None: slurm_launch += ['-p', slurm_partition]
        if slurm_time != None: slurm_launch += ['-t', slurm_time]

        # LAUNCH
        procs, jobsID = [], []
        for i in range(gNum*nRuns):

            procs += [
                Popen(
                    ['%s \"{ %s %s; }\"' %
                        (str(' ').join(slurm_launch                 # Slurm submitting script
                            + ['-j', '\'' +  exec_path.split('/')[-1]
                                + ' %04i %s %04i\''
                                    % (i, env(i)['TORQUE_PARAMETER'], i%nRuns)]
                            + ([] if slurm_chain == []
                                else ['-c', str(slurm_chain[i])])),
                        str(' ').join(['%s=%s' % (key, env(i)[key]) # environment variables
                            for key in env(i)]),
                        exec_path)],                                # simulation executable
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
            for i in range(gNum*nRuns)]

    for proc in procs: proc.wait()  # wait for them to finish

    # LOAD TEMPORARY FILES

    tmp_out = []
    for i in range(gNum):
        tmp_out += [[]]
        for j in range(nRuns):
            tmp_out[-1] += [
                Dat(path.join(tmp_dir, tmp_template % (i*nRuns + j)),
                    loadWork=True)]

    # ARRAYS OF DATA

    activeWork = np.array(
        [[tmp_out[i][j].activeWork[skipDump:].mean()
            for j in range(nRuns)] for i in range(gNum)])
    activeWorkForce = np.array(
        [[tmp_out[i][j].activeWorkForce[skipDump:].mean()
            for j in range(nRuns)] for i in range(gNum)])
    activeWorkOri = np.array(
        [[tmp_out[i][j].activeWorkOri[skipDump:].mean()
            for j in range(nRuns)] for i in range(gNum)])
    orderParameter = np.array(
        [[tmp_out[i][j].orderParameter[skipDump:].mean()
            for j in range(nRuns)] for i in range(gNum)])
    torqueIntegral1 = np.array(
        [[tmp_out[i][j].torqueIntegral1[skipDump:].mean()
            for j in range(nRuns)] for i in range(gNum)])
    torqueIntegral2 = np.array(
        [[tmp_out[i][j].torqueIntegral2[skipDump:].mean()
            for j in range(nRuns)] for i in range(gNum)])

    # OUT

    with open(out_file, 'wb') as output:
        pickle.dump([
            exec_path,
            initSim, Niter, dt, nRuns, gValues,
            seed, seeds,
            N, lp, phi,
            activeWork, activeWorkForce, activeWorkOri, orderParameter,
                torqueIntegral1, torqueIntegral2],
            output)

    # CLEAN

    if get_env('CLEAN', default=True, vartype=bool):
        move(out_file, path.join(out_dir, sim_name + '.datG'))  # move output file to output directory
        rmr(sim_dir, ignore_errors=True)                        # delete simulation directory

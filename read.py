"""
Module read provides classes to read from data files produced by simulation
scripts.
"""

import struct
import numpy as np
import os
import pickle

from coll_dyn_activem.init import get_env
from coll_dyn_activem.maths import relative_positions, angle

class _Read:
    """
    Generic class to read binary output files.
    """

    def __init__(self, filename):
        """
        Load file.

        Parameters
        ----------
        filename : string
            Path to data file.
        """

        # FILE
        self.filename = filename
        self.file = open(self.filename, 'rb')
        self.fileSize = os.path.getsize(filename)

    def __del__(self):
        try:
            self.file.close()
            return True
        except AttributeError: return False # self.file was not loaded

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_value, tb): return self.__del__()

    def _bpe(self, type):
        """
        Returns number of bytes corresponding to type.

        Parameters
        ----------
        type : string
            Type of value to read.
        """

        return struct.calcsize(type)

    def _read(self, type):
        """
        Read element from file with type.

        Parameters
        ----------
        type : string
            Type of value to read.
        """

        return struct.unpack(type, self.file.read(self._bpe(type)))[0]

class Dat(_Read):
    """
    Read data files from simulations.

    (.dat: see coll_dyn_activem/particle.hpp -> class System &
    coll_dyn_activem/launch.py)
    NOTE: FORCE_DAT=True can be used to enforce the choice of .dat format.
    (.dat0: see coll_dyn_activem/particle.hpp -> class System0 &
    coll_dyn_activem/launch0.py)
    NOTE: FORCE_DAT0=True can be used to enforce the choice of .dat0 format.
    """

    def __init__(self, filename, loadWork=True):
        """
        Get data from header.

        Parameters
        ----------
        filename : string
            Path to data file.
        loadWork : bool or 'r'
            Load dump arrays. (default: True)
            NOTE: if loadWork=='r', force re-extract dumps from data file.
        """

        # FILE
        super().__init__(filename)

        try:

            try:

                ########
                # .DAT #
                ########
                self._type = 'dat'

                # HEADER INFORMATION
                self.N = self._read('i')                # number of particles
                self.lp = self._read('d')               # persistence length
                self.phi = self._read('d')              # packing fraction
                self.L = self._read('d')                # system size
                self.rho = self.N/(self.L**2)           # particle density
                self.g = self._read('d')                # torque parameter
                self.seed = self._read('i')             # random seed
                self.dt = self._read('d')               # time step
                self.framesWork = self._read('i')       # number of frames on which to sum the active work before dumping
                self.dumpParticles = self._read('b')    # dump positions and orientations to output file
                self.dumpPeriod = self._read('i')       # period of dumping of positions and orientations in number of frames

                # FILE PARTS LENGTHS
                self.headerLength = self.file.tell()                        # length of header in bytes
                self.particleLength = 5*self._bpe('d')*self.dumpParticles   # length the data of a single particle takes in a frame
                self.frameLength = self.N*self.particleLength               # length the data of a single frame takes in a file
                self.workLength = 8*self._bpe('d')                          # length the data of a single work and order parameter dump takes in a file

                # ESTIMATION OF NUMBER OF COMPUTED WORK AND ORDER SUMS AND FRAMES
                self.numberWork = (self.fileSize
                    - self.headerLength                                     # header
                    - self.frameLength                                      # first frame
                    )//(
                    self.framesWork*self.frameLength
                        + self.workLength)                                  # number of cumputed work sums
                self.frames = 0 if not(self.dumpParticles) else (
                    self.fileSize - self.headerLength
                    - self.numberWork*self.workLength)//self.frameLength    # number of frames which the file contains

                # FILE CORRUPTION CHECK
                if self.fileSize != (
                    self.headerLength                   # header
                    + self.frames*self.frameLength      # frames
                    + self.numberWork*self.workLength): # work sums
                    raise ValueError("Invalid data file size.")

                return

            except:

                if get_env('FORCE_DAT', default=False, vartype=bool):
                    raise ValueError(
                        "Invalid data file size ('%s')." % self.filename)

            try:

                #########
                # .DAT0 #
                #########
                self._type = 'dat0'

                # FILE
                _Read.__init__(self, filename)

                # HEADER INFORMATION
                self.N = self._read('i')                # number of particles
                self.epsilon = self._read('d')          # coefficient parameter of potential
                self.v0 = self._read('d')               # self-propulsion velocity
                self.D = self._read('d')                # translational diffusivity
                self.Dr = self._read('d')               # rotational diffusivity
                self.lp = self._read('d')               # persistence length
                self.phi = self._read('d')              # packing fraction
                self.L = self._read('d')                # system size
                self.rho = self.N/(self.L**2)           # particle density
                self.seed = self._read('i')             # random seed
                self.dt = self._read('d')               # time step
                self.framesWork = self._read('i')       # number of frames on which to sum the active work before dumping
                self.dumpParticles = self._read('b')    # dump positions and orientations to output file
                self.dumpPeriod = self._read('i')       # period of dumping of positions and orientations in number of frames

                # DIAMETERS
                self.diameters = np.empty((self.N,))    # array of diameters
                for i in range(self.N): self.diameters[i] = self._read('d')

                # FILE PARTS LENGTHS
                self.headerLength = self.file.tell()                        # length of header in bytes
                self.particleLength = 9*self._bpe('d')*self.dumpParticles   # length the data of a single particle takes in a frame
                self.frameLength = self.N*self.particleLength               # length the data of a single frame takes in a file
                self.workLength = 4*self._bpe('d')                          # length the data of a single work and order parameter dump takes in a file

                # ESTIMATION OF NUMBER OF COMPUTED WORK AND ORDER SUMS AND FRAMES
                self.numberWork = (self.fileSize
                    - self.headerLength                                     # header
                    - self.frameLength                                      # first frame
                    )//(
                    self.framesWork*self.frameLength
                        + self.workLength)                                  # number of cumputed work sums
                self.frames = 0 if not(self.dumpParticles) else (
                    self.fileSize - self.headerLength
                    - self.numberWork*self.workLength)//self.frameLength    # number of frames which the file contains

                # FILE CORRUPTION CHECK
                if self.fileSize != (
                    self.headerLength                   # header
                    + self.frames*self.frameLength      # frames
                    + self.numberWork*self.workLength): # work sums
                    raise ValueError(
                        "Invalid data file size ('%s')." % self.filename)

                return

            except:

                if get_env('FORCE_DAT0', default=False, vartype=bool):
                    raise ValueError(
                        "Invalid data file size ('%s')." % self.filename)

            try:

                #########
                # .DATN #
                #########
                self._type = 'datN'

                # FILE
                _Read.__init__(self, filename)

                # HEADER INFORMATION
                self.N = self._read('i')                # number of particles
                self.epsilon = self._read('d')          # coefficient parameter of potential
                self.v0 = self._read('d')               # self-propulsion velocity
                self.D = self._read('d')                # translational diffusivity
                self.Dr = self._read('d')               # rotational diffusivity
                self.lp = self._read('d')               # persistence length
                self.phi = self._read('d')              # packing fraction
                self.L = self._read('d')                # system size
                self.rho = self.N/(self.L**2)           # particle density
                self.seed = self._read('i')             # random seed
                self.dt = self._read('d')               # time step
                self.framesWork = 1                     # number of frames on which to sum the active work before dumping (here set as 0 to avoid crash of some functions)
                self.dumpParticles = True               # dump positions and orientations to output file
                self.dumpPeriod = 1                     # period of dumping of positions and orientations in number of frames

                # FRAMES
                self.init = self._read('i')                             # initialisation number of iterations
                self.NLin = self._read('i')                             # number of linearly splaced blocks of frames
                self.NiterLin = self._read('i')                         # number of iterations in blocks
                self.NLog = self._read('i')                             # number of logarithmically spaced frames in blocks
                self.frames = self._read('i')                           # number of frames
                self.frameIndices = []                                  # frame indices which were saved
                for _ in range(self.frames):
                    self.frameIndices += [self._read('i')]
                self.frames += 1                                        # count frame 0
                self.frameIndices = np.array([0] + self.frameIndices)   # count frame 0

                # DIAMETERS
                self.diameters = np.empty((self.N,))    # array of diameters
                for i in range(self.N): self.diameters[i] = self._read('d')

                # FILE PARTS LENGTHS
                self.headerLength = self.file.tell()                        # length of header in bytes
                self.particleLength = 9*self._bpe('d')                      # length the data of a single particle takes in a frame
                self.frameLength = self.N*self.particleLength               # length the data of a single frame takes in a file
                self.workLength = 0*self._bpe('d')                          # length the data of a single work and order parameter dump takes in a file

                # FILE CORRUPTION CHECK
                if self.fileSize != (
                    self.headerLength                   # header
                    + self.frames*self.frameLength):    # frames
                    raise ValueError(
                        "Invalid data file size ('%s')." % self.filename)

                return

            except:

                if get_env('FORCE_DAT0', default=False, vartype=bool):
                    raise ValueError(
                        "Invalid data file size ('%s')." % self.filename)

            if not(hasattr(self, '_type')):
                raise TypeError(
                    "No appropriate file type found ('%s')." % self.filename)

        finally:

            # COMPUTED NORMALISED RATE OF ACTIVE WORK
            self._loadWork(load=loadWork)

    def getWork(self, time0, time1):
        """
        Returns normalised active work between frames `time0' and `time1'.

        Parameters
        ----------
        time0 : int
            Initial frame.
        time1 : int
            Final frame.

        Returns
        -------
        work : float
            Normalised rate of active work.
        """

        work = np.sum(list(map(
            lambda t: self._work(t),
            range(int(time0), int(time1)))))
        work /= self.N*self.dt*(time1 - time0)

        return work

    def getPositions(self, time, *particle, **kwargs):
        """
        Returns positions of particles at time.

        Parameters
        ----------
        time : int
            Frame.
        particle : int
            Indexes of particles.
            NOTE: if none is given, then all particles are returned.

        Optional keyword parameters
        ---------------------------
        centre : (2,) array like
            Returns position relative to `centre'.

        Returns
        -------
        positions : (*, 2) float Numpy array
            Positions at `time'.
        """

        if particle == (): particle = range(self.N)

        positions = np.array(list(map(
            lambda index: self._position(time, index),
            particle)))

        if 'centre' in kwargs:
            return relative_positions(positions, kwargs['centre'], self.L)
        return positions

    def getDisplacements(self, time0, time1, *particle, jump=1, norm=False):
        """
        Returns displacements of particles between `time0' and `time1'.

        Parameters
        ----------
        time0 : int
            Initial frame.
        time1 : int
            Final frame.
        particle : int
            Indexes of particles.
            NOTE: if none is given, then all particles are returned.
        jump : int
            Period in number of frames at which to check if particles have
            crossed any boundary. (default: 1)
            NOTE: `jump' must be chosen so that particles do not move a distance
                  greater than half the box size during this time.
            NOTE: This is only relevant for .dat files since these do not embed
                  unfolded positions.
        norm : bool
            Return norm of displacements rather than 2D displacements.
            (default: False)

        Returns
        -------
        displacements : [not(norm)] (*, 2) float Numpy array
                        [norm] (*,) float Numpy array
            Displacements between `time0' and `time1'.
        """

        if particle == (): particle = range(self.N)
        time0 = int(time0)
        time1 = int(time1)

        if self._type == 'dat':

            jump = int(jump)

            displacements = -self.getPositions(time0, *particle)

            increments = np.zeros((len(particle), 2))
            positions1 = -displacements.copy()
            for t in list(range(time0, time1, jump)) + [time1 - 1]:
                positions0 = positions1.copy()
                positions1 = self.getPositions(t + 1, *particle)
                increments += (
                    ((positions0 - self.L/2)*(positions1 - self.L/2) < 0)   # if position switches "sign"
                    *(np.abs(positions0 - self.L/2) > self.L/4)             # (and) if particle is not in the centre of the box
                    *np.sign(positions0 - self.L/2)                         # "sign" of position
                    *self.L)

            displacements += positions1 + increments

        else:

            displacements = (
                np.array(list(map(
                    lambda index: self._unfolded_position(time1, index),
                    particle)))
                - np.array(list(map(
                    lambda index: self._unfolded_position(time0, index),
                    particle))))

        if norm: return np.sqrt(np.sum(displacements**2, axis=-1))
        return displacements

    def getDistancePositions(self, time, particle0, particle1):
        """
        Returns distance between particles with indexes `particle0' and
        `particle1' at time `time' and their respective positions.

        Parameters
        ----------
        time : int
            Index of frame.
        particle0 : int
            Index of first particle.
        particle1 : int
            Index of second particle.

        Returns
        -------
        dist : float
            Distance between particles.
        pos0 : (2,) float numpy array
            Position of particle0.
        pos1 : (2,) float numpy array
            Position of particle1.
        """

        pos0, pos1 = self.getPositions(time, particle0, particle1)
        return np.sqrt(
            self._diffPeriodic(pos0[0], pos1[0])**2
            + self._diffPeriodic(pos0[1], pos1[1])**2), pos0, pos1

    def getDistance(self, time, particle0, particle1):
        """
        Returns distance between particles with indexes `particle0' and
        `particle1' at time `time'.

        Parameters
        ----------
        time : int
            Index of frame.
        particle0 : int
            Index of first particle.
        particle1 : int
            Index of second particle.

        Returns
        -------
        dist : float
            Distance between particles.
        """

        return self.getDistancePositions(time, particle0, particle1)[0]

    def getOrientations(self, time, *particle):
        """
        Returns orientations of particles at time.

        Parameters
        ----------
        time : int
            Frame
        particle : int
            Indexes of particles.
            NOTE: if none is given, then all particles are returned.

        Returns
        -------
        orientations : (*,) float Numpy array
            Orientations at `time'.
        """

        if particle == (): particle = range(self.N)

        return np.array(list(map(
            lambda index: self._orientation(time, index),
            particle)))

    def getVelocities(self, time, *particle, norm=False):
        """
        Returns velocities of particles at time.

        Parameters
        ----------
        time : int
            Frame.
        particle : int
            Indexes of particles.
            NOTE: if none is given, then all particles are returned.
        norm : bool
            Return norm of velocities rather than 2D velocities.
            (default: False)

        Returns
        -------
        velocities : [not(norm)] (*, 2) float Numpy array
                     [norm] (*,) float Numpy array
            Velocities at `time'.
        """

        if particle == (): particle = range(self.N)

        velocities = np.array(list(map(
            lambda index: self._velocity(time, index),
            particle)))
        if norm: return np.sqrt(np.sum(velocities**2, axis=-1))
        return velocities

    def getDirections(self, time, *particle):
        """
        Returns normalised self-propulsion vector of particles at time.

        Parameters
        ----------
        time : int
            Frame
        particle : int
            Indexes of particles.
            NOTE: if none is given, then all particles are returned.

        Returns
        -------
        orientations : (*, 2) float Numpy array
            Unitary self-propulsion vectors at `time'.
        """

        if particle == (): particle = range(self.N)

        return np.array(list(map(
            lambda theta: np.array([np.cos(theta), np.sin(theta)]),
            self.getOrientations(time, *particle))))

    def getPropulsions(self, time, *particle, norm=False):
        """
        Returns self-propulsion vectors of particles at time.

        Parameters
        ----------
        time : int
            Frame.
        particle : int
            Indexes of particles.
            NOTE: if none is given, then all particles are returned.
        norm : bool
            Return norm of self-propulsion vectors rather than 2D
            self-propulsion vectors.
            (default: False)

        Returns
        -------
        propulsions : [not(norm)] (*, 2) float Numpy array
                      [norm] (*,) float Numpy array
            Self-propulsion vectors at `time'.
        """

        if particle == (): particle = range(self.N)

        propulsions = np.array(list(map(
            lambda index: self._propulsion(time, index),
            particle)))
        if norm: return np.sqrt(np.sum(propulsions**2, axis=-1))
        return propulsions

    def getOrderParameter(self, time, norm=False):
        """
        Returns order parameter, i.e. mean direction, at time.

        Parameters
        ----------
        time : int
            Frame.
        norm : bool
            Return norm of order parameter. (default: False)

        Returns
        -------
        orderParameter : float if `norm' else (2,) float Numpy array
            Order parameter at `time'.
        """

        orderParameter = np.sum(self.getDirections(time), axis=0)/self.N
        if norm: return np.sqrt(np.sum(orderParameter**2))
        return orderParameter

    def getGlobalPhase(self, time):
        """
        Returns global phase at time `time'.

        Parameters
        ----------
        time : int
            Frame.

        Returns
        -------
        phi : float
            Global phase in radians.
        """

        return angle(*self.getOrderParameter(time, norm=False))

    def getTorqueIntegral0(self, time0, time1):
        """
        Returns normalised zeroth integral in the expression of the modified
        active work for control-feedback modified dynamics from `time0' to
        `time1'.
        (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#ABP%20cloning%20algorithm)

        NOTE: Using Stratonovitch convention.

        Parameters
        ----------
        time0 : int
            Initial frame.
        time1 : int
            Final frame.

        Returns
        -------
        torqueIntegral : float
            Normalised integral.
        """

        if time0 == time1: return 0

        torqueIntegral = 0
        for time in range(time0, time1):
            torqueIntegral += np.sum(
                (self.getOrderParameter(time, norm=True)
                    *np.sin(
                        self.getOrientations(time)
                            - self.getGlobalPhase(time))
                + self.getOrderParameter(time + 1, norm=True)
                    *np.sin(
                        self.getOrientations(time + 1)
                            - self.getGlobalPhase(time + 1)))
                *(self.getOrientations(time + 1) - self.getOrientations(time))
            )/2
            # torqueIntegral += np.sum(
            #     list(map(
            #         lambda i: np.sum(
            #             np.sin(self.getOrientations(time, i)
            #                 - self.getOrientations(time))
            #             + np.sin(self.getOrientations(time + 1, i)
            #                 - self.getOrientations(time + 1)))
            #             *(self.getOrientations(time + 1, i)
            #                 - self.getOrientations(time, i)),
            #         range(self.N)))
            # )/2

        return torqueIntegral/(self.N*(time1 - time0)*self.dt)

    def getTorqueIntegral1(self, time0, time1):
        """
        Returns normalised first integral in the expression of the modified
        active work for control-feedback modified dynamics from `time0' to
        `time1'.
        (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#ABP%20cloning%20algorithm)

        NOTE: Using Stratonovitch convention.

        Parameters
        ----------
        time0 : int
            Initial frame.
        time1 : int
            Final frame.

        Returns
        -------
        torqueIntegral : float
            Normalised integral.
        """

        if time0 == time1: return 0

        torqueIntegral = 0
        for time in range(time0, time1):
            torqueIntegral += (
                self.getOrderParameter(time, norm=True)**2
                + self.getOrderParameter(time + 1, norm=True)**2)/2

        return torqueIntegral/(time1-time0)

    def getTorqueIntegral2(self, time0, time1):
        """
        Returns normalised second integral in the expression of the modified
        active work for control-feedback modified dynamics from `time0' to
        `time1'.
        (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#ABP%20cloning%20algorithm)

        NOTE: Using Stratonovitch convention.

        Parameters
        ----------
        time0 : int
            Initial frame.
        time1 : int
            Final frame.

        Returns
        -------
        torqueIntegral : float
            Normalised integral.
        """

        if time0 == time1: return 0

        torqueIntegral = 0
        for time in range(time0, time1):
            torqueIntegral += (
                (self.getOrderParameter(time, norm=True)**2)
                    *np.sum(np.sin(
                        self.getOrientations(time)
                            - self.getGlobalPhase(time))**2)
                + (self.getOrderParameter(time + 1, norm=True)**2)
                    *np.sum(np.sin(
                        self.getOrientations(time + 1)
                            - self.getGlobalPhase(time + 1))**2))/2

        return torqueIntegral/(self.N*(time1-time0))

    def toGrid(self, time, array, nBoxes=None, box_size=None, centre=None,
        average=True):
        """
        Maps square sub-system of centre `centre' and length `box_size' to a
        square grid with `nBoxes' boxes in every direction, and associates to
        each box of this grid the sum or averaged value of the (self.N, *)-array
        `array' over the indexes corresponding to particles within this box at
        time `time'.

        Parameters
        ----------
        time : int
            Frame index.
        array : (self.N, *) float array-like
            Array of values to be put on the grid.
        nBoxes : int
            Number of grid boxes in each direction. (default: None)
            NOTE: if nBoxes==None, then nBoxes = int(sqrt(self.N)).
        box_size : float
            Length of the sub-system to consider.
            NOTE: if box_size==None, then box_size = self.L. (default: None)
        centre : array-like
            Coordinates of the centre of the sub-system. (default: None)
            NOTE: if centre==None, then centre = (0, 0).
        average : bool
            Return average of quantity per box, otherwise return sum.
            (default: False)

        Returns
        -------
        grid : (nBoxes, nBoxes, *) float Numpy array
            Averaged grid.
        """

        array = np.array(array)
        if array.shape[0] != self.N: raise ValueError(
            "Array first-direction length different than number of particles.")

        if nBoxes == None: nBoxes = np.sqrt(self.N)
        nBoxes = int(nBoxes)

        if box_size == None: box_size = self.L

        try:
            if centre == None: centre = (0, 0)
        except ValueError: pass
        centre = np.array(centre)

        grid = np.zeros((nBoxes,)*2 + array.shape[1:])
        sumN = np.zeros((nBoxes,)*2)	# array of the number of particles in each grid box

        in_box = lambda particle: (
            np.max(np.abs(self.getPositions(time, particle, centre=centre)))
            <= box_size/2)
        positions = self.getPositions(time, centre=centre)
        for particle in range(self.N):
            if in_box(particle):
                grid_index = tuple(np.array(
                    ((positions[particle] + box_size/2)//(box_size/nBoxes))
                    % ((nBoxes,)*2),
                    dtype=int))
                grid[grid_index] += array[particle]
                sumN[grid_index] += 1
        sumN = np.reshape(sumN,
            (nBoxes,)*2 + (1,)*len(array.shape[1:]))

        if average: return np.divide(grid, sumN,
            out=np.zeros(grid.shape), where=sumN!=0)
        return grid

    def _loadWork(self, load=True):
        """
        Load active work, order parameter, and torque integrals dumps from
            * self.filename + '.work.pickle': normalised rate of active work,
            * self.filename + '.work.force.pickle': force part of the normalised
            rate of active work,
            * self.filename + '.work.ori.pickle': orientation part of the
            normalised rate of active work,
            * self.filename + '.order.vec.pickle': vectorial order parameter,
            * self.filename + '.order.pickle': order parameter,
            * self.filename + '.torque.int1.pickle': first torque integral,
            * self.filename + '.torque.int2.pickle': second torque integral,
        if they exist or extract them from data file and then pickle them to
        files.

        Parameters
        ----------
        load : bool or 'r'
            Load dump arrays. (default: True)
            NOTE: if loadWork=='r', force re-extract dumps from data file.
        """

        if not(load) or self._type == 'datN': return

        # ACTIVE WORK

        try:    # try loading

            if load == 'r': raise FileNotFoundError

            with open(self.filename + '.work.pickle', 'rb') as workFile:
                self.activeWork = pickle.load(workFile)
                if self.activeWork.size != self.numberWork:
                    raise ValueError("Invalid active work array size.")

        except (FileNotFoundError, EOFError):   # active work file does not exist or file is empty

            # COMPUTE
            self.activeWork = np.empty(self.numberWork)
            for i in range(self.numberWork):
                self.file.seek(
                    self.headerLength                           # header
                    + self.frameLength                          # frame with index 0
                    + (1 + i)*self.framesWork*self.frameLength  # all following packs of self.framesWork frames
                    + i*self.workLength)                        # previous values of the active work
                self.activeWork[i] = self._read('d')

            # DUMP
            with open(self.filename + '.work.pickle', 'wb') as workFile:
                pickle.dump(self.activeWork, workFile)

        # ACTIVE WORK (FORCE)

        try:    # try loading

            if load == 'r': raise FileNotFoundError

            with open(self.filename + '.work.force.pickle', 'rb') as workFile:
                self.activeWorkForce = pickle.load(workFile)
                if self.activeWorkForce.size != self.numberWork:
                    raise ValueError("Invalid active work (force) array size.")

        except (FileNotFoundError, EOFError):   # active work (force) file does not exist or file is empty

            # COMPUTE
            self.activeWorkForce = np.empty(self.numberWork)
            for i in range(self.numberWork):
                self.file.seek(
                    self.headerLength                           # header
                    + self.frameLength                          # frame with index 0
                    + (1 + i)*self.framesWork*self.frameLength  # all following packs of self.framesWork frames
                    + i*self.workLength                         # previous values of the active work
                    + self._bpe('d'))                           # value of active work
                self.activeWorkForce[i] = self._read('d')

            # DUMP
            with open(self.filename + '.work.force.pickle', 'wb') as workFile:
                pickle.dump(self.activeWorkForce, workFile)

        # ACTIVE WORK (ORIENTATION)

        try:    # try loading

            if load == 'r': raise FileNotFoundError

            with open(self.filename + '.work.ori.pickle', 'rb') as workFile:
                self.activeWorkOri = pickle.load(workFile)
                if self.activeWorkOri.size != self.numberWork:
                    raise ValueError("Invalid active work (ori) array size.")

        except (FileNotFoundError, EOFError):   # active work (orientation) file does not exist or file is empty

            # COMPUTE
            self.activeWorkOri = np.empty(self.numberWork)
            for i in range(self.numberWork):
                self.file.seek(
                    self.headerLength                           # header
                    + self.frameLength                          # frame with index 0
                    + (1 + i)*self.framesWork*self.frameLength  # all following packs of self.framesWork frames
                    + i*self.workLength                         # previous values of the active work
                    + 2*self._bpe('d'))                         # value of active work and force part of active work
                self.activeWorkOri[i] = self._read('d')

            # DUMP
            with open(self.filename + '.work.ori.pickle', 'wb') as workFile:
                pickle.dump(self.activeWorkOri, workFile)

        # ORDER PARAMETER

        try:    # try loading

            if load == 'r': raise FileNotFoundError

            with open(self.filename + '.order.pickle', 'rb') as workFile:
                self.orderParameter = pickle.load(workFile)
                if self.orderParameter.size != self.numberWork:
                    raise ValueError("Invalid order parameter array size.")

        except (FileNotFoundError, EOFError):   # order parameter file does not exist or file is empty

            # COMPUTE
            self.orderParameter = np.empty(self.numberWork)
            for i in range(self.numberWork):
                self.file.seek(
                    self.headerLength                           # header
                    + self.frameLength                          # frame with index 0
                    + (1 + i)*self.framesWork*self.frameLength  # all following packs of self.framesWork frames
                    + i*self.workLength                         # previous values of the active work
                    + 3*self._bpe('d'))                         # values of the different parts of active work
                self.orderParameter[i] = self._read('d')

            # DUMP
            with open(self.filename + '.order.pickle', 'wb') as workFile:
                pickle.dump(self.orderParameter, workFile)

        if self._type == 'dat':

            # VECTORIAL ORDER PARAMETER

            try:    # try loading

                if load == 'r': raise FileNotFoundError

                with open(self.filename + '.order.vec.pickle', 'rb') as workFile:
                    self.orderParameterVec = pickle.load(workFile)
                    if self.orderParameterVec.size != 2*self.numberWork:
                        raise ValueError("Invalid order parameter array size.")

            except (FileNotFoundError, EOFError):   # order parameter file does not exist or file is empty

                # COMPUTE
                self.orderParameterVec = np.empty((self.numberWork, 2))
                for i in range(self.numberWork):
                    self.file.seek(
                        self.headerLength                           # header
                        + self.frameLength                          # frame with index 0
                        + (1 + i)*self.framesWork*self.frameLength  # all following packs of self.framesWork frames
                        + i*self.workLength                         # previous values of the active work
                        + 4*self._bpe('d'))                         # values of the different parts of active work
                    self.orderParameterVec[i] = np.array(
                        [self._read('d'), self._read('d')])

                # DUMP
                with open(self.filename + '.order.vec.pickle', 'wb') as workFile:
                    pickle.dump(self.orderParameterVec, workFile)

            # FIRST TORQUE INTEGRAL

            try:    # try loading

                if load == 'r': raise FileNotFoundError

                with open(self.filename + '.torque.int1.pickle', 'rb') as iFile:
                    self.torqueIntegral1 = pickle.load(iFile)
                    if self.torqueIntegral1.size != self.numberWork:
                        raise ValueError("Invalid 1st torque int. array size.")

            except (FileNotFoundError, EOFError):   # first torque integral file does not exist or file is empty

                # COMPUTE
                self.torqueIntegral1 = np.empty(self.numberWork)
                for i in range(self.numberWork):
                    self.file.seek(
                        self.headerLength                           # header
                        + self.frameLength                          # frame with index 0
                        + (1 + i)*self.framesWork*self.frameLength  # all following packs of self.framesWork frames
                        + i*self.workLength                         # previous values of the active work
                        + 6*self._bpe('d'))                         # values of the different parts of active work and the order parameter
                    self.torqueIntegral1[i] = self._read('d')

                # DUMP
                with open(self.filename + '.torque.int1.pickle', 'wb') as iFile:
                    pickle.dump(self.torqueIntegral1, iFile)

            # SECOND TORQUE INTEGRAL

            try:    # try loading

                if load == 'r': raise FileNotFoundError

                with open(self.filename + '.torque.int2.pickle', 'rb') as iFile:
                    self.torqueIntegral2 = pickle.load(iFile)
                    if self.torqueIntegral2.size != self.numberWork:
                        raise ValueError("Invalid 2nd torque int. array size.")

            except (FileNotFoundError, EOFError):   # second torque integral file does not exist or file is empty

                # COMPUTE
                self.torqueIntegral2 = np.empty(self.numberWork)
                for i in range(self.numberWork):
                    self.file.seek(
                        self.headerLength                           # header
                        + self.frameLength                          # frame with index 0
                        + (1 + i)*self.framesWork*self.frameLength  # all following packs of self.framesWork frames
                        + i*self.workLength                         # previous values of the active work
                        + 7*self._bpe('d'))                         # values of the different parts of active work, the order parameter, and the first torque integral
                    self.torqueIntegral2[i] = self._read('d')

                # DUMP
                with open(self.filename + '.torque.int2.pickle', 'wb') as iFile:
                    pickle.dump(self.torqueIntegral2, iFile)

    def _position(self, time, particle):
        """
        Returns array of position of particle at time.

        Parameters
        ----------
        time : int
            Frame.
        particle : int
            Index of particle.

        Returns
        -------
        position : (2,) float Numpy array
            Position of `particle' at `time'.
        """

        time = self._getFrameIndex(time)

        self.file.seek(
            self.headerLength                                           # header
            + time*self.frameLength                                     # other frames
            + particle*self.particleLength                              # other particles
            + (np.max([time - 1, 0])//self.framesWork)*self.workLength) # active work sums (taking into account the frame with index 0)
        return np.array([self._read('d'), self._read('d')])

    def _orientation(self, time, particle):
        """
        Returns orientation of particle at time.

        Parameters
        ----------
        time : int
            Frame.
        particle : int
            Index of particle.

        Returns
        -------
        orientation : (2,) float Numpy array
            Orientation of `particle' at `time'.
        """

        time = self._getFrameIndex(time)

        self.file.seek(
            self.headerLength                                           # header
            + time*self.frameLength                                     # other frames
            + particle*self.particleLength                              # other particles
            + 2*self._bpe('d')                                          # positions
            + (np.max([time - 1, 0])//self.framesWork)*self.workLength) # active work sums (taking into account the frame with index 0)
        return self._read('d')

    def _velocity(self, time, particle):
        """
        Returns array of velocity of particle at time.

        Parameters
        ----------
        time : int
            Frame.
        particle : int
            Index of particle.

        Returns
        -------
        velocity : (2,) float Numpy array
            Velocity of `particle' at `time'.
        """

        time = self._getFrameIndex(time)

        self.file.seek(
            self.headerLength                                           # header
            + time*self.frameLength                                     # other frames
            + particle*self.particleLength                              # other particles
            + 3*self._bpe('d')                                          # positions and orientation
            + (np.max([time - 1, 0])//self.framesWork)*self.workLength) # active work sums (taking into account the frame with index 0)
        return np.array([self._read('d'), self._read('d')])

    def _propulsion(self, time, particle):
        """
        Returns array of self-propulsion vector of particle at time.

        Parameters
        ----------
        time : int
            Frame.
        particle : int
            Index of particle.

        Returns
        -------
        position : (2,) float Numpy array
            Self-propulsion vector of `particle' at `time'.
        """

        if not(self._type in ('dat0', 'datN')):
            return self.getDirections(time, particle)[0]

        time = self._getFrameIndex(time)

        self.file.seek(
            self.headerLength                                           # header
            + time*self.frameLength                                     # other frames
            + particle*self.particleLength                              # other particles
            + 5*self._bpe('d')                                          # positions, orientation, and velocities
            + (np.max([time - 1, 0])//self.framesWork)*self.workLength) # active work sums (taking into account the frame with index 0)
        return np.array([self._read('d'), self._read('d')])

    def _unfolded_position(self, time, particle):
        """
        Returns array of unfolded position of particle at time.

        Parameters
        ----------
        time : int
            Frame.
        particle : int
            Index of particle.

        Returns
        -------
        position : (2,) float Numpy array
            Position of `particle' at `time'.
        """

        if not(self._type in ('dat0', 'datN')):
            raise AttributeError(
                'Unfolded positions are not available for .%s files.'
                % self._type)

        time = self._getFrameIndex(time)

        self.file.seek(
            self.headerLength                                           # header
            + time*self.frameLength                                     # other frames
            + particle*self.particleLength                              # other particles
            + 7*self._bpe('d')                                          # positions, orientation, velocities, and propulsions
            + (np.max([time - 1, 0])//self.framesWork)*self.workLength) # active work sums (taking into account the frame with index 0)
        return np.array([self._read('d'), self._read('d')])

    def _work(self, time):
        """
        Returns active work between `time' and `time' + 1.

        Parameters
        ----------
        time : int
            Frame.

        Returns
        -------
        work : float
            Normalised rate of active work between `time' and `time' + 1.
        """

        time = self._getFrameIndex(time)

        work = np.sum(list(map(         # sum over time
            lambda u, dr: np.dot(u,dr), # sum over particles
            *(self.getDisplacements(time, time + 1),
                self.getDirections(time)
                    + self.getDirections(time + 1)))))/2

        return work

    def _diffPeriodic(self, x0, x1):
        """
        Returns algebraic distance from x0 to x1 taking into account periodic
        boundary conditions.

        Parameters
        ----------
        x0 : float
            Coordinate of first point.
        x1 : float
            Coordinate of second point.

        Returns
        -------
        diff : float
            Algebraic distance from x0 to x1.
        """

        diff = x1 - x0
        if np.abs(diff) <= self.L/2: return diff

        diff = (1 - 2*(diff > 0))*np.min([
            np.abs(x0) + np.abs(self.L - x1), np.abs(self.L - x0) + np.abs(x1)])
        return diff

    def _getFrameIndex(self, frame):
        """
        Returns index of frame in file.

        NOTE: This function is meant mainly for .datN files.

        Parameters
        ----------
        frame : int
            Requested frame index.

        Returns
        -------
        frameFile : int
            Index of frame in file.
        """

        frame = int(frame)

        if self._type in ('dat', 'dat0'): return frame
        try:
            return np.where(self.frameIndices == frame)[0][0]
        except IndexError: raise ValueError("Frame %i not in file." % frame)

class DatR(_Read):
    """
    Read data files from simulations of interacting Brownian rotors.

    (see coll_dyn_activem/particle.hpp -> class Rotors &
    coll_dyn_activem/launchR.py)
    """

    def __init__(self, filename, loadOrder=True):
        """
        Get data from header.

        Parameters
        ----------
        filename : string
            Path to data file.
        loadOrder : bool or 'r'
            Load dump arrays. (default: True)
            NOTE: if loadOrder=='r', force re-extract dumps from data file.
        """

        # FILE
        super().__init__(filename)

        # HEADER INFORMATION
        self.N = self._read('i')            # number of rotors
        self.Dr = self._read('d')           # rotational diffusivity
        self.g = self._read('d')            # aligning torque parameter
        self.dt = self._read('d')           # time step
        self.framesOrder = self._read('i')  # number of frames on which to average the order parameter before dumping
        self.dumpRotors = self._read('b')   # dump orientations to output file
        self.dumpPeriod = self._read('i')   # period of dumping of orientations in number of frames
        self.seed = self._read('i')         # random seed

        # FILE PARTS LENGTHS
        self.headerLength = self.file.tell()        # length of header in bytes
        self.rotorLength = 1*self._bpe('d')         # length the data of a single rotor takes in a frame
        self.frameLength = self.N*self.rotorLength  # length the data of a single frame takes in a file
        self.orderLength = 2*self._bpe('d')         # length the data of a single order parameter dump takes in a file

        # ESTIMATION OF NUMBER OF FRAMES
        self.numberOrder = (self.fileSize
            - self.headerLength                                     # header
            - self.frameLength                                      # first frame
            )//(
            self.framesOrder*self.frameLength
                + self.orderLength)                                 # number of cumputed order sums
        self.frames = 0 if not(self.dumpRotors) else (
            self.fileSize - self.headerLength
            - self.numberOrder*self.orderLength)//self.frameLength  # number of frames which the file contains

        # FILE CORRUPTION CHECK
        if self.fileSize != (
            self.headerLength                       # header
            + self.frames*self.frameLength          # frames
            + self.numberOrder*self.orderLength):   # work sums
            raise ValueError("Invalid data file size.")

        # COMPUTED ORDER PARAMETER
        self._loadOrder(load=loadOrder)

    def getOrientations(self, time, *rotor):
        """
        Returns orientations of rotors at time.

        Parameters
        ----------
        time : int
            Frame
        rotor : int
            Indexes of rotors.
            NOTE: if none is given, then all rotors are returned.

        Returns
        -------
        orientations : (*,) float Numpy array
            Orientations at `time'.
        """

        if rotor == (): rotor = range(self.N)

        return np.array(list(map(
            lambda index: self._orientation(time, index),
            rotor)))

    def getDirections(self, time, *rotor):
        """
        Returns self-propulsion vector of rotors at time.

        Parameters
        ----------
        time : int
            Frame
        rotor : int
            Indexes of rotors.
            NOTE: if none is given, then all rotors are returned.

        Returns
        -------
        orientations : (*, 2) float Numpy array
            Unitary self-propulsion vectors at `time'.
        """

        if rotor == (): rotor = range(self.N)

        return np.array(list(map(
            lambda theta: np.array([np.cos(theta), np.sin(theta)]),
            self.getOrientations(time, *rotor))))

    def getOrderParameter(self, time, norm=False):
        """
        Returns order parameter, i.e. mean direction, at time.

        Parameters
        ----------
        time : int
            Frame.
        norm : bool
            Return norm of order parameter. (default: False)

        Returns
        -------
        orderParameter : float if `norm' else (2,) float Numpy array
            Order parameter at `time'.
        """

        orderParameter = np.sum(self.getDirections(time), axis=0)/self.N
        if norm: return np.sqrt(np.sum(orderParameter**2))
        return orderParameter

    def getGlobalPhase(self, time):
        """
        Returns global phase at time `time'.

        Parameters
        ----------
        time : int
            Frame.

        Returns
        -------
        phi : float
            Global phase in radians.
        """

        return angle(*self.getOrderParameter(time, norm=False))

    def _orientation(self, time, rotor):
        """
        Returns orientation of rotor at time.

        Parameters
        ----------
        time : int
            Frame.
        particle : int
            Index of rotor.

        Returns
        -------
        orientation : (2,) float Numpy array
            Orientation of `rotor' at `time'.
        """

        self.file.seek(
            self.headerLength                                               # header
            + time*self.frameLength                                         # other frames
            + rotor*self.rotorLength                                        # other rotors
            + (np.max([time - 1, 0])//self.framesOrder)*self.orderLength)   # active work sums (taking into account the frame with index 0)
        return self._read('d')

    def _loadOrder(self, load=True):
        """
        Load order parameter dumps from
            * self.filename + '.order.pickle': order parameter,
            * self.filename + '.order.sq.pickle': squared order parameter,
        if they exist or extract them from data file and then pickle them to
        files.

        Parameters
        ----------
        load : bool or 'r'
            Load dump arrays. (default: True)
            NOTE: if loadOrder=='r', force re-extract dumps from data file.
        """

        if not(load): return

        # ORDER PARAMETER

        try:    # try loading

            if load == 'r': raise FileNotFoundError

            with open(self.filename + '.order.pickle', 'rb') as orderFile:
                self.orderParameter = pickle.load(orderFile)
                if self.orderParameter.size != self.numberOrder:
                    raise ValueError("Invalid order parameter array size.")

        except (FileNotFoundError, EOFError):   # order parameter file does not exist or file is empty

            # COMPUTE
            self.orderParameter = np.empty(self.numberOrder)
            for i in range(self.numberOrder):
                self.file.seek(
                    self.headerLength                           # header
                    + self.frameLength                          # frame with index 0
                    + (1 + i)*self.framesOrder*self.frameLength # all following packs of self.framesOrder frames
                + i*self.orderLength)                           # previous values of the order parameter
                self.orderParameter[i] = self._read('d')

            # DUMP
            with open(self.filename + '.order.pickle', 'wb') as orderFile:
                pickle.dump(self.orderParameter, orderFile)

        # SQUARED ORDER PARAMETER

        try:    # try loading

            if load == 'r': raise FileNotFoundError

            with open(self.filename + '.order.sq.pickle', 'rb') as orderFile:
                self.orderParameterSq = pickle.load(orderFile)
                if self.orderParameterSq.size != self.numberOrder:
                    raise ValueError("Invalid sq. order parameter array size.")

        except (FileNotFoundError, EOFError):   # squared order parameter file does not exist or file is empty

            # COMPUTE
            self.orderParameterSq = np.empty(self.numberOrder)
            for i in range(self.numberOrder):
                self.file.seek(
                    self.headerLength                           # header
                    + self.frameLength                          # frame with index 0
                    + (1 + i)*self.framesOrder*self.frameLength # all following packs of self.framesOrder frames
                    + i*self.orderLength                        # previous values of the order parameter
                    + self._bpe('d'))                           # value of active work
                self.orderParameterSq[i] = self._read('d')

            # DUMP
            with open(self.filename + '.order.sq.pickle', 'wb') as orderFile:
                pickle.dump(self.orderParameterSq, orderFile)

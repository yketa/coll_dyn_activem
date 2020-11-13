"""
Module flow provides classes to compute and analyse displacements, velocities,
and orientations in order to characterise the flow of systems of active
particles.

(see https://yketa.github.io/PhD_Wiki/#Flow%20characteristics)
(see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#ABP%20flow%20characteristics)
"""

import numpy as np
from scipy.stats import norm as norm_gen
from scipy.special import lambertw
from collections import OrderedDict
from operator import itemgetter

from coll_dyn_activem.read import Dat
from coll_dyn_activem.structure import Positions
from coll_dyn_activem.maths import pycpp, Distribution, JointDistribution,\
    wave_vectors_2D, g2Dto1D, g2Dto1Dgrid, mean_sterr, linspace, logspace,\
    angle, divide_arrays
from coll_dyn_activem.rotors import nu_pdf_th as nu_pdf_th_ABP

# CLASSES

class Displacements(Positions):
    """
    Compute and analyse displacements from simulation data.

    (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#Active%20Brownian%20particles)
    (see https://yketa.github.io/PhD_Wiki/#Active%20Ornstein-Uhlenbeck%20particles)
    """

    def __init__(self, filename, skip=1, corruption=None):
        """
        Loads file.

        Parameters
        ----------
        filename : string
            Name of input data file.
        skip : int
            Skip the `skip' first computed frames in the following calculations.
            (default: 1)
            NOTE: This can be changed at any time by setting self.skip.
            NOTE: This does not apply to .datN files.
        corruption : str or None
            Pass corruption test for given file type (see
            coll_dyn_activem.read.Dat). (default: None)
            NOTE: if corruption == None, then the file has to pass corruption
                  tests.
        """

        super().__init__(filename, skip=skip, corruption=corruption)    # initialise with super class

    def getDisplacements(self, time0, time1, *particle, jump=1, norm=False,
        cage_relative=False, neighbours=None):
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
        cage_relative : bool
            Remove displacement of the centre of mass given by nearest
            neighbours determined by Voronoi tesselation at `time0'.
            (default: False)
        neighbours : coll_dyn_activem.maths.DictList or None
            Neighbour list (see self.getNeighbourList) to use if
            `cage_relative'. (default: None)
            NOTE: if neighbours == None, then it is computed with
                  self.getNeighbourList.

        Returns
        -------
        displacements : [not(norm)] (*, 2) float Numpy array
                        [norm] (*,) float Numpy array
            Displacements between `time0' and `time1'.
        """

        if particle == (): particle = range(self.N)

        if not(cage_relative):
            return super().getDisplacements(
                time0, time1, *particle, jump=jump, norm=norm)

        origDisplacements = super().getDisplacements(
            time0, time1, jump=jump, norm=False)
        if type(neighbours) == type(None):
            neighbours = self.getNeighbourList(time0)

        displacements = np.array(
            itemgetter(*particle)(origDisplacements.copy()))
        for i, index in zip(particle, range(len(particle))):
            if not(i in neighbours): continue   # no neighbours
            nNeighbours = len(neighbours[i])
            for (j, _) in neighbours[i]:
                displacements[index] -= origDisplacements[j]/nNeighbours

        if norm: return np.sqrt(np.sum(displacements**2, axis=-1))
        return displacements

    def nDisplacements(self, dt, int_max=None, jump=1, norm=False,
        cage_relative=False):
        """
        Returns array of displacements with lag time `dt'.

        Parameters
        ----------
        dt : int
            Displacement lag time.
        int_max : int or None
            Maximum number of intervals to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of intervals.
        jump : int
            Period in number of frames at which to check if particles have
            crossed any boundary. (default: 1)
            NOTE: `jump' must be chosen so that particles do not move a distance
                  greater than half the box size during this time.
        norm : bool
            Return norm of displacements rather than 2D displacement.
            (default: False)
        cage_relative : bool
            Remove displacement of the centre of mass given by nearest
            neighbours at initial time. (default: False)
            (see self.getDisplacements)

        Returns
        -------
        displacements : [not(norm)] (*, self.N, 2) float numpy array
                        [norm] (*, self.N) float numpy array
            Array of computed displacements.
        """

        displacements = []
        for time0 in self._time0(dt, int_max=int_max):
            displacements += [
                self.getDisplacements(time0, time0 + dt, jump=jump, norm=norm,
                    cage_relative=cage_relative)]
        displacements = np.array(displacements)

        return displacements

    def displacementsPDF(self, dt, int_max=None, jump=1, cage_relative=False):
        """
        Returns probability density function of displacement norm over lag time
        `dt'.

        Parameters
        ----------
        dt : int
            Displacement lag time.
        int_max : int or None
            Maximum number of intervals to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of intervals.
        jump : int
            Period in number of frames at which to check if particles have
            crossed any boundary. (default: 1)
            NOTE: `jump' must be chosen so that particles do not move a distance
                  greater than half the box size during this time.
        cage_relative : bool
            Remove displacement of the centre of mass given by nearest
            neighbours at initial time. (default: False)
            (see self.getDisplacements)

        Returns
        -------
        axes : numpy array
            Values at which the probability density function is evaluated.
        pdf : float numpy array
            Values of the probability density function.
        """

        return Distribution(
            self.nDisplacements(
                dt, int_max=int_max, jump=jump, norm=True,
                cage_relative=cage_relative)).pdf()

    def displacementsHist(self,
        dt, nBins, int_max=None, jump=1, cage_relative=False,
        vmin=None, vmax=None, log=False, rescaled_to_max=False):
        """
        Returns histogram with `nBins' bins of displacement norm over lag time
        `dt'.

        Parameters
        ----------
        dt : int
            Displacement lag time.
        nBins : int
            Number of bins of the histogram.
        int_max : int or None
            Maximum number of intervals to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of intervals.
        jump : int
            Period in number of frames at which to check if particles have
            crossed any boundary. (default: 1)
            NOTE: `jump' must be chosen so that particles do not move a distance
                  greater than half the box size during this time.
        cage_relative : bool
            Remove displacement of the centre of mass given by nearest
            neighbours at initial time. (default: False)
            (see self.getDisplacements)
        vmin : float
            Minimum value of the bins. (default: minimum computed displacement)
        vmax : float
            Maximum value of the bins. (default: maximum computed displacement)
        log : bool
            Consider the log of the occupancy of the bins. (default: False)
        rescaled_to_max : bool
            Rescale occupancy of the bins by its maximum over bins.
            (default: False)

        Returns
        -------
        bins : float numpy array
            Values of the bins.
        hist : float numpy array
            Occupancy of the bins.
        """

        return Distribution(self.nDisplacements(
            dt, int_max=int_max, jump=jump, norm=True,
            cage_relative=cage_relative)).hist(
                nBins, vmin=vmin, vmax=vmax, log=log,
                rescaled_to_max=rescaled_to_max)

    def dtDisplacements(self, dt, int_max=100, jump=1, norm=False,
        cage_relative=False):
        """
        Returns array of displacements with lag times `dt'.

        Parameters
        ----------
        dt : int array-like
            Displacement lag times.
        int_max : int
            Maximum number of intervals to consider. (default: 100)
        jump : int
            Period in number of frames at which to check if particles have
            crossed any boundary. (default: 1)
            NOTE: `jump' must be chosen so that particles do not move a distance
                  greater than half the box size during this time.
        norm : bool
            Return norm of displacements rather than 2D displacement.
            (default: False)
        cage_relative : bool
            Remove displacement of the centre of mass given by nearest
            neighbours at initial time. (default: False)
            (see self.getDisplacements)

        Returns
        -------
        displacements : [not(norm)] (*, dt.size, self.N, 2) float numpy array
                        [norm] (*, dt.size, self.N) float numpy array
            Array of computed displacements.
        """

        dt = np.array(dt)

        # array of initial times
        if self._type == 'datN':
            time0 = itemgetter(
                *linspace(self.skip, len(self.time0) - 1, int_max,
                    endpoint=True))(
                    self.time0)
        else:
            time0 = np.array(list(OrderedDict.fromkeys(
                np.linspace(self.skip, self.frames - dt.max() - 1, int_max,
                    endpoint=False, dtype=int))))

        if self._type == 'dat':

            displacements = np.empty((time0.size, dt.size, self.N, 2))
            for j in range(dt.size):
                if j > 0:
                    for i in range(time0.size):
                        displacements[i][j] = (         # displacements between time0[i] and time0[i] + dt[j]
                            displacements[i][j - 1]     # displacements between time0[i] and time0[i] + dt[j - 1]
                            + self.getDisplacements(    # displacements between time0[i] + dt[j - 1] and time0[i] + dt[j]
                                time0[i] + dt[j - 1], time0[i] + dt[j],
                                jump=jump, cage_relative=cage_relative))
                else:
                    for i in range(time0.size):
                        displacements[i][0] = self.getDisplacements(    # displacements between time0[i] and time0[i] + dt[0]
                            time0[i], time0[i] + dt[0],
                            jump=jump, cage_relative=cage_relative)

        else:

            displacements = []
            for t0 in time0:
                if cage_relative: neighbours = self.getNeighbourList(t0)
                else: neighbours = None
                displacements += [[
                    self.getDisplacements(t0, t0 + t,
                        cage_relative=cage_relative, neighbours=neighbours)
                    for t in dt]]
            displacements = np.array(displacements)

        if norm: return np.sqrt(np.sum(displacements**2, axis=-1))
        return displacements

    def msd(self, n_max=100, int_max=100, min=None, max=None, jump=1,
        cage_relative=False):
        """
        Compute mean square displacement.

        (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#ABP%20flow%20characteristics)
        (see self._msd)

        Parameters
        ----------
        n_max : int
            Maximum number of lag times at which to compute the displacements.
            (default: 100)
        int_max : int
            Maximum number of different intervals to consider when computed
            displacements for a given lag time. (default: 100)
        min : int or None
            Minimum time at which to compute the displacements. (default: None)
            NOTE: if min == None, then min = 1.
        max : int or None
            Maximum time at which to compute the displacements. (default: None)
            NOTE: if max == None, then max is taken to be the maximum according
                  to the choice of int_max.
        jump : int
            Compute displacements by treating frames in packs of `jump'. This
            can speed up calculation but also give unaccuracies if
            `jump'*self.dt is of the order of the system size. (default: 1)
        cage_relative : bool
            Remove displacement of the centre of mass given by nearest
            neighbours at initial time. (default: False)
            (see self.getDisplacements)

        Returns
        -------
        msd_sterr : (3, *) float numpy array
            Array of:
                (0) lag time at which the mean square displacement is computed,
                (1) mean square displacement,
                (2) standard error of the computed mean square displacement.
        """

        dt, displacements = self._displacements(
            n_max=n_max, int_max=int_max, min=min, max=max, jump=jump,
            cage_relative=cage_relative)

        return self._msd(dt, displacements)

    def _msd(self, dt, displacements):
        """
        Compute mean square displacement.

        (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#ABP%20flow%20characteristics)

        Parameters
        ----------
        dt : (*,) foat numpy array
            Array of lag times.
        displacements : (**, *, N, 2) float numpy array
            Array of displacements. (see self.dtDisplacements)

        Returns
        -------
        msd_sterr : (3, *) float numpy array
            Array of:
                (0) lag time at which the mean square displacement is computed,
                (1) mean square displacement,
                (2) standard error of the computed mean square displacement.
        """

        msd_sterr = []
        for i in range(dt.size):
            disp = displacements[:, i]
            if self.N > 1:
                disp -= disp.mean(axis=1).reshape(displacements.shape[0], 1, 2) # substract mean displacement of particles during each considered intervals
            disp.reshape(displacements.shape[0]*self.N, 2)
            msd_sterr += [[
                dt[i],
                *mean_sterr(np.sum(disp**2, axis=-1).flatten())]]

        return np.array(msd_sterr)

    def msd_th_ABP(self, dt):
        """
        Returns value of theoretical mean squared displacement at lag time `dt'
        for a single ABP.

        (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#One%20ABP)

        Parameters
        ----------
        dt : float
            Lag time at which to evaluate the theoretical mean squared
            displacement.

        Returns
        -------
        msd : float
            Mean squared displacement.
        """

        if self._type == 'dat': # custom relations between parameters
            return msd_th_ABP(self.v0, 1./(3.*self.lp), 1./self.lp, dt)
        else:                   # general parameters
            return msd_th_ABP(self.v0, self.D, self.Dr, dt)

    def msd_th_AOUP(self, dt):
        """
        Returns value of theoretical mean squared displacement at lag time `dt'
        for a single AOUP.

        (see https://yketa.github.io/PhD_Wiki/#One%20AOUP)

        Parameters
        ----------
        dt : float
            Lag time at which to evaluate the theoretical mean squared
            displacement.

        Returns
        -------
        msd : float
            Mean squared displacement.
        """

        return msd_th_AOUP(self.D, self.Dr, dt)

    def overlap(self, n_max=100, int_max=100, min=None, max=None, jump=1,
        a=1, cage_relative=False):
        """
        Compute dynamical overlap function.

        (see https://yketa.github.io/PhD_Wiki/#Flow%20characteristics)
        (see self._overlap)

        Parameters
        ----------
        n_max : int
            Maximum number of lag times at which to compute the displacements.
            (default: 100)
        int_max : int
            Maximum number of different intervals to consider when computed
            displacements for a given lag time. (default: 100)
        min : int or None
            Minimum time at which to compute the displacements. (default: None)
            NOTE: if min == None, then min = 1.
        max : int or None
            Maximum time at which to compute the displacements. (default: None)
            NOTE: if max == None, then max is taken to be the maximum according
                  to the choice of int_max.
        jump : int
            Compute displacements by treating frames in packs of `jump'. This
            can speed up calculation but also give unaccuracies if
            `jump'*self.dt is of the order of the system size. (default: 1)
        a : float
            Parameter of the dynamical overlap function. (default: 1)
        cage_relative : bool
            Remove displacement of the centre of mass given by nearest
            neighbours at initial time. (default: False)
            (see self.getDisplacements)

        Returns
        -------
        Q : (2, *) float numpy array
            Array of:
                (0) lag time at which the dynamical overlap function is
                    computed,
                (1) dynamical overlap function.
        """

        dt, displacements = self._displacements(
            n_max=n_max, int_max=int_max, min=min, max=max, jump=jump,
            cage_relative=cage_relative)

        return self._overlap(dt, displacements, a=a)

    def overlap_relaxation_free_AOUP(q=0.5, a=1):
        """
        Returns relaxation time for a free Ornstein-Uhlenbeck particle, given as
        the time for the dynamical overlap function to decrease below threshold
        `q'.

        (see https://yketa.github.io/PhD_Wiki/#One%20AOUP)

        Parameters
        ----------
        q : float
            Dynamical overlap function threshold. (default: 0.5)
        a : float
            Parameter of the dynamical overlap function. (default: 1)

        Returns
        -------
        t : float
            Relaxation time.
        """

        return overlap_relaxation_free_AOUP(self.D, self.Dr, q=q, a=a)

    def _overlap(self, dt, displacements, a=1):
        """
        Compute dynamical overlap function.

        (see https://yketa.github.io/PhD_Wiki/#Flow%20characteristics)

        Parameters
        ----------
        dt : (*,) foat numpy array
            Array of lag times.
        displacements : (**, *, N, 2) float numpy array
            Array of displacements. (see self.dtDisplacements)
        a : float
            Parameter of the dynamical overlap function. (default: 1)

        Returns
        -------
        Q : (2, *) float numpy array
            Array of:
                (0) lag time at which the dynamical overlap function is
                    computed,
                (1) dynamical overlap function.
        """

        Q = []
        for i in range(dt.size):
            disp = displacements[:, i]
            Q += [[
                dt[i],
                (np.exp(-(disp**2).sum(axis=-1)/(a**2))).mean()]]

        return np.array(Q)

    def susceptibility(self, n_max=100, int_max=100, min=None, max=None, jump=1,
        a=1, cage_relative=False):
        """
        Compute dynamical susceptibility.

        (see https://yketa.github.io/PhD_Wiki/#Flow%20characteristics)
        (see self._susceptibility)

        Parameters
        ----------
        n_max : int
            Maximum number of lag times at which to compute the displacements.
            (default: 100)
        int_max : int
            Maximum number of different intervals to consider when computed
            displacements for a given lag time. (default: 100)
        min : int or None
            Minimum time at which to compute the displacements. (default: None)
            NOTE: if min == None, then min = 1.
        max : int or None
            Maximum time at which to compute the displacements. (default: None)
            NOTE: if max == None, then max is taken to be the maximum according
                  to the choice of int_max.
        jump : int
            Compute displacements by treating frames in packs of `jump'. This
            can speed up calculation but also give unaccuracies if
            `jump'*self.dt is of the order of the system size. (default: 1)
        a : float
            Parameter of the dynamical susceptibility. (default: 1)
        cage_relative : bool
            Remove displacement of the centre of mass given by nearest
            neighbours at initial time. (default: False)
            (see self.getDisplacements)

        Returns
        -------
        chi : (2, *) float numpy array
            Array of:
                (0) lag time at which the dynamical susceptibility is computed,
                (1) dynamical overlap function.
        """

        dt, displacements = self._displacements(
            n_max=n_max, int_max=int_max, min=min, max=max, jump=jump,
            cage_relative=cage_relative)

        return self._susceptiblity(dt, displacements, a=a)

    def _susceptibility(self, dt, displacements, a=1):
        """
        Compute dynamical susceptibility.

        (see https://yketa.github.io/PhD_Wiki/#Flow%20characteristics)

        Parameters
        ----------
        dt : (*,) foat numpy array
            Array of lag times.
        displacements : (**, *, N, 2) float numpy array
            Array of displacements. (see self.dtDisplacements)
        a : float
            Parameter of the dynamical susceptibility. (default: 1)

        Returns
        -------
        chi : (2, *) float numpy array
            Array of:
                (0) lag time at which the dynamical susceptibility is computed,
                (1) dynamical overlap function.
        """

        chi = []
        for i in range(dt.size):
            disp = displacements[:, i]
            chi += [[
                dt[i],
                ((np.exp(-(disp**2).sum(axis=-1)/(a**2)).mean(axis=-1)).var()
                    )*self.N]]

        return np.array(chi)

    def _time0(self, dt, int_max=None):
        """
        Returns array of initial times to evaluate displacements over lag time
        `dt'.

        Parameters
        ----------
        dt : int
            Displacement lag time.
        int_max : int or None
            Maximum number of initial times to return. (default: None)
            NOTE: if int_max == None, a maximum number of them are returned.

        Returns
        -------
        time0 : (*,) float numpy array
            Array of initial times.
        """

        if self._type == 'datN':
            if not(dt in self.deltat):
                raise ValueError("Lag time %i not in file." % dt)
            time0 = self.time0
        else:
            time0 = np.linspace(
                self.skip, self.frames - 1,
                int((self.frames - 1 - self.skip)//dt),
                endpoint=False, dtype=int)
        if int_max == None: return time0
        indexes = list(OrderedDict.fromkeys(
            np.linspace(0, time0.size, int_max, endpoint=False, dtype=int)))
        return np.array(itemgetter(*indexes)(time0), ndmin=1)

    def _displacements(self, n_max=100, int_max=100, min=None, max=None,
        jump=1, cage_relative=False):
        """
        Returns arrays of lag times and displacements evaluated over these lag
        times.

        Parameters
        ----------
        n_max : int
            Maximum number of lag times at which to compute the displacements.
            (default: 100)
        int_max : int
            Maximum number of different intervals to consider when computed
            displacements for a given lag time. (default: 100)
        min : int or None
            Minimum time at which to compute the displacements. (default: None)
            NOTE: if min == None, then min = 1.
        max : int or None
            Maximum time at which to compute the displacements. (default: None)
            NOTE: if max == None, then max is taken to be the maximum according
                  to the choice of int_max.
        jump : int
            Compute displacements by treating frames in packs of `jump'. This
            can speed up calculation but also give unaccuracies if
            `jump'*self.dt is of the order of the system size. (default: 1)
        cage_relative : bool
            Remove displacement of the centre of mass given by nearest
            neighbours at initial time. (default: False)
            (see self.getDisplacements)

        Returns
        -------
        dt : (*,) foat numpy array
            Array of lag times.
        displacements : (**, *, N, 2) float numpy array
            Array of displacements. (see self.dtDisplacements)
        """

        min = 1 if min == None else int(min)
        if self._type == 'datN':
            max = self.deltat.max() if max == None else int(max)
        else:
            max = ((self.frames - self.skip - 1)//int_max if max == None
                else int(max))

        # COMPUTE RELEVANT DISPLACEMENTS FROM DATA

        # array of lag times
        if self._type == 'datN':
            dt = self.deltat[(self.deltat >= min)*(self.deltat <= max)]
            dt = itemgetter(*linspace(0, len(dt) - 1, n_max, endpoint=True))(dt)
        else:
            dt = logspace(min, max, n_max)

        displacements = self.dtDisplacements(dt,    # array of displacements for different initial times and lag times
            int_max=int_max, jump=jump, norm=False, cage_relative=cage_relative)

        return np.array(dt), displacements

class Velocities(Dat):
    """
    Compute and analyse velocities from simulation data.

    (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#Active%20Brownian%20particles)
    (see https://yketa.github.io/PhD_Wiki/#Active%20Ornstein-Uhlenbeck%20particles)
    """

    def __init__(self, filename, skip=1, corruption=None):
        """
        Loads file.

        Parameters
        ----------
        filename : string
            Name of input data file.
        skip : int
            Skip the `skip' first computed frames in the following calculations.
            (default: 1)
            NOTE: This can be changed at any time by setting self.skip.
            NOTE: This does not apply to .datN files.
        corruption : str or None
            Pass corruption test for given file type (see
            coll_dyn_activem.read.Dat). (default: None)
            NOTE: if corruption == None, then the file has to pass corruption
                  tests.
        """

        super().__init__(filename, loadWork=False, corruption=corruption)   # initialise with super class

        self.skip = skip    # skip the `skip' first frames in the analysis

    def nVelocities(self, int_max=None, norm=False):
        """
        Returns array of velocities.

        Parameters
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.
        norm : bool
            Return norm of velocities rather than 2D velocities.
            (default: False)

        Returns
        -------
        velocities : [not(norm)] (*, self.N, 2) float numpy array
                     [norm] (*, self.N) float numpy array
            Array of computed velocities.
        """

        return np.array(list(map(
            lambda time0: self.getVelocities(time0, norm=norm),
            self._time0(int_max=int_max))))

    def velocitiesPDF(self, int_max=None):
        """
        Returns probability density function of velocity norm.

        PARAMETERS
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.

        Returns
        -------
        axes : numpy array
            Values at which the probability density function is evaluated.
        pdf : float numpy array
            Values of the probability density function.
        """

        return Distribution(self.nVelocities(int_max=int_max, norm=True)).pdf()

    def velocitiesHist(self, nBins, int_max=None, vmin=None, vmax=None,
        log=False, rescaled_to_max=False):
        """
        Returns histogram with `nBins' bins of velocity norm.

        Parameters
        ----------
        nBins : int
            Number of bins of the histogram.
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.
        vmin : float
            Minimum value of the bins. (default: minimum computed velocity)
        vmax : float
            Maximum value of the bins. (default: maximum computed velocity)
        log : bool
            Consider the log of the occupancy of the bins. (default: False)
        rescaled_to_max : bool
            Rescale occupancy of the bins by its maximum over bins.
            (default: False)

        Returns
        -------
        bins : float numpy array
            Values of the bins.
        hist : float numpy array
            Occupancy of the bins.
        """

        return Distribution(self.nVelocities(int_max=int_max, norm=True)).hist(
            nBins, vmin=vmin, vmax=vmax, log=log,
            rescaled_to_max=rescaled_to_max)

    def energySpectrum(self, int_max=None, nBoxes=None):
        """
        Returns kinetic energy spectrum.

        (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#ABP%20flow%20characteristics)

        Parameters
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.
        nBoxes : int
            Number of boxes in each direction to compute the grid of velocities
            which FFT will be computed. (default: None)
            NOTE: If nBoxes == None, then nBoxes = int(sqrt(self.N)).

        Returns
        -------
        E : (*, 2) float Numpy array
            Array of (k, E(k)).
        """

        if nBoxes == None: nBoxes = int(np.sqrt(self.N))
        nBoxes = int(nBoxes)

        wave_vectors = np.sqrt(np.sum(
            wave_vectors_2D(nBoxes, nBoxes, self.L/nBoxes)**2, axis=-1))    # grid of wave vector norms

        FFTsq = []    # list of squared velocity FFT
        for time, velocity in zip(
            self._time0(int_max=int_max),
            self.nVelocities(int_max=int_max, norm=True)):

            velocitiesFFT = np.fft.fft2(    # FFT of displacement grid
                self.toGrid(time, velocity, nBoxes=nBoxes),
                axes=(0, 1))
            FFTsq += [np.real(np.conj(velocitiesFFT)*velocitiesFFT)]

        return g2Dto1Dgrid(
            wave_vectors*np.mean(FFTsq, axis=0), wave_vectors,
            average_grid=False)

    def velocitiesCor(self, nBins, int_max=None, min=None, max=None,
        rescale_pair_distribution=False):
        """
        Compute spatial correlations of particles' velocities (and density).

        (see self.getRadialCorrelations)

        Parameters
        ----------
        nBins : int
            Number of intervals of distances on which to compute the
            correlations.
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.
        min : float or None
            Minimum distance (included) at which to compute the correlations.
            (default: None)
            NOTE: if min == None then min = 0.
        max : float or None
            Maximum distance (excluded) at which to compute the correlations.
            (default: None)
            NOTE: if max == None then max = self.L/2.
        rescale_pair_distribution : bool
            Rescale correlations by pair distribution function. (default: False)

        Returns
        -------
        Cvv : (*, 2) float Numpy array
            Array of (r, Cvv(r), errCvv(r)) with Cvv(r) the cylindrically
            averaged spatial correlations of velocity and errCvv(r) the standard
            error on this measure.
        """

        correlations = np.array([
            (lambda v:
                self.getRadialCorrelations(
                    t, v/np.sqrt((v**2).sum(axis=-1).mean()),
                    nBins, min=min, max=max,
                    rescale_pair_distribution=rescale_pair_distribution))(
                self.getVelocities(t, norm=False))
            for t in self._time0(int_max=int_max)])

        return np.array([
            [correlations[0, bin, 0], *mean_sterr(correlations[:, bin, 1])]
            for bin in range(nBins)])

    def velocitiesOriCor(self, int_max=None):
        """
        Compute radial correlations of particles' velocity orientation (see
        https://yketa.github.io/PhD_Wiki/#Flow%20characteristics).

        Parameters
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.

        Returns
        -------
        Cvv : (*, 3) float Numpy array
            Array of (r, Cvv(r), errCvv(r)) with Cvv(r) the cylindrically
            averaged spatial correlations of velocity orientation, and
            errCvv(r) the standard error on this quantity.
        """

        correlations = np.array([
            pycpp.getVelocitiesOriCor(
                self.getPositions(t), self.L, self.getVelocities(t, norm=False),
                sigma=self.diameters.mean())
            for t in self._time0(int_max=int_max)])

        return np.array([
            [correlations[0, bin, 0], *mean_sterr(correlations[:, bin, 1])]
            for bin in range(correlations.shape[1])])

    def _time0(self, int_max=None):
        """
        Returns array of frames at which to compute velocities.

        Parameters
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.

        Returns
        -------
        time0 : (*,) float numpy array
            Array of frames.
        """

        if self._type == 'datN':
            time0 = self.time0[self.time0 != self.frameIndices[-1]]
        else:
            time0 = np.array(range(self.skip, self.frames - 1))
        # NOTE: It is important to remove the last frame since the velocities are 0.

        if int_max == None: return time0
        indexes = linspace(0, time0.size, int_max, endpoint=False)
        return np.array(itemgetter(*indexes)(time0), ndmin=1)

class Orientations(Dat):
    """
    Compute and analyse orientations from simulation data.

    (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#Active%20Brownian%20particles)
    (see https://yketa.github.io/PhD_Wiki/#Active%20Ornstein-Uhlenbeck%20particles)
    """

    def __init__(self, filename, skip=1, corruption=None):
        """
        Loads file.

        Parameters
        ----------
        filename : string
            Name of input data file.
        skip : int
            Skip the `skip' first computed frames in the following calculations.
            (default: 1)
            NOTE: This can be changed at any time by setting self.skip.
            NOTE: This does not apply to .datN files.
        corruption : str or None
            Pass corruption test for given file type (see
            coll_dyn_activem.read.Dat). (default: None)
            NOTE: if corruption == None, then the file has to pass corruption
                  tests.
        """

        super().__init__(filename, loadWork=False, corruption=corruption)   # initialise with super class

        self.skip = skip    # skip the `skip' first frames in the analysis

    def nOrientations(self, int_max=None):
        """
        Returns array of orientations.

        Parameters
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.

        Returns
        -------
        orientations : (*, self.N) float numpy array
            Array of computed orientations.
        """

        return np.array(list(map(
            lambda time0: self.getOrientations(time0),
            self._time0(int_max=int_max))))

    def nDirections(self, int_max=None):
        """
        Returns array of directions.

        Parameters
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.

        Returns
        -------
        directions : (*, self.N, 2) float numpy array
            Array of computed directions.
        """

        return np.array(list(map(
            lambda time0: self.getDirections(time0),
            self._time0(int_max=int_max))))

    def nOrder(self, int_max=None, norm=False):
        """
        Returns array of order parameter.

        Parameters
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.
        norm : bool
            Return norm of order parameter rather than 2D order parameter.
            (default: False)

        Returns
        -------
        order : [not(norm)] (*, 2) float numpy array
                     [norm] (*,) float numpy array
            Array of computed order parameters.
        """

        return np.array(list(map(
            lambda time0: self.getOrderParameter(time0, norm=norm),
            self._time0(int_max=int_max))))

    def orderPDF(self, int_max=None):
        """
        Returns probability density function of order parameter norm.

        PARAMETERS
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.

        Returns
        -------
        axes : numpy array
            Values at which the probability density function is evaluated.
        pdf : float numpy array
            Values of the probability density function.
        """

        return Distribution(self.nOrder(int_max=int_max, norm=True)).pdf()

    def orderHist(self, nBins, int_max=None, vmin=None, vmax=None,
        log=False, rescaled_to_max=False):
        """
        Returns histogram with `nBins' bins of order parameter norm.

        Parameters
        ----------
        nBins : int
            Number of bins of the histogram.
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.
        vmin : float
            Minimum value of the bins.
            (default: minimum computed order parameter)
        vmax : float
            Maximum value of the bins.
            (default: maximum computed order parameter)
        log : bool
            Consider the log of the occupancy of the bins. (default: False)
        rescaled_to_max : bool
            Rescale occupancy of the bins by its maximum over bins.
            (default: False)

        Returns
        -------
        bins : float numpy array
            Values of the bins.
        hist : float numpy array
            Occupancy of the bins.
        """

        return Distribution(self.nOrder(int_max=int_max, norm=True)).hist(
            nBins, vmin=vmin, vmax=vmax, log=log,
            rescaled_to_max=rescaled_to_max)

    def orientationsCor(self, nBins, int_max=None, min=None, max=None,
        rescale_pair_distribution=False):
        """
        Compute spatial correlations of particles' orientations (and density).

        (see self.getRadialCorrelations)

        Parameters
        ----------
        nBins : int
            Number of intervals of distances on which to compute the
            correlations.
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.
        min : float or None
            Minimum distance (included) at which to compute the correlations.
            (default: None)
            NOTE: if min == None then min = 0.
        max : float or None
            Maximum distance (excluded) at which to compute the correlations.
            (default: None)
            NOTE: if max == None then max = self.L/2.

        Returns
        -------
        Cuu : (*, 2) float Numpy array
            Array of (r, Cuu(r), errCuu(r)) with Cuu(r) the cylindrically
            averaged spatial correlations of orientation and errCuu(r) the
            standard error on this measure.
        rescale_pair_distribution : bool
            Rescale correlations by pair distribution function. (default: False)
        """

        correlations = np.array([
            self.getRadialCorrelations(t, self.getDirections(t),
                nBins, min=min, max=max,
                rescale_pair_distribution=rescale_pair_distribution)
            for t in self._time0(int_max=int_max)])

        return np.array([
            [correlations[0, bin, 0], *mean_sterr(correlations[:, bin, 1])]
            for bin in range(nBins)])

    def nu_pdf_th_ABP(self, *nu):
        """
        Returns value of theoretical probability density function of the order
        parameter norm for ABPs.

        (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#N-interacting%20Brownian%20rotors)

        Parameters
        ----------
        nu : float
            Values of the order parameter norm at which to evaluate the
            probability density function.

        Returns
        -------
        pdf : (*,) float numpy array
            Probability density function.
        """

        return nu_pdf_th_ABP(self.N, self.g, 1/self.lp, *nu)

    def _time0(self, int_max=None):
        """
        Returns array of frames at which to compute orientations.

        Parameters
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.

        Returns
        -------
        time0 : (*,) float numpy array
            Array of frames.
        """

        if self._type == 'datN':
            time0 = self.time0
        else:
            time0 = np.array(range(self.skip, self.frames))

        if int_max == None: return time0
        indexes = linspace(0, time0.size, int_max, endpoint=False)
        return np.array(itemgetter(*indexes)(time0), ndmin=1)

class Propulsions(Dat):
    """
    Compute and analyse self-propulsion vectors from simulation data.

    (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#Active%20Brownian%20particles)
    (see https://yketa.github.io/PhD_Wiki/#Active%20Ornstein-Uhlenbeck%20particles)
    """

    def __init__(self, filename, skip=1, corruption=None):
        """
        Loads file.

        Parameters
        ----------
        filename : string
            Name of input data file.
        skip : int
            Skip the `skip' first computed frames in the following calculations.
            (default: 1)
            NOTE: This can be changed at any time by setting self.skip.
            NOTE: This does not apply to .datN files.
        corruption : str or None
            Pass corruption test for given file type (see
            coll_dyn_activem.read.Dat). (default: None)
            NOTE: if corruption == None, then the file has to pass corruption
                  tests.
        """

        super().__init__(filename, loadWork=False, corruption=corruption)   # initialise with super class

        self.skip = skip    # skip the `skip' first frames in the analysis

    def nPropulsions(self, int_max=None, norm=False):
        """
        Returns array of self-propulsion vectors.

        Parameters
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.
        norm : bool
            Consider norm of self-propulsion vectors rather than 2D
            self-propulsion vectors. (default: False)

        Returns
        -------
        propulsions : [not(norm)] (*, self.N, 2) float numpy array
                      [norm] (*, self.N) float numpy array
            Array of computed self-propulsion vectors.
        """

        return np.array(list(map(
            lambda time0: self.getPropulsions(time0, norm=norm),
            self._time0(int_max=int_max))))

    def propulsionsPDF(self, int_max=None, norm=False, axis='x'):
        """
        Returns probability density function of self-propulsion vector.

        PARAMETERS
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.
        norm : bool
            Consider norm of self-propulsion vectors rather than 2D
            self-propulsion vectors. (default: False)
        axis : str ('x', 'y', 'both')
            Consider either x-axis, y-axis (1D PDF), or both axes (2D PDF).
            (default: 'x')
            NOTE: This is considered only if norm == False.

        Returns
        -------
        [norm or (not(norm) and (axis == 'x' or axis == 'y'))]
        axes : numpy array
            Values at which the probability density function is evaluated.
        pdf : float numpy array
            Values of the probability density function.
        [not(norm) and axis == 'both']
        pdf3D : (*, 3) float Numpy array
            (0) Value of the x-coordinate at which the PDF is evaluated.
            (1) Value of the y-coordinate at which the PDF is evaluated.
            (2) PDF.
        """

        if norm:
            return Distribution(
                self.nPropulsions(int_max=int_max, norm=True)).pdf()
        else:
            propulsions = self.nPropulsions(int_max=int_max, norm=False)
            if axis == 'both':
                return JointDistribution(
                    propulsions[:, :, 0], propulsions[:, :, 1]).pdf()
            elif axis == 'x':
                return Distribution(propulsions[:, :, 0]).pdf()
            elif axis == 'y':
                return Distribution(propulsions[:, :, 1]).pdf()
            else:
                raise ValueError("Axis '%s' not in ('x', 'y', 'both').")

    def propulsionsHist(self, nBins, int_max=None, vmin=None, vmax=None,
        log=False, rescaled_to_max=False, norm=False, axis='x'):
        """
        Returns histogram with `nBins' bins of self-propulsion vectors.

        Parameters
        ----------
        nBins : int
            Number of bins of the histogram.
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.
        vmin : float
            Minimum value of the bins.
            (default: minimum computed self-propulsion vector norm or
            coordinate)
        vmax : float
            Maximum value of the bins.
            (default: maximum computed self-propulsion vector norm or
            coordinate)
        log : bool
            Consider the log of the occupancy of the bins. (default: False)
            NOTE: This will be considered only if norm or (not(norm) and
                  (axis == 'x' or axis == 'y')).
        rescaled_to_max : bool
            Rescale occupancy of the bins by its maximum over bins.
            (default: False)
            NOTE: This will be considered only if norm or (not(norm) and
                  (axis == 'x' or axis == 'y')).
        norm : bool
            Consider norm of self-propulsion vectors rather than 2D
            self-propulsion vectors. (default: False)
        axis : str ('x', 'y', 'both')
            Consider either x-axis, y-axis (1D histogram), or both axes
            (2D histogram). (default: 'x')
            NOTE: This is considered only if norm == False.

        Returns
        -------
        [norm or (not(norm) and (axis == 'x' or axis == 'y'))]
        bins : float numpy array
            Values of the bins.
        hist : float numpy array
            Occupancy of the bins.
        [not(norm) and axis == 'both']
        hist : (nBins**2, 3) float Numpy array
            Values of the histogram:
                (0) Bin value of the x-coordinate.
                (1) Bin value of the y-coordinate.
                (2) Proportion.
        """

        if norm:
            return Distribution(
                self.nPropulsions(int_max=int_max, norm=True)).hist(
                    nBins, vmin=vmin, vmax=vmax, log=log,
                    rescaled_to_max=rescaled_to_max)
        else:
            propulsions = self.nPropulsions(int_max=int_max, norm=False)
            if axis == 'both':
                return JointDistribution(
                    propulsions[:, :, 0], propulsions[:, :, 1]).hist(
                        nBins, vmin1=vmin, vmax1=vmax, vmin2=vmin, vmax2=vmax)
            elif axis == 'x':
                return Distribution(propulsions[:, :, 0]).hist(
                        nBins, vmin=vmin, vmax=vmax, log=log,
                        rescaled_to_max=rescaled_to_max)
            elif axis == 'y':
                return Distribution(propulsions[:, :, 1]).hist(
                        nBins, vmin=vmin, vmax=vmax, log=log,
                        rescaled_to_max=rescaled_to_max)

    def propulsions_dist_th_AOUP(self, *p):
        """
        Returns value of theoretical self-propulsion vector distribution for
        AOUPs.

        (see https://yketa.github.io/PhD_Wiki/#Active%20Ornstein-Uhlenbeck%20particles)

        Parameters
        ----------
        p : float 2-uple
            Value of the self-propulsion vector.

        Returns
        -------
        Pp : (*,) float numpy array
            Probability density function.
        """

        # return np.array(list(map(
        #     lambda _: np.exp(-(_[0]**2 + _[1]**2)/(2.*self.D*self.Dr))/(
        #         2*np.pi*self.D*self.Dr),
        #     p)))
        return np.array(list(map(
            lambda _: np.prod(list(map(
                lambda __: norm_gen.pdf(__,
                    loc=0, scale=np.sqrt(self.D*self.Dr)),
                _))),
            p)))

    def propulsionsCor(self, n_max=100, int_max=100, min=None, max=None):
        """
        Compute time correlations of particles' self-propulsion vectors.

        Parameters
        ----------
        n_max : int
            Maximum number of times at which to compute the self-propulsion
            vectors. (default: 100)
        int_max : int or None
            Maximum number of frames to consider. (default: 100)
        min : int or None
            Minimum time at which to compute the correlation. (default: None)
            NOTE: if min == None, then min = 1.
        max : int or None
            Maximum time at which to compute the correlation. (default: None)
            NOTE: if max == None, then max is taken to be the maximum according
                  to the choice of int_max.

        Returns
        -------
        Cpp : (*, 2) float Numpy array
            Array of (t, Cpp(t)) with Cpp(t) the correlation function.
        """

        Cpp = []

        min = 1 if min == None else int(min)
        max = ((self.frames - self.skip - 1)//int_max if max == None
            else int(max))
        _dt = logspace(min, max, n_max) # array of lag times

        for dt in _dt:

            time0 = np.linspace(
                self.skip, self.frames - 1,
                int((self.frames - 1 - self.skip)//dt),
                endpoint=False, dtype=int)
            if int_max != None:
                indexes = list(OrderedDict.fromkeys(
                    np.linspace(0, time0.size, int_max,
                        endpoint=False, dtype=int)))
                time0 = np.array(itemgetter(*indexes)(time0), ndmin=1)

            Cpp += [[dt,
                np.mean(list(map(
                    lambda t: list(map(
                        lambda x, y: np.dot(x, y), *(
                            self.getPropulsions(t, norm=False),
                            self.getPropulsions(int(t + dt), norm=False)))),
                    time0)))]]

        return np.array(Cpp)

    def propulsions_cor_th_AOUP(self, *dt):
        """
        Returns value of theoretical self-propulsion vector time correlation for
        AOUPs.

        (see https://yketa.github.io/PhD_Wiki/#Active%20Ornstein-Uhlenbeck%20particles)

        Parameters
        ----------
        dt : float
            Lag time.

        Returns
        -------
        cpp : (*,) float numpy array
            Time correlation.
        """

        return np.array(list(map(
            lambda t: 2*self.D*self.Dr*np.exp(-self.Dr*t),
            dt)))

    def _time0(self, int_max=None):
        """
        Returns array of frames at which to compute self-propulsion vectors.

        Parameters
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.

        Returns
        -------
        time0 : (*,) float numpy array
            Array of frames.
        """

        if self._type == 'datN':
            time0 = self.time0
        else:
            time0 = np.array(range(self.skip, self.frames))

        if int_max == None: return time0
        indexes = linspace(0, time0.size, int_max, endpoint=False)
        return np.array(itemgetter(*indexes)(time0), ndmin=1)

# FUNCTIONS

def msd_th_ABP(v0, D, Dr, dt):
    """
    Returns value of theoretical mean squared displacement at lag time `dt'
    for a single ABP.

    (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#One%20ABP)

    Parameters
    ----------
    v0 : float
        Self-propulsion velocity.
    D : float
        Translational diffusivity.
    Dr : float
        Rotational diffusivity.
    dt : float
        Lag time at which to evaluate the theoretical mean squared
        displacement.

    Returns
    -------
    msd : float
        Mean squared displacement.
    """

    return 4*D*dt + (2*(v0**2)/Dr)*(dt + (np.exp(-Dr*dt) - 1)/Dr)

def msd_th_AOUP(D, Dr, dt):
    """
    Returns value of theoretical mean squared displacement at lag time `dt'
    for a single AOUP.

    (see https://yketa.github.io/PhD_Wiki/#One%20AOUP)

    Parameters
    ----------
    D : float
        Translational diffusivity.
    Dr : float
        Rotational diffusivity.
    dt : float
        Lag time at which to evaluate the theoretical mean squared
        displacement.

    Returns
    -------
    msd : float
        Mean squared displacement.
    """

    return 4*D*(dt + (np.exp(-Dr*dt) - 1)/Dr)

def overlap_free_AOUP(D, Dr, dt, a=1):
    """
    Returns value of the dynamical overlap funtionat lag time `dt' for a free
    Ornstein-Uhlenbeck particle.

    (see https://yketa.github.io/PhD_Wiki/#One%20AOUP)

    Parameters
    ----------
    D : float
        Translational diffusivity.
    Dr : float
        Rotational diffusivity.
    dt : float
        Lag time at which to evaluate the dynamical overlap function.
    a : float
        Parameter of the dynamical overlap function. (default: 1)

    Returns
    -------
    Q : float
        Dynamical overlap.
    """

    return 1./(1. + msd_th_AOUP(D, Dr, dt)/(a**2))

def overlap_relaxation_free_AOUP(D, Dr, q=0.5, a=1):
    """
    Returns relaxation time for a free Ornstein-Uhlenbeck particle, given as the
    time for the dynamical overlap function to decrease below threshold `q'.

    (see https://yketa.github.io/PhD_Wiki/#One%20AOUP)

    Parameters
    ----------
    D : float
        Translational diffusivity.
    Dr : float
        Rotational diffusivity.
    q : float
        Dynamical overlap function threshold. (default: 0.5)
    a : float
        Parameter of the dynamical overlap function. (default: 1)

    Returns
    -------
    t : float
        Relaxation time.
    """

    return (
        ((a**2)*Dr*(1 - q)
            + 4*q*D*(1 + lambertw(-np.exp(-1 + (a**2)*(q - 1)*Dr/(4*q*D)))))
        /(4*q*D*Dr)).real

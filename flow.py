"""
Module flow provides classes to compute and analyse displacements, velocities,
and orientations in order to characterise the flow of systems of active
particles.

(see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#ABP%20flow%20characteristics)
"""

import numpy as np
from scipy.stats import norm as norm_gen
from scipy.special import lambertw
from collections import OrderedDict
from operator import itemgetter
import struct

from coll_dyn_activem.read import Dat
from coll_dyn_activem.structure import Positions
from coll_dyn_activem.maths import pycpp, Distribution, JointDistribution,\
    mean_sterr, linspace, logspace, angle, divide_arrays, wo_mean,\
    wave_vectors_dq, normalise1D
from coll_dyn_activem.rotors import nu_pdf_th as nu_pdf_th_ABP
from coll_dyn_activem._pycpp import getBondsBrokenBonds

# CLASSES

class Displacements(Positions):
    """
    Compute and analyse displacements from simulation data.

    (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#Active%20Brownian%20particles)
    """

    def getDisplacements(self, time0, time1, *particle, jump=1, norm=False,
        remove_cm=False, cage_relative=False, neighbours=None):
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
        remove_cm : bool
            Remove centre of mass displacement. (default: False)
            NOTE: does not affect result if self.N == 1.
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
                time0, time1, *particle, jump=jump, norm=norm,
                remove_cm=remove_cm)

        origDisplacements = super().getDisplacements(
            time0, time1, jump=jump, norm=False, remove_cm=remove_cm)
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
        remove_cm=False, cage_relative=False):
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
            crossed any boundary. (default: 1) (see self.getDisplacements)
        norm : bool
            Return norm of displacements rather than 2D displacement.
            (default: False)
        remove_cm : bool
            Remove centre of mass displacement. (default: False)
            NOTE: does not affect result if self.N == 1.
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
                    remove_cm=remove_cm, cage_relative=cage_relative)]
        displacements = np.array(displacements)

        return displacements

    def displacementsPDF(self, dt, int_max=None, jump=1,
        remove_cm=False, cage_relative=False):
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
            crossed any boundary. (default: 1) (see self.getDisplacements)
        remove_cm : bool
            Remove centre of mass displacement. (default: False)
            NOTE: does not affect result if self.N == 1.
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
                remove_cm=remove_cm, cage_relative=cage_relative)).pdf()

    def displacementsHist(self,
        dt, nBins, int_max=None, jump=1, remove_cm=False, cage_relative=False,
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
            crossed any boundary. (default: 1) (see self.getDisplacements)
        remove_cm : bool
            Remove centre of mass displacement. (default: False)
            NOTE: does not affect result if self.N == 1.
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
            remove_cm=remove_cm, cage_relative=cage_relative)).hist(
                nBins, vmin=vmin, vmax=vmax, log=log,
                rescaled_to_max=rescaled_to_max)

    def dtDisplacements(self, dt, int_max=100, jump=1, norm=False,
        remove_cm=False, cage_relative=False, initial_times=False):
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
            crossed any boundary. (default: 1) (see self.getDisplacements)
        norm : bool
            Return norm of displacements rather than 2D displacement.
            (default: False)
        remove_cm : bool
            Remove centre of mass displacement. (default: False)
            NOTE: does not affect result if self.N == 1.
        cage_relative : bool
            Remove displacement of the centre of mass given by nearest
            neighbours at initial time. (default: False)
            (see self.getDisplacements)
        initial_times : bool
            Return initial times at which displacements are computed.
            (default: False)

        Returns
        -------
        displacements : [not(norm)] (*, dt.size, self.N, 2) float numpy array
                        [norm] (*, dt.size, self.N) float numpy array
            Array of computed displacements.
        time0 : [initial_times] (*,) int numpy array
            Initial times at which displacements are computed.
        """

        dt = np.array(dt)

        # array of initial times
        if self._type == 'datN':
            time0 = self.time0 if self.time0.size == 1 else np.array(itemgetter(
                *linspace(self.skip, len(self.time0) - 1, int_max,
                    endpoint=True))(
                    self.time0),
                ndmin=1)
        else:
            time0 = np.array(list(OrderedDict.fromkeys(
                np.linspace(self.skip, self.frames - dt.max() - 1, int_max,
                    endpoint=False, dtype=int))),
                ndmin=1)

        if self._type == 'dat':

            displacements = np.empty((time0.size, dt.size, self.N, 2))
            for j in range(dt.size):
                if j > 0:
                    for i in range(time0.size):
                        displacements[i][j] = (         # displacements between time0[i] and time0[i] + dt[j]
                            displacements[i][j - 1]     # displacements between time0[i] and time0[i] + dt[j - 1]
                            + self.getDisplacements(    # displacements between time0[i] + dt[j - 1] and time0[i] + dt[j]
                                time0[i] + dt[j - 1], time0[i] + dt[j],
                                jump=jump,
                                remove_cm=remove_cm,
                                cage_relative=cage_relative))
                else:
                    for i in range(time0.size):
                        displacements[i][0] = self.getDisplacements(    # displacements between time0[i] and time0[i] + dt[0]
                            time0[i], time0[i] + dt[0],
                            jump=jump,
                            remove_cm=remove_cm,
                            cage_relative=cage_relative)

        else:

            if cage_relative:
                neighbours = list(map(
                    lambda t0: self.getNeighbourList(t0),
                    time0))
            else: neighbours = np.full(time0.shape, fill_value=None)

            displacements = np.array(list(map(
                lambda t0: list(map(
                    lambda t: self.getDisplacements(t0, t0 + t,
                        remove_cm=remove_cm,
                        cage_relative=cage_relative,
                        neighbours=neighbours[time0.tolist().index(t0)]),
                    dt)),
                time0)))

        if norm: return np.sqrt(np.sum(displacements**2, axis=-1))
        if initial_times: return displacements, time0
        return displacements

    def msd(self, n_max=100, int_max=100, min=None, max=None, jump=1,
        cage_relative=False, dtDisplacements=None):
        """
        Compute mean square displacement.

        (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#ABP%20flow%20characteristics)

        Parameters
        ----------
        n_max : int
            Maximum number of lag times at which to compute the displacements.
            (default: 100)
            NOTE: This is overridden if dtDisplacements != None.
        int_max : int
            Maximum number of different intervals to consider when computing
            displacements for a given lag time. (default: 100)
            NOTE: This is overridden if dtDisplacements != None.
        min : int or None
            Minimum lag time at which to compute the displacements.
            (default: None)
            NOTE: if min == None, then min = 1.
            NOTE: This is overridden if dtDisplacements != None.
        max : int or None
            Maximum lag time at which to compute the displacements.
            (default: None)
            NOTE: if max == None, then max is taken to be the maximum according
                  to the choice of int_max.
            NOTE: This is overridden if dtDisplacements != None.
        jump : int
            Period in number of frames at which to check if particles have
            crossed any boundary. (default: 1) (see self.getDisplacements)
            NOTE: This is overridden if dtDisplacements != None.
        cage_relative : bool
            Remove displacement of the centre of mass given by nearest
            neighbours at initial time. (default: False)
            (see self.getDisplacements)
            NOTE: This is overridden if dtDisplacements != None.
        dtDisplacements : ((*,) int array-like,
                          (**, *, self.N, 2) float array-like) or None
            Lag time and displacements at these lag times from which to compute
            quantity.
            NOTE: if dtDisplacements == None, then compute with
                  self._displacements.

        Returns
        -------
        msd_stderr_chi : (*, 4) float numpy array
            Array of:
                (0) lag time,
                (1) mean square displacement,
                (2) standard error on the computed mean square displacement,
                (3) susceptibility of the computed mean square displacement.
            (see self._mean_stderr_chi)
        """

        if type(dtDisplacements) == type(None):
            dt, displacements = self._displacements(
                n_max=n_max, int_max=int_max, min=min, max=max, jump=jump,
                cage_relative=cage_relative,
                initial_times=False)
        else:
            dt, displacements = dtDisplacements

        quantities = (wo_mean(displacements, axis=-2)**2).sum(axis=-1)
        return self._mean_stderr_chi(dt, quantities)

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

    def displacementsCor(self, dt, nBins, int_max=100, min=None, max=None,
        jump=1, transformation=(lambda disp: disp), remove_cm=False,
        rescale_pair_distribution=False):
        """
        Compute radial correlations of particles' displacements.

        Parameters
        ----------
        dt : int
            Lag time at which to compute displacements.
        nBins : int
            Number of intervals of distances on which to compute the
            correlations.
        int_max : int
            Maximum number of different intervals to consider when computing
            displacements for a given lag time. (default: 100)
        min : float or None
            Minimum distance (included) at which to compute the correlations.
            (default: None)
            NOTE: if min == None then min = 0.
        max : float or None
            Maximum distance (excluded) at which to compute the correlations.
            (default: None)
            NOTE: if max == None then max = self.L/2.
        jump : int
            Period in number of frames at which to check if particles have
            crossed any boundary. (default: 1) (see self.getDisplacements)
        transformation : function of numpy array to numpy array
            Transformation to apply on individual particles' displacement
            ((2,) float numpy array) before computing the correlations.
            (default: (lambda disp: disp))
            NOTE: transformed displacements are then divided by their root mean
                  square.
        remove_cm : bool
            Remove centre of mass displacement. (default: False)
            NOTE: does not affect result if self.N == 1.
        rescale_pair_distribution : bool
            Rescale correlations by pair distribution function. (default: False)

        Returns
        -------
        Cuu : (*, 3) float Numpy array
            Array of (r, Cuu(r), errCuu(r)) with Cuu(r) the cylindrically
            averaged spatial correlations of displacement and errCuu(r) the
            standard error on this measure.
        zeta : (2,) float Numpy array
            Cooperativity and standard error on this measure.
        """

        time0 = self._time0(dt, int_max=int_max)
        displacements = np.array(list(map(
            lambda t: (lambda d: d/np.sqrt((d**2).sum(axis=-1).mean()))(
                np.array(list(map(
                    lambda disp: np.array(transformation(disp), ndmin=1),
                    self.getDisplacements(t, t + dt,
                        jump=jump, norm=False, remove_cm=remove_cm,
                        cage_relative=False, neighbours=None))))),
            time0)))
        corZeta = list(map(
            lambda t, d: self.getRadialCorrelations(
                t, d, nBins, min=min, max=max,
                rescale_pair_distribution=rescale_pair_distribution),
            *(time0, displacements)))
        correlations = np.array([corZeta[t][0] for t in range(len(corZeta))])
        zeta = np.array([corZeta[t][1] for t in range(len(corZeta))])

        return (
            np.array([
                [correlations[0, bin, 0],
                    *mean_sterr(correlations[:, bin, 1])]
                for bin in range(nBins)]),
            mean_sterr(zeta))

    def overlap(self, a=1, n_max=100, int_max=100, min=None, max=None, jump=1,
        cage_relative=False, dtDisplacements=None, heaviside=False):
        """
        Compute dynamical overlap function.

        Parameters
        ----------
        a : float
            Parameter of the dynamical overlap function. (default: 1)
        n_max : int
            Maximum number of lag times at which to compute the displacements.
            (default: 100)
            NOTE: This is overridden if dtDisplacements != None.
        int_max : int
            Maximum number of different intervals to consider when computing
            displacements for a given lag time. (default: 100)
            NOTE: This is overridden if dtDisplacements != None.
        min : int or None
            Minimum lag time at which to compute the displacements.
            (default: None)
            NOTE: if min == None, then min = 1.
            NOTE: This is overridden if dtDisplacements != None.
        max : int or None
            Maximum lag time at which to compute the displacements.
            (default: None)
            NOTE: if max == None, then max is taken to be the maximum according
                  to the choice of int_max.
            NOTE: This is overridden if dtDisplacements != None.
        jump : int
            Period in number of frames at which to check if particles have
            crossed any boundary. (default: 1) (see self.getDisplacements)
            NOTE: This is overridden if dtDisplacements != None.
        cage_relative : bool
            Remove displacement of the centre of mass given by nearest
            neighbours at initial time. (default: False)
            (see self.getDisplacements)
            NOTE: This is overridden if dtDisplacements != None.
        heaviside : bool
            Use Heaviside function rather than exponential of square as window
            function. (default: False)
        dtDisplacements : ((*,) int array-like,
                          (**, *, self.N, 2) float array-like) or None
            Lag time and displacements at these lag times from which to compute
            quantity.
            NOTE: if dtDisplacements == None, then compute with
                  self._displacements.

        Returns
        -------
        Q_stderr_chi : (*, 4) float numpy array
            Array of:
                (0) lag time,
                (1) dynamical overlap,
                (2) standard error on the computed dynamical overlap,
                (3) susceptibility of the computed dynamical overlap.
            (see self._mean_stderr_chi)
        """

        if type(dtDisplacements) == type(None):
            dt, displacements = self._displacements(
                n_max=n_max, int_max=int_max, min=min, max=max, jump=jump,
                cage_relative=cage_relative,
                initial_times=False)
        else:
            dt, displacements = dtDisplacements
        displacements2 = (wo_mean(displacements/a, axis=-2)**2).sum(axis=-1)

        if heaviside: quantities = (displacements2 > 1)*1.0
        else: quantities = np.exp(-displacements2)
        return self._mean_stderr_chi(dt, quantities)

    def overlap_relaxation_free_AOUP(q=0.5, a=1):
        """
        Returns relaxation time for a free Ornstein-Uhlenbeck particle, given as
        the time for the dynamical overlap function to decrease below threshold
        `q'.

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

    def selfIntScattFunc(self, k, dk=0.1,
        n_max=100, int_max=100, min=None, max=None,
        jump=1, cage_relative=False, dtDisplacements=None):
        """
        Compute self-intermediate scattering function.

        Parameters
        ----------
        k : float
            Wave-vector norm.
        dk : float
            Width of the wave-vector norm interval. (default: 0.1)
            (see coll_dyn_activem.maths.wave_vectors_dq)
        n_max : int
            Maximum number of lag times at which to compute the displacements.
            (default: 100)
            NOTE: This is overridden if dtDisplacements != None.
        int_max : int
            Maximum number of different intervals to consider when computing
            displacements for a given lag time. (default: 100)
            NOTE: This is overridden if dtDisplacements != None.
        min : int or None
            Minimum lag time at which to compute the displacements.
            (default: None)
            NOTE: if min == None, then min = 1.
            NOTE: This is overridden if dtDisplacements != None.
        max : int or None
            Maximum lag time at which to compute the displacements.
            (default: None)
            NOTE: if max == None, then max is taken to be the maximum according
                  to the choice of int_max.
            NOTE: This is overridden if dtDisplacements != None.
        jump : int
            Period in number of frames at which to check if particles have
            crossed any boundary. (default: 1) (see self.getDisplacements)
            NOTE: This is overridden if dtDisplacements != None.
        cage_relative : bool
            Remove displacement of the centre of mass given by nearest
            neighbours at initial time. (default: False)
            (see self.getDisplacements)
            NOTE: This is overridden if dtDisplacements != None.
        dtDisplacements : ((*,) int array-like,
                          (**, *, self.N, 2) float array-like) or None
            Lag time and displacements at these lag times from which to compute
            quantity.
            NOTE: if dtDisplacements == None, then compute with
                  self._displacements.

        Returns
        -------
        Fs_stderr_chi : (*, 4) float numpy array
            Array of:
                (0) lag time,
                (1) self-intermediate scattering function,
                (2) standard error on the computed self-intermediate scattering
                    function,
                (3) susceptibility of the computed self-intermediate scattering
                    function.
            (see self._mean_stderr_chi)
        wv : (*, 2) float Numpy array
            Array of (2\\pi/L nx, 2\\pi/L ny) wave vectors corresponding to the
            interval.
        """

        if type(dtDisplacements) == type(None):
            dt, displacements = self._displacements(
                n_max=n_max, int_max=int_max, min=min, max=max, jump=jump,
                cage_relative=cage_relative,
                initial_times=False)
        else:
            dt, displacements = dtDisplacements

        wave_vectors = wave_vectors_dq(self.L, k, dq=dk)
        dx, dy = (
            np.transpose(wo_mean(displacements, axis=-2), axes=(3, 0, 1, 2)))
        _msc = np.array(list(map(
            lambda kx, ky: self._mean_stderr_chi(dt, np.cos(kx*dx + ky*dy)),
            *np.transpose(wave_vectors))))
        assert _msc[:, :, 0].var(axis=0).sum() < 1e-12  # check Fs are computed at the same lag times
        msc = np.transpose([
            _msc[0, :, 0],                              # lag times
            _msc[:, :, 1].mean(axis=0),                 # self-intermediate scattering function
            np.sqrt((_msc[:, :, 2]**2).mean(axis=0)),   # standard error
            _msc[:, :, 3].mean(axis=0)])                # susceptibility
        return msc, wave_vectors

    def selfIntScattFunc_relaxation_free_AOUP(k):
        """
        Returns structural relaxation time for a free Ornstein-Uhlenbeck
        particle, given as the time for the self-intermediate scattering
        function to decrease below \\exp(-1).

        Parameters
        ----------
        k : float
            Wave-vector norm.

        Returns
        -------
        t : float
            Relaxation time.
        """

        return selfIntScattFunc_relaxation_free_AOUP(k, self.D, self.Dr)

    def vanHove(self, dt, nBins, int_max=None, jump=1, remove_cm=False,
        cage_relative=False, vmin=None, vmax=None):
        """
        Compute van Hove function.

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
            crossed any boundary. (default: 1) (see self.getDisplacements)
        remove_cm : bool
            Remove centre of mass displacement. (default: False)
            NOTE: does not affect result if self.N == 1.
        cage_relative : bool
            Remove displacement of the centre of mass given by nearest
            neighbours at initial time. (default: False)
            (see self.getDisplacements)
        vmin : float
            Minimum value of the bins. (default: None)
            NOTE: if vmin == None, then vmin = 0.
        vmax : float
            Maximum value of the bins. (default: None)
            NOTE: if vmax == None, then vmax = self.L/2.

        Returns
        -------
        G : (*, 2) float numpy array
            Array of:
                (0) distance,
                (1) radial van Hove function,
                (2) standard error on this measure.
        Gs : (*, 2) float numpy array
            Array of:
                (0) distance,
                (1) self part of the radial van Hove function,
                (2) standard error on this measure.
        """

        time0 = self._time0(dt, int_max=int_max)

        vmin = 0 if vmin == None else vmin
        vmax = self.L/2 if vmax == None else vmax

        _G, _Gs = [], []
        for t0 in time0:

            positions = self.getPositions(t0)
            displacements = self.getDisplacements(t0, t0 + dt,
                jump=jump, norm=False,
                remove_cm=remove_cm, cage_relative=cage_relative)
            distances = pycpp.getVanHoveDistances(
                positions, displacements, self.L)

            for dist, _list in zip(
                (distances, np.sqrt((displacements**2).sum(axis=-1))),
                (_G, _Gs)):

                bins, hist = Distribution(dist).hist(
                    nBins, vmin=vmin, vmax=vmax,
                    log=False, rescaled_to_max=False, occupation=True)
                hist = hist[bins > 0]
                hist /= distances.size
                bins = bins[bins > 0]
                _list += [
                    (hist/(2*np.pi*bins))               # radial
                    *(self.N/((vmax - vmin)/nBins))]    # normalisation

        G = np.array([[b, *mean_sterr(g)]
            for b, g in zip(bins, np.transpose(_G))])
        Gs = np.array([[b, *mean_sterr(gs)]
            for b, gs in zip(bins, np.transpose(_Gs))])
        return G, Gs

    def orientationNeighbours(self, time0, *dt, A1=1.15, jump=1,
        remove_cm=False):
        """
        Returns arrays of number of neighbouring particles which have the same
        displacement orientation between `time0' and `time0' + `dt'.

        Parameters
        ----------
        time0 : int
            Initial time.
        dt : int
            Lag time.
        A1 : float
            Distance relative to their diameters below which particles are
            considered bonded. (default: 1.15)
        jump : int
            Period in number of frames at which to check if particles have
            crossed any boundary. (default: 1)
            NOTE: `jump' must be chosen so that particles do not move a distance
                  greater than half the box size during this time.
            NOTE: This is only relevant for .dat files since these do not embed
                  unfolded positions.
        remove_cm : bool
            Remove centre of mass displacement. (default: False)
            NOTE: does not affect result if self.N == 1.

        Returns
        -------
        oneigbours : (**, self.N) int numpy array
            Number of neighbouring particles with same displacement orientation
            between `time0' and `time0' + `dt' with ** the number of `dt'
            provided.
        """

        return pycpp.getOrientationNeighbours(A1, self.L, self.diameters,
            self.getPositions(time0),
            *[self.getDisplacements(
                time0, time0 + t, jump=jump, remove_cm=remove_cm)
                for t in dt])

    def brokenBonds(self, time0, *dt, A1=1.15, A2=1.5, diameters=True):
        """
        Returns arrays of number of broken bonds for each particle, defined by
        the number of particles at distance lesser than `A1' at `time0' and
        greater than `A2' at `time0' + `dt'.

        Parameters
        ----------
        time0 : int
            Initial time.
        dt : int
            Lag time.
        A1 : float
            Distance below which particles are considered bonded.
            (default: 1.15)
        A2 : float
            Distance above which particles are considered unbonded.
            (default: 1.5)
        diameters : bool
            Rescale distances by respective average diameters. (default: True)

        Returns
        -------
        brokenBonds : (**, self.N) int numpy array
            Number of broken bonds between `time0' and `time0' + `dt' with **
            the number of `dt' provided.
        """

        if diameters: sigma = self.diameters
        else: sigma = np.full(self.diameters.shape, fill_value=1)

        return pycpp.getBrokenBonds(A1, A2, self.L, sigma,
            self.getPositions(time0),
            *[self.getPositions(time0 + t) for t in dt])

    def brokenBondsCor(self, dt, nBins, brokenBondsMin=1, A1=1.15, A2=1.5,
        int_max=100, min=None, max=None, rescale_pair_distribution=False):
        """
        Compute radial correlations of broken bonds.

        Parameters
        ----------
        dt : int
            Lag time at which to compute broken bonds.
        nBins : int
            Number of intervals of distances on which to compute the
            correlations.
        brokenBondsMin : int
            Threshold on the number of broken bonds. (default: 1)
        A1 : float
            Distance relative to their diameters below which particles are
            considered bonded. (default: 1.15)
        A2 : float
            Distance relative to their diameters above which particles are
            considered unbonded. (default: 1.5)
        int_max : int
            Maximum number of different intervals to consider when computing
            broken boonds for a given lag time. (default: 100)
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
        bb : (*,) float Numpy array
            Array of proportion of particles with at least brokenBondsMin broken
            bonds.
        Cbb : (*, 3) float Numpy array
            Array of (r, Cbb(r), errCbb(r)) with Cbb(r) the cylindrically
            averaged spatial correlations of broken bonds and errCbb(r) the
            standard error on this measure.
        zeta : (2,) float Numpy array
            Cooperativity and standard error on this measure.
        chi : float
            Susceptibility of broken bonds.
        """

        time0 = self._time0(dt, int_max=int_max)
        bonds = np.array(list(map(
            lambda t:
                (self.brokenBonds(t, dt, A1=A1, A2=A2)[0] >= brokenBondsMin)*1,
            time0)))
        corZeta = list(map(
            lambda t, b: self.getRadialCorrelations(
                t, b/((lambda B: 1 if B == 0 else B)(np.sqrt((b**2).mean()))),
                nBins, min=min, max=max,
                rescale_pair_distribution=rescale_pair_distribution),
            *(time0, bonds)))
        correlations = np.array([corZeta[t][0] for t in range(len(corZeta))])
        zeta = np.array([corZeta[t][1] for t in range(len(corZeta))])
        chi = self.N*bonds.mean(axis=-1).var()

        return (
            bonds.mean(axis=-1),
            np.array([
                [correlations[0, bin, 0],
                    *mean_sterr(correlations[:, bin, 1])]
                for bin in range(nBins)]),
            mean_sterr(zeta),
            chi)

    def bondBreakingCor(self, A1=1.15, A2=1.5,
        n_max=100, int_max=100, min=None, max=None):
        """
        Compute bond breaking correlation function.

        Parameters
        ----------
        A1 : float
            Distance relative to their diameters below which particles are
            considered bonded. (default: 1.15)
        A2 : float
            Distance relative to their diameters above which particles are
            considered unbonded. (default: 1.5)
        n_max : int
            Maximum number of lag times at which to compute the broken bonds.
            (default: 100)
        int_max : int
            Maximum number of different intervals to consider when computing
            broken bonds for a given lag time. (default: 100)
        min : int or None
            Minimum lag time at which to compute the broken bonds.
            (default: None)
            NOTE: if min == None, then min = 1.
        max : int or None
            Maximum lag time at which to compute the broken bonds.
            (default: None)
            NOTE: if max == None, then max is taken to be the maximum according
                  to the choice of int_max.

        Returns
        -------
        Fb : (*, 3) float Numpy array
            Array of:
                (0) lag time,
                (1) velocity correlation at this lag time,
                (2) standard error on this measure.
        """

        # LAG TIMES AND INITIAL TIMES

        time0, dt = self._dt(n_max=n_max, int_max=int_max, min=min, max=max)

        # COMPUTE CORRELATIONS

        bb = np.array(list(map(
            lambda t0: (
                lambda pos0: list(map(
                    lambda dt: (
                        lambda ini, fin: 1 - fin.sum()/ini.sum())(
                        *getBondsBrokenBonds(
                            pos0,
                            self.getDisplacements(t0, t0 + dt, remove_cm=True),
                            self.diameters, self.L, A1=A1, A2=A2)),
                    dt)))(
                self.getPositions(t0)),
            time0)))

        return np.array(list(map(
            lambda i, b: [dt[i], *mean_sterr(b)],
            *(range(bb.shape[1]), np.transpose(bb)))))

    def brokenPairs(self, time0, time1, A1=1.15, A2=1.5):
        """
        Returns array of broken bond truth values for each pair, where particles
        in the pair are at distance lesser than `A1' at `time0' and greater than
        `A2' at `time1'.

        (see pycpp.pairIndex)

        Parameters
        ----------
        time0 : int
            Initial time.
        time1 : int
            Final time.
        A1 : float
            Distance relative to their diameters below which particles are
            considered bonded. (default: 1.15)
        A2 : float
            Distance relative to their diameters above which particles are
            considered unbonded. (default: 1.5)

        Returns
        -------
        brokenPairs : (self.N*(self.N - 1)/2,) bool numpy array
            Broken bond between `time0' and `time1' truth values for each pair.
        """

        return pycpp.getBrokenBonds(A1, A2, self.L, self.diameters,
            self.getPositions(time0), self.getPositions(time1),
            pairs=True)[1][0]

    def brokenBondDensity(self, time0, *dt, A1=1.15, A2=1.5, k=0):
        """
        Returns fraction of particles with `k' broken bonds between `time0' and
        `time0' + `dt'.

        (see self.brokenBonds)

        Parameters
        ----------
        time0 : int
            Initial time.
        dt : int
            Lag time.
        A1 : float
            Distance relative to their diameters below which particles are
            considered bonded. (default: 1.15)
        A2 : float
            Distance relative to their diameters above which particles are
            considered unbonded. (default: 1.5)
        k : int
            Number of broken bonds.

        Returns
        -------
        phi : (**,) float numpy array
            Fraction of particles with `k' broken bonds between `time0' and
            `time0' + `dt' with ** the number of `dt' provided.
        """

        brokenBonds = self.brokenBonds(time0, *dt, A1=A1, A2=A2)
        return (brokenBonds == k).sum(axis=-1)/self.N

    def d2min(self, time0, time1, A1=2**(1./6.)):
        """
        Compute nonaffine squared displacements between `time0' and `time1'.

        Parameters
        ----------
        time0 : int
            Initial time.
        time1 : int
            Final time.
        A1 : float
            Distance relative to their diameters below which particles are
            considered bonded. (default: 1.15)

        Returns
        -------
        D2min : (self.N,) float numpy array
            Nonaffine squared displacements.
        """

        return pycpp.nonaffineSquaredDisplacement(
            self.getPositions(time0), self.getPositions(time1), self.L,
            A1, self.diameters)

    def _dt(self, n_max=100, int_max=100, min=None, max=None):
        """
        Returns initial times and lag times for intervals of time between `min'
        and `max'.

        Parameters
        ----------
        n_max : int
            Maximum number of lag times at which to compute the displacements.
            (default: 100)
        int_max : int
            Maximum number of different intervals to consider when computing
            displacements for a given lag time. (default: 100)
        min : int or None
            Minimum lag time at which to compute the displacements.
            (default: None)
            NOTE: if min == None, then min = 1.
        max : int or None
            Maximum lag time at which to compute the displacements.
            (default: None)
            NOTE: if max == None, then max is taken to be the maximum according
                  to the choice of int_max.

        Returns
        -------
        time0 : (*,) int numpy array
            Array of initial times.
        dt : (**,) int numpy array
            Array of lag times.
        """

        min = 1 if min == None else int(min)

        if self._type == 'datN':

            # LAG TIMES
            max = self.deltat.max() if max == None else int(max)
            dt = self.deltat[(self.deltat >= min)*(self.deltat <= max)]
            dt = itemgetter(*linspace(0, len(dt) - 1, n_max, endpoint=True))(dt)

            # INITIAL TIMES
            time0 = self.time0

        else:

            # LAG TIMES
            max = ((self.frames - self.skip - 1)//int_max if max == None
                else int(max))
            dt = linspace(min, max, n_max)

            # INITIAL TIMES
            time0 = np.linspace(
                self.skip, self.frames - 1,
                int((self.frames - 1 - self.skip)//dt.max()),
                endpoint=False, dtype=int)

        # INITIAL TIMES
        if int_max != None:
            indexes = list(OrderedDict.fromkeys(
                np.linspace(0, time0.size, int_max, endpoint=False, dtype=int)))
            time0 =  np.array(itemgetter(*indexes)(time0), ndmin=1)

        return np.array(time0), np.array(dt)

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
        time0 : (*,) int numpy array
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
        jump=1, remove_cm=False, cage_relative=False, initial_times=False):
        """
        Returns arrays of lag times and displacements evaluated over these lag
        times.

        Parameters
        ----------
        n_max : int
            Maximum number of lag times at which to compute the displacements.
            (default: 100)
        int_max : int
            Maximum number of different intervals to consider when computing
            displacements for a given lag time. (default: 100)
        min : int or None
            Minimum lag time at which to compute the displacements.
            (default: None)
            NOTE: if min == None, then min = 1.
        max : int or None
            Maximum lag time at which to compute the displacements.
            (default: None)
            NOTE: if max == None, then max is taken to be the maximum according
                  to the choice of int_max.
        jump : int
            Period in number of frames at which to check if particles have
            crossed any boundary. (default: 1) (see self.getDisplacements)
        remove_cm : bool
            Remove centre of mass displacement. (default: False)
            NOTE: does not affect result if self.N == 1.
        cage_relative : bool
            Remove displacement of the centre of mass given by nearest
            neighbours at initial time. (default: False)
            (see self.getDisplacements)
        initial_times : bool
            Return initial times at which displacements are computed.
            (default: False)

        Returns
        -------
        dt : (*,) int numpy array
            Array of lag times.
        displacements : (**, *, N, 2) float numpy array
            Array of displacements. (see self.dtDisplacements)
        time0 : [initial_times] (**,) int numpy array
            Initial times at which displacements are computed.
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
            dt = np.array(
                itemgetter(*linspace(0, len(dt) - 1, n_max, endpoint=True))(dt),
                ndmin=1)
        else:
            dt = logspace(min, max, n_max)

        displacements, time0 = self.dtDisplacements(dt, # array of displacements for different initial times and lag times
            int_max=int_max, jump=jump, norm=False,
            remove_cm=remove_cm, cage_relative=cage_relative,
            initial_times=True)

        dt = np.array(dt)
        if initial_times: return dt, displacements, time0
        return dt, displacements

    def _mean_stderr_chi(self, dt, quantities):
        """
        Returns mean, standard error and susceptibility for dynamic quantities.

        Parameters
        ----------
        dt : (*,) float array-like
            Array of lag times.
        quantities : (**, *, ***) float-array like
            Array of quantities.
            NOTE: **  = number of initial times
                  *   = number of lag times
                  *** = number of particles

        Returns
        -------
        out : (*, 4) float numpy array
            Array of:
                (0) lag time,
                (1) mean dynamic quantity at lag time,
                (2) standard error on the dynamic quantity at lag time,
                (3) susceptibility of the dynamic quantity at lag time.
        """

        dt = np.array(dt)
        quantities = np.array(quantities)
        assert quantities.ndim == 3
        assert quantities.shape[1] == dt.size
        assert quantities.shape[2] == self.N

        out = np.array(list(map(
            lambda i:
                (lambda q: [dt[i], *mean_sterr(q), self.N*q.var()])(
                    quantities[:, i].mean(axis=-1)),
            range(dt.size))))

        return out

class Velocities(Dat):
    """
    Compute and analyse velocities from simulation data.

    (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#Active%20Brownian%20particles)
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

        Dat.__init__(self, filename, loadWork=False, corruption=corruption) # initialise with super class (Dat)

        self.skip = skip    # skip the `skip' first frames in the analysis

    def nVelocities(self, int_max=None, norm=False, remove_cm=False):
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
        remove_cm : bool
            Remove centre of mass velocity. (default: False)
            NOTE: does not affect result if self.N == 1.

        Returns
        -------
        velocities : [not(norm)] (*, self.N, 2) float numpy array
                     [norm] (*, self.N) float numpy array
            Array of computed velocities.
        """

        return np.array(list(map(
            lambda time0:
                self.getVelocities(time0, norm=norm, remove_cm=remove_cm),
            self._time0(int_max=int_max))))

    def velocitiesPDF(self, int_max=None, remove_cm=False):
        """
        Returns probability density function of velocity norm.

        PARAMETERS
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.
        remove_cm : bool
            Remove centre of mass velocity. (default: False)
            NOTE: does not affect result if self.N == 1.

        Returns
        -------
        axes : numpy array
            Values at which the probability density function is evaluated.
        pdf : float numpy array
            Values of the probability density function.
        """

        return Distribution(
            self.nVelocities(int_max=int_max, norm=True, remove_cm=remove_cm)
            ).pdf()

    def velocitiesHist(self, nBins, remove_cm=False, int_max=None,
        vmin=None, vmax=None, log=False, rescaled_to_max=False):
        """
        Returns histogram with `nBins' bins of velocity norm.

        Parameters
        ----------
        nBins : int
            Number of bins of the histogram.
        remove_cm : bool
            Remove centre of mass velocity. (default: False)
            NOTE: does not affect result if self.N == 1.
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

        return Distribution(
            self.nVelocities(int_max=int_max, norm=True, remove_cm=remove_cm)
            ).hist(
                nBins, vmin=vmin, vmax=vmax, log=log,
                rescaled_to_max=rescaled_to_max)

    def energySpectrum(self, *q, dq=0.1, remove_cm=False, int_max=None):
        """
        Returns kinetic energy spectrum.

        (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#ABP%20flow%20characteristics)

        Parameters
        ----------
        q : float
            Wave vector norms at which to compute energy spectrum.
        dq : float
            Width of wave vector norm interval. (default: 0.1)
        remove_cm : bool
            Remove centre of mass velocity. (default: False)
            NOTE: does not affect result if self.N == 1.
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.

        Returns
        -------
        El : (*, 3) float Numpy array
            Array of (k, El(k), stdErr El(k)), with El(k) the longitudinal
            energy spectrum.
        Et : (*, 3) float Numpy array
            Array of (k, Et(k), stdErr Et(k)), with Et(k) the transversal energy
            spectrum.
        NOTE: E(k) = El(k) + Et(t), stdErr E(k) = stdErr El(k) + stdErr Et(k).
        """

        wv_len = np.array(list(map(
            lambda qn: len(wave_vectors_dq(self.L, qn, dq)),
            q)))
        q = np.array(q)[wv_len > 0]

        FTsq = []   # list of squared velocity fourier transform
        FTLsq, FTTsq = [], []
        for time, velocity in zip(
            self._time0(int_max=int_max),
            self.nVelocities(int_max=int_max, norm=False, remove_cm=remove_cm)):

            pos = self.getPositions(time)

            FTLsq += [
                list(map(
                    lambda qn: (2*np.pi*qn/(self.L**2))*np.mean(
                        list(map(
                            lambda qv: (np.abs(np.sum(
                                np.exp(-1j*(qv*pos).sum(axis=-1, keepdims=True))
                                    *(velocity
                                        *np.array([[qv[0], qv[1]]])
                                        /np.linalg.norm(qv))))**2),
                            wave_vectors_dq(self.L, qn, dq))),
                        axis=0),
                    q))]
            FTTsq += [
                list(map(
                    lambda qn: (2*np.pi*qn/(self.L**2))*np.mean(
                        list(map(
                            lambda qv: (np.abs(np.sum(
                                np.exp(-1j*(qv*pos).sum(axis=-1, keepdims=True))
                                    *(velocity
                                        *np.array([[-qv[1], qv[0]]])
                                        /np.linalg.norm(qv))))**2),
                            wave_vectors_dq(self.L, qn, dq))),
                        axis=0),
                    q))]

        return (
            np.array([[qn, *mean_sterr(vFTsq)]
                for qn, vFTsq in zip(q, np.transpose(FTLsq))]),
            np.array([[qn, *mean_sterr(vFTsq)]
                for qn, vFTsq in zip(q, np.transpose(FTTsq))]))

    def velocitiesw(self, dt, remove_cm=False):
        """
        Compute velocity Fourier transform in time.

        Parameters
        ----------
        dt : float
            Time step over which to discretise the trajectory. Missing frames
            will be interpolated.
        remove_cm : bool
            Remove centre of mass velocity. (default: False)
            NOTE: does not affect result if self.N == 1.

        Returns
        -------
        vw : (*, 2) complex numpy array
            Array of (w, v(w)) with v(w) the Fourier transform in time.
        """

        return None

    def velocitiesCor(self, nBins, remove_cm=False, int_max=None,
        min=None, max=None, rescale_pair_distribution=False):
        """
        Compute radial correlations of particles' velocities.

        Parameters
        ----------
        nBins : int
            Number of intervals of distances on which to compute the
            correlations.
        remove_cm : bool
            Remove centre of mass velocity. (default: False)
            NOTE: does not affect result if self.N == 1.
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
        Cvv : (*, 3) float Numpy array
            Array of (r, Cvv(r), errCvv(r)) with Cvv(r) the cylindrically
            averaged spatial correlations of velocity and errCvv(r) the standard
            error on this measure.
        zeta : (2,) float Numpy array
            Cooperativity and standard error on this measure.
        """

        corZeta = [
            (lambda v:
                self.getRadialCorrelations(
                    t, v/np.sqrt((v**2).sum(axis=-1).mean()),
                    nBins, min=min, max=max,
                    rescale_pair_distribution=rescale_pair_distribution))(
                self.getVelocities(t, norm=False, remove_cm=remove_cm))
            for t in self._time0(int_max=int_max)]
        correlations = np.array([corZeta[t][0] for t in range(len(corZeta))])
        zeta = np.array([corZeta[t][1] for t in range(len(corZeta))])

        return (
            np.array([
                [correlations[0, bin, 0],
                    *mean_sterr(correlations[:, bin, 1])]
                for bin in range(nBins)]),
            mean_sterr(zeta))

    def velocitiesOriCor(self, remove_cm=False, int_max=None):
        """
        Compute radial correlations of particles' velocity orientation.

        Parameters
        ----------
        remove_cm : bool
            Remove centre of mass velocity. (default: False)
            NOTE: does not affect result if self.N == 1.
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
                self.getPositions(t), self.L,
                    self.getVelocities(t, norm=False, remove_cm=remove_cm),
                sigma=self.diameters.mean())
            for t in self._time0(int_max=int_max)])

        return np.array([
            [correlations[0, bin, 0], *mean_sterr(correlations[:, bin, 1])]
            for bin in range(correlations.shape[1])])

    def velocitiesTimeCor(self, remove_cm=False,
        n_max=100, int_max=100, min=None, max=None):
        """
        Compute time correlations of particles' velocities, and time-dependent
        fourth moment of kurtosis.

        Parameters
        ----------
        remove_cm : bool
            Remove centre of mass velocity. (default: False)
            NOTE: does not affect result if self.N == 1.
        n_max : int
            Maximum number of lag times at which to compute the velocities.
            (default: 100)
        int_max : int
            Maximum number of different intervals to consider when computing
            velocities for a given lag time. (default: 100)
        min : int or None
            Minimum lag time at which to compute the velocities. (default: None)
            NOTE: if min == None, then min = 1.
        max : int or None
            Maximum lag time at which to compute the velocities. (default: None)
            NOTE: if max == None, then max is taken to be the maximum according
                  to the choice of int_max.

        Returns
        -------
        Cvv : (*, 3) float Numpy array
            Array of:
                (0) lag time,
                (1) velocity correlation at this lag time,
                (2) standard error on this measure.
        v0sq : (**,) float Numpy array
            Array of mean squared velocity at initial times.
        kurtosis : (*, 3) float Numpy array
            Array of:
                (0) lag time,
                (1) fourth moment of kurtosis at this lag time.
        """

        # LAG TIMES AND INITIAL TIMES

        time0, dt = self._dt(n_max=n_max, int_max=int_max, min=min, max=max)

        # COMPUTE CORRELATIONS

        initVelocity = np.array(list(map(
            lambda t0:
                self.getVelocities(t0, norm=False,
                    remove_cm=remove_cm),
            time0)))
        v0sq = (initVelocity**2).sum(axis=-1).mean(axis=-1)
        assert initVelocity.shape == (time0.size, self.N, 2)
        initVelocity = initVelocity.reshape(time0.size, 1, self.N, 2)

        finVelocity = np.array(list(map(
            lambda t0: np.array(list(map(
                lambda t: self.getVelocities(t0 + t, norm=False,
                    remove_cm=remove_cm),
                dt))),
            time0)))
        assert finVelocity.shape == (time0.size, dt.size, self.N, 2)

        prodVelocity = (initVelocity*finVelocity).sum(axis=-1).mean(axis=-1)
        velSqDiff = (
            (finVelocity**2).sum(axis=-1).mean(axis=-1)
            - (initVelocity**2).sum(axis=-1).mean(axis=-1))
        assert prodVelocity.shape == velSqDiff.shape
        assert prodVelocity.shape == (time0.size, dt.size)

        cvv = np.array(list(map(
            lambda i, t: [t, *mean_sterr(prodVelocity[:, i]/v0sq)],
            *(range(len(dt)), dt))))
        kurtosis = np.array(list(map(
            lambda i, t:
                (lambda dEc: [t, (dEc**4).mean()/(((dEc**2).mean())**2)])(
                    velSqDiff[:, i]),
            *(range(len(dt)), dt))))

        return cvv, v0sq, kurtosis

    def orientationNeighbours(self, *time0, A1=1.15, remove_cm=False):
        """
        Returns arrays of number of neighbouring particles which have the same
        velocity orientation at `time0'.

        Parameters
        ----------
        time0 : int
            Initial time.
        A1 : float
            Distance relative to their diameters below which particles are
            considered bonded. (default: 1.15)
        remove_cm : bool
            Remove centre of mass velocity. (default: False)
            NOTE: does not affect result if self.N == 1.

        Returns
        -------
        oneigbours : (**, self.N) int numpy array
            Number of neighbouring particles with same velocity orientation at
            `time0' with ** the number of `time0' provided.
        """

        return np.array([
            pycpp.getOrientationNeighbours(A1, self.L, self.diameters,
                self.getPositions(t),
                self.getVelocities(t, norm=False, remove_cm=remove_cm))[0]
            for t in time0])

    def _dt(self, n_max=100, int_max=100, min=None, max=None):
        """
        Returns initial times and lag times for intervals of time between `min'
        and `max'.

        Parameters
        ----------
        n_max : int
            Maximum number of lag times at which to compute the velocities.
            (default: 100)
        int_max : int
            Maximum number of different intervals to consider when computing
            velocities for a given lag time. (default: 100)
        min : int or None
            Minimum lag time at which to compute the velocities. (default: None)
            NOTE: if min == None, then min = 1.
        max : int or None
            Maximum lag time at which to compute the velocities. (default: None)
            NOTE: if max == None, then max is taken to be the maximum according
                  to the choice of int_max.

        Returns
        -------
        time0 : (*,) int numpy array
            Array of initial times.
        dt : (**,) int numpy array
            Array of lag times.
        """

        min = 1 if min == None else int(min)

        if self._type == 'datN':

            # LAG TIMES
            max = self.deltat.max() if max == None else int(max)
            dt = self.deltat[(self.deltat >= min)*(self.deltat <= max)]
            dt = itemgetter(*linspace(0, len(dt) - 1, n_max, endpoint=True))(dt)

            # INITIAL TIMES
            time0 = self.time0

        else:

            # LAG TIMES
            max = ((self.frames - self.skip - 1)//int_max if max == None
                else int(max))
            dt = linspace(min, max, n_max)

            # INITIAL TIMES
            time0 = np.linspace(
                self.skip, self.frames - 1,
                int((self.frames - 1 - self.skip)//dt.max()),
                endpoint=False, dtype=int)

        # INITIAL TIMES
        if int_max != None:
            indexes = list(OrderedDict.fromkeys(
                np.linspace(0, time0.size, int_max, endpoint=False, dtype=int)))
            time0 =  np.array(itemgetter(*indexes)(time0), ndmin=1)

        return np.array(time0), np.array(dt)

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
        time0 : (*,) int numpy array
            Array of frames.
        """

        if self._type == 'datN':
            time0 = []
            for t0 in self.time0[self.time0 != self.frameIndices[-1]]:
                try:
                    self._unfolded_position(t0, self.N - 1)
                    time0 += [t0]
                except struct.error:
                    break
            time0 = np.array(time0)
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
        Compute spatial correlations of particles' orientations.

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
        Cuu : (*, 2) float Numpy array
            Array of (r, Cuu(r), errCuu(r)) with Cuu(r) the cylindrically
            averaged spatial correlations of orientation and errCuu(r) the
            standard error on this measure.
        zeta : (2,) float Numpy array
            Cooperativity and standard error on this measure.
        """

        corZeta = [
            self.getRadialCorrelations(t, self.getDirections(t),
                nBins, min=min, max=max,
                rescale_pair_distribution=rescale_pair_distribution)
            for t in self._time0(int_max=int_max)]
        correlations = np.array([corZeta[t][0] for t in range(len(corZeta))])
        zeta = np.array([corZeta[t][1] for t in range(len(corZeta))])

        return (
            np.array([
                [correlations[0, bin, 0],
                    *mean_sterr(correlations[:, bin, 1])]
                for bin in range(nBins)]),
            mean_sterr(zeta))

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
        time0 : (*,) int numpy array
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

        # LAG TIMES AND INITIAL TIMES

        time0, dt = self._dt(n_max=n_max, int_max=int_max, min=min, max=max)

        # COMPUTE CORRELATIONS

        Cpp = np.array(list(map(
            lambda t: [t, np.mean(list(map(
                lambda t0:
                    (wo_mean(self.getPropulsions(t0, norm=False), axis=-2)
                    *wo_mean(self.getPropulsions(t0 + t, norm=False), axis=-2)
                    ).sum(axis=-1),
                time0)))],
            dt)))

        return Cpp

    def propulsionsRadialCor(self, nBins, int_max=None, min=None, max=None,
        rescale_pair_distribution=False):
        """
        Compute spatial correlations of particles' self-propulsions.

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
        Cpp : (*, 2) float Numpy array
            Array of (r, Cpp(r), errCpp(r)) with Cvv(r) the cylindrically
            averaged spatial correlations of self-propulsion and errCvv(r) the
            standard error on this measure.
        zeta : (2,) float Numpy array
            Cooperativity and standard error on this measure.
        """

        corZeta = [
            (lambda p:
                self.getRadialCorrelations(
                    t, p/np.sqrt((p**2).sum(axis=-1).mean()),
                    nBins, min=min, max=max,
                    rescale_pair_distribution=rescale_pair_distribution))(
                self.getPropulsions(t, norm=False))
            for t in self._time0(int_max=int_max)]
        correlations = np.array([corZeta[t][0] for t in range(len(corZeta))])
        zeta = np.array([corZeta[t][1] for t in range(len(corZeta))])

        return (
            np.array([
                [correlations[0, bin, 0],
                    *mean_sterr(correlations[:, bin, 1])]
                for bin in range(nBins)]),
            mean_sterr(zeta))

    def propulsions_cor_th_AOUP(self, *dt):
        """
        Returns value of theoretical self-propulsion vector time correlation for
        AOUPs.

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

    def _dt(self, n_max=100, int_max=100, min=None, max=None):
        """
        Returns initial times and lag times for intervals of time between `min'
        and `max'.

        Parameters
        ----------
        n_max : int
            Maximum number of lag times at which to compute the propulsions.
            (default: 100)
        int_max : int
            Maximum number of different intervals to consider when computing
            propulsions for a given lag time. (default: 100)
        min : int or None
            Minimum lag time at which to compute the propulsions.
            (default: None)
            NOTE: if min == None, then min = 1.
        max : int or None
            Maximum lag time at which to compute the propulsions.
            (default: None)
            NOTE: if max == None, then max is taken to be the maximum according
                  to the choice of int_max.

        Returns
        -------
        time0 : (*,) int numpy array
            Array of initial times.
        dt : (**,) int numpy array
            Array of lag times.
        """

        min = 1 if min == None else int(min)

        if self._type == 'datN':

            # LAG TIMES
            max = self.deltat.max() if max == None else int(max)
            dt = self.deltat[(self.deltat >= min)*(self.deltat <= max)]
            dt = itemgetter(*linspace(0, len(dt) - 1, n_max, endpoint=True))(dt)

            # INITIAL TIMES
            time0 = self.time0

        else:

            # LAG TIMES
            max = ((self.frames - self.skip - 1)//int_max if max == None
                else int(max))
            dt = linspace(min, max, n_max)

            # INITIAL TIMES
            time0 = np.linspace(
                self.skip, self.frames - 1,
                int((self.frames - 1 - self.skip)//dt.max()),
                endpoint=False, dtype=int)

        # INITIAL TIMES
        if int_max != None:
            indexes = list(OrderedDict.fromkeys(
                np.linspace(0, time0.size, int_max, endpoint=False, dtype=int)))
            time0 =  np.array(itemgetter(*indexes)(time0), ndmin=1)

        return np.array(time0), np.array(dt)

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
        time0 : (*,) int numpy array
            Array of frames.
        """

        if self._type == 'datN':
            time0 = self.time0
        else:
            time0 = np.array(range(self.skip, self.frames))

        if int_max == None: return time0
        indexes = linspace(0, time0.size, int_max, endpoint=False)
        return np.array(itemgetter(*indexes)(time0), ndmin=1)

class BondOrder(Velocities, Positions):
    """
    Compute and analyse hexatic bond order parameter from simulation data.
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

        Velocities.__init__(self, filename, skip=skip, corruption=corruption)   # initialise with super class (Velocities)

    def bondOrderCor(self, nBins, int_max=None, min=None, max=None, arg=False,
        rescale_pair_distribution=True):
        """
        Compute radial correlations of particles' hexatic bond order parameter.

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
        arg : bool
            Compute correlations of direction of the bond order parameter rather
            than bond order parameter itself. (default: False)
        rescale_pair_distribution : bool
            Rescale correlations by pair distribution function. (default: True)

        Returns
        -------
        Cpp : (*, 3) float Numpy array
            Array of (r, Cpp(r), errCpp(r)) with Cpp(r) the cylindrically
            averaged spatial correlations of bond order parameter and errCpp(r)
            the standard error on this measure.
        zeta : (2,) float Numpy array
            Cooperativity and standard error on this measure.
        """

        corZeta = [
            (lambda p:
                self.getRadialCorrelations(
                    t, p,
                    nBins, min=min, max=max,
                    rescale_pair_distribution=rescale_pair_distribution))(
                (lambda _p:
                    _p/np.sqrt((_p*_p.conj()).mean().real) if not(arg)
                    else np.array(list(map(
                        lambda a: (np.cos(a), np.sin(a)),
                        np.angle(_p)))))(
                self.getBondOrderParameter(t)))
            for t in self._time0(int_max=int_max)]
        correlations = np.array([corZeta[t][0] for t in range(len(corZeta))])
        zeta = np.array([corZeta[t][1] for t in range(len(corZeta))])

        return (
            np.array([
                [correlations[0, bin, 0],
                    *mean_sterr(correlations[:, bin, 1])]
                for bin in range(nBins)]),
            mean_sterr(zeta))

    def bondOrderTimeCor(self, n_max=100, int_max=100, min=None, max=None,
        arg=False):
        """
        Compute time correlations of particles' bond order parameter.

        Parameters
        ----------
        n_max : int
            Maximum number of lag times at which to compute the bond order
            parameters. (default: 100)
        int_max : int
            Maximum number of different intervals to consider when computing
            bond order parameters for a given lag time. (default: 100)
        min : int or None
            Minimum lag time at which to compute the bond order parameters.
            (default: None)
            NOTE: if min == None, then min = 1.
        max : int or None
            Maximum lag time at which to compute the bond order parameters.
            (default: None)
            NOTE: if max == None, then max is taken to be the maximum according
                  to the choice of int_max.
        arg : bool
            Compute correlations of direction of the bond order parameter rather
            than bond order parameter itself. (default: False)

        Returns
        -------
        Cpp : (*, 3) float Numpy array
            Array of:
                (0) lag time,
                (1) bond order parameter correlation at this lag time,
                (2) standard error on this measure.
        psi6sq : (**,) float Numpy array
            Array of mean squared bond order parameter at initial times.
        """

        # LAG TIMES AND INITIAL TIMES

        time0, dt = self._dt(n_max=n_max, int_max=int_max, min=min, max=max)

        # COMPUTE CORRELATIONS

        initPsi6 = np.array(list(map(
            lambda t0:
                self.getBondOrderParameter(t0),
            time0)))
        psi6sq = (initPsi6*initPsi6.conj()).mean(axis=-1).real
        assert initPsi6.shape == (time0.size, self.N)
        if arg:
            initPsi6 = np.angle(initPsi6)
            initPsi6 = initPsi6.reshape(time0.size, 1, self.N, 1)
            initPsi6 = np.concatenate(
                (np.cos(initPsi6), np.sin(initPsi6)), axis=-1)
        else:
            initPsi6 = initPsi6.reshape(time0.size, 1, self.N)

        finPsi6 = np.array(list(map(
            lambda t0: np.array(list(map(
                lambda t: self.getBondOrderParameter(t0 + t),
                dt))),
            time0)))
        assert finPsi6.shape == (time0.size, dt.size, self.N)
        if arg:
            finPsi6 = np.angle(finPsi6)
            finPsi6 = finPsi6.reshape(time0.size, dt.size, self.N, 1)
            finPsi6 = np.concatenate(
                (np.cos(finPsi6), np.sin(finPsi6)), axis=-1)
        else:
            finPsi6 = finPsi6.conj()

        prodPsi6 = initPsi6*finPsi6
        if prodPsi6.ndim == 4: prodPsi6 = prodPsi6.sum(axis=-1)
        prodPsi6 = prodPsi6.mean(axis=-1)
        assert prodPsi6.shape == (time0.size, dt.size)

        cpp = np.array(list(map(
            lambda i, t: [t, *mean_sterr(prodPsi6[:, i])],
            *(range(len(dt)), dt))))

        return cpp.real, psi6sq

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
    Returns value of the dynamical overlap funtion at lag time `dt' for a free
    Ornstein-Uhlenbeck particle.

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

def selfIntScattFunc_free_AOUP(k, D, Dr, dt):
    """
    Returns value of the self-intermediate scattering function at lag time `dt'
    for a free Ornstein-Uhlenbeck particle.

    Parameters
    ----------
    k : float
        Wave-vector norm.
    D : float
        Translational diffusivity.
    Dr : float
        Rotational diffusivity.
    dt : float
        Lag time at which to evaluate the self-intermediate scattering function.

    Returns
    -------
    Q : float
        Dynamical overlap.
    """

    return np.exp(-(k**2)*msd_th_AOUP(D, Dr, dt)/2)

def selfIntScattFunc_relaxation_free_AOUP(k, D, Dr):
    """
    Returns structural relaxation time for a free Ornstein-Uhlenbeck particle,
    given as the time for the self-intermediate scattering function to decrease
    below \\exp(-1).

    Parameters
    ----------
    k : float
        Wave-vector norm.
    D : float
        Translational diffusivity.
    Dr : float
        Rotational diffusivity.

    Returns
    -------
    t : float
        Relaxation time.
    """

    return (
        (Dr
            + 2*D*(k**2)*(1 + lambertw(-np.exp(-1 - Dr/(2*D*(k**2))))))
        /(2*D*Dr*(k**2)))

def overlap(disp, dt, *a):
    """
    Returns overlap for different time and length scales.

    Parameters
    ----------
    disp : (*, **, ***, 2) float array-like
        Displacements.
    dt : (**,) float array-like
        Lag times corresponding to displacements.
    a : float
        Length scale.

    Returns
    -------
    chi : (****, **, 4) float numpy array
        (0) Lag time.
        (1) Overlap
        (2) Standard error on the overlap.
        (3) Dynamic susceptibility.
    """

    chi = np.array(list(map(
        lambda x:
            list(map(
                lambda i:
                    (lambda quantities:
                        (lambda q:
                            [dt[i], *mean_sterr(q), disp.shape[-2]*q.var()])(
                        quantities.mean(axis=-1)))(
                    ((disp[:, i]**2).sum(axis=-1) < x**2)*1.0),
                range(len(dt)))),
        a)))

    return chi

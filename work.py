"""
Module work provides classes to compute and analyse active work and active work
autocorrelations and correlations with order parameter.
"""

import numpy as np
from collections import OrderedDict
from operator import itemgetter

from coll_dyn_activem.read import Dat
from coll_dyn_activem.maths import Distribution, mean_sterr,\
    linspace, logspace, CurveFit

class ActiveWork(Dat):
    """
    Compute and analyse active work from simulation data.

    (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#Active%20Brownian%20particles)
    """

    def __init__(self, filename, workPart='all', skip=1, corruption=None):
        """
        Loads file.

        Parameters
        ----------
        filename : string
            Name of input data file.
        workPart : string
            Part of the active work to consider in computations:
                * 'all': active work,
                * 'force': force part of the active work,
                * 'orientation': orientation part of the active work,
                * 'noise': noise part of the active work.
            (default: 'all')
            NOTE: This can be changed at any time by calling self._setWorkPart.
        skip : int
            Skip the `skip' first computed values of the active work in the
            following calculations. (default: 1)
            NOTE: This can be changed at any time by setting self.skip.
        corruption : str or None
            Pass corruption test for given file type (see
            coll_dyn_activem.read.Dat). (default: None)
            NOTE: if corruption == None, then the file has to pass corruption
                  tests.
        """

        super().__init__(filename, loadWork=True, corruption=corruption)    # initialise with super class

        self.workDict = {   # hash table of active work parts
            'all': self.activeWork,
            'force': self.activeWorkForce,
            'orientation': self.activeWorkOri,
            'noise':
                self.activeWork - self.activeWorkForce - self.activeWorkOri}
        self._setWorkPart(workPart)

        self.skip = skip    # skip the `skip' first measurements of the active work in the analysis

    def nWork(self, n, int_max=None):
        """
        Returns normalised rate of active work averaged on packs of size `n' of
        consecutive individual active works.

        NOTE: Individual active work refers to the normalised rate of active
              work on self.dumpPeriod*self.framesWork consecutive frames and
              stored as element of self.workArray.

        Parameters
        ----------
        n : int
            Size of packs on which to average active work.
        int_max : int or None
            Maximum number of packs consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of packs.
                  int_max cannot exceed the maximum number of nonoverlapping
                  packs.

        Returns
        -------
        workAvegared : float numpy array
            Array of computed active works.
        """

        workAvegared = []
        for i in self._time0(n, int_max=int_max):
            workAvegared += [self.workArray[i:i + n].mean()]

        return np.array(workAvegared)

    def varWork(self, n_max=100, int_max=100, min=None, max=None, log=True):
        """
        Parameters
        ----------
        n_max : int
            Maximum number of values at which to evaluate the variance.
            (default: 100)
        int_max : int or None
            Maximum number of different intervals to consider in order to
            compute the mean which appears in the variance expression.
            (default: None)
            NOTE: if int_max == None, then a maximum number of intervalls will
                  be considered.
        min : int or None
            Minimum value at which to compute the variance. (default: None)
            NOTE: this value is passed as `min' to self.n.
        max : int or None
            Maximum value at which to compute the correlation. (default: None)
            NOTE: this value is passed as `max' to self.n.
        log : bool
            Logarithmically space values at which the variance is computed.
            (default: True)

        Returns
        -------
        var : (*, 2) float Numpy array
            Array of variance:
                (0) absolute time over which the work is averaged,
                (1) computed variance on this interval.
        """

        var = []
        for n in self._n(n_max=n_max, min=min, max=max, log=log):
            var += [[
                n*self.dumpPeriod*self.framesWork*self.dt,
                self.nWork(n, int_max=int_max).var()]]

        return np.array(var)

    def nWorkPDF(self, n):
        """
        Returns probability density function of normalised rate of active work
        on packs of `n' of consecutive individual active works.

        NOTE: Individual active work refers to the normalised rate of active
              work on self.dumpPeriod*self.framesWork consecutive frames and
              stored as element of self.workArray.

        Parameters
        ----------
        n : int
            Size of packs on which to average active work.

        Returns
        -------
        axes : numpy array
            Values at which the probability density function is evaluated.
        pdf : float numpy array
            Values of the probability density function.
        """

        return Distribution(self.nWork(n)).pdf()

    def nWorkHist(self, n, nBins, vmin=None, vmax=None, log=False,
        rescaled_to_max=False):
        """
        Returns histogram with `nBins' bins of normalised rate of active work on
        packs of `n' of consecutive individual active works.

        NOTE: Individual active work refers to the normalised rate of active
              work on self.dumpPeriod*self.framesWork consecutive frames and
              stored as element of self.workArray.

        Parameters
        ----------
        n : int
            Size of packs on which to average active work.
        nBins : int
            Number of bins of the histogram.
        vmin : float
            Minimum value of the bins. (default: self.nWork(n).min())
        vmax : float
            Maximum value of the bins. (default: self.nWork(n).max())
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

        return Distribution(self.nWork(n)).hist(nBins,
            vmin=vmin, vmax=vmax, log=log, rescaled_to_max=rescaled_to_max)

    def nWorkGauss(self, n, *x, cut=None,
        rescaled_to_max=False):
        """
        Returns values of the Gaussian function corresponding to the mean and
        variance of self.nWork(n).

        Parameters
        ----------
        n : int
            Size of packs on which to average active work.
        x : float
            Values at which to evaluate the Gaussian function.
        cut : float or None
            Width in units of self.nWork(n).std() to consider when computing
            mean and standard deviation. (see coll_dyn_activem.maths.meanStdCut)
            (default: None)
            NOTE: if cut == None, the width is taken to infinity, i.e. no value
                  is excluded.
        rescaled_to_max : bool
            Rescale function by its computed maximum. (default: False)

        Returns
        -------
        gauss : float numpy array
            Values of the Gaussian function at x.
        """

        return Distribution(self.nWork(n)).gauss(*x,
            cut=cut, rescaled_to_max=rescaled_to_max)

    def corWorkWorkAve(self,
        tau0=1, n_max=100, int_max=None, min=None, max=None, log=True):
        """
        Compute correlations of the fluctuations of the work averaged over
        `tau0' at the beginning of an interval and the fluctuations of the work
        averaged over the whole interval.

        Parameters
        ----------
        tau0 : int
            Number of consecutive individual active works on which to average
            it. (default: 1)
        n_max : int
            Maximum number of values at which to evaluate the correlation.
            (default: 100)
        int_max : int or None
            Maximum number of different intervals to consider in order to
            compute the mean which appears in the correlation expression.
            (default: None)
            NOTE: if int_max == None, then a maximum number of disjoint
                     intervals will be considered.
        min : int or None
            Minimum value at which to compute the correlation. (default: None)
            NOTE: if min == None then this value is passed as `min' to self.n,
                  otherwise the minimum of `tau0' and `min' is taken.
        max : int or None
            Maximum value at which to compute the correlation. (default: None)
            NOTE: this value is passed as `max' to self.n.
        log : bool
            Logarithmically space values at which the correlations are
            computed. (default: True)

        Returns
        -------
        cor : (5, *) numpy array
            Array of:
                (0) value at which the correlation is computed,
                (1) mean of the computed correlation,
                (2) standard error of the computed correlation,
                (3) standard deviation of the active work computed at the
                    beginning of the interval,
                (4) standard deviation of the active work computed over the
                    interval.
        """

        cor = []
        for n in self._n(n_max=n_max, max=max, log=log,
            min=(None if min == None else np.max([min, tau0]))):
            worksTot = (lambda l: l - np.mean(l))(
                self.nWork(n, int_max=int_max))                     # fluctuations of the active wok on intervals of size n
            worksIni = (lambda l: l - np.mean(l))(
                list(map(
                    lambda t: self.workArray[t:t + tau0].mean(),    # fluctuations of the active work averaged on tau0 at the beginning of these intervals
                    self._time0(n, int_max=int_max))))
            workWork = worksTot*worksIni
            cor += [[n, *mean_sterr(workWork),
                np.std(worksTot), np.std(worksIni)]]

        return np.array(cor)

    def corWorkWorkIns(self, workPart1=None, workPart2=None,
        tau0=1, n_max=100, int_max=None, min=None, max=None, log=True):
        """
        Compute (cross) correlations of the fluctuations of the work averaged
        over `tau0' between different times.

        Parameters
        ----------
        workPart1 : string
            Part of the active work to consider at the beginning of the
            interval:
                * 'all': active work,
                * 'force': force part of the active work,
                * 'orientation': orientation part of the active work,
                * 'noise': noise part of the active work.
            (default: None)
            NOTE: if workPart1 == None, then self.workArray is taken.
        workPart2 : string
            Part of the active work to consider at the end of the interval.
            (default: None)
            NOTE: if workPart2 == None, then self.workArray is taken.
        tau0 : int
            Number of consecutive individual active works on which to average
            it. (default: 1)
        n_max : int
            Maximum number of values at which to evaluate the correlation.
            (default: 100)
        int_max : int or None
            Maximum number of different intervals to consider in order to
            compute the mean which appears in the correlation expression.
            (default: None)
            NOTE: if int_max == None, then a maximum number of disjoint
                  intervals will be considered.
        min : int or None
            Minimum value at which to compute the correlation. (default: None)
            NOTE: if min == None then min = `tau0', otherwise the minimum of
                  `tau0' and `min' is taken.
        max : int or None
            Maximum value at which to compute the correlation. (default: None)
            NOTE: this value is passed as `max' to self.n.
        log : bool
            Logarithmically space values at which the correlations are
            computed. (default: True)

        Returns
        -------
        cor : (5, *) numpy array
            Array of:
                (0) value at which the correlation is computed,
                (1) mean of the computed correlation,
                (2) standard error of the computed correlation,
                (3) standard deviation of the active work computed at the
                    beginning of the interval,
                (4) standard deviation of the active work computed at the end
                    of the interval.
        """

        work1 = (self.workArray if workPart1 == None else
            self.workDict[workPart1])
        work2 = (self.workArray if workPart2 == None else
            self.workDict[workPart2])

        cor = []
        for n in self._n(n_max=n_max, log=log,
            min=(2*tau0 if min == None else np.max([tau0 + min, 2*tau0])),
            max=(None if max == None else tau0 + max)):
            worksIni = (lambda l: l - np.mean(l))(
                list(map(
                    lambda t: work1[t:t + tau0].mean(),         # fluctuations of the active work averaged between t0 and t0 + tau0
                    self._time0(n, int_max=int_max))))
            worksFin = (lambda l: l - np.mean(l))(
                list(map(
                    lambda t: work2[t + n - tau0:t + n].mean(), # fluctuations of the active work averaged between t0 and t0 + tau0
                    self._time0(n, int_max=int_max))))
            workWork = worksIni*worksFin
            cor += [[n - tau0, *mean_sterr(workWork),
                np.std(worksIni), np.std(worksFin)]]

        return np.array(cor)

    def corWorkWorkInsBruteForce(self, workPart1=None, workPart2=None,
        tau0=1, n_max=100, int_max=None, max=None, log=True):
        """
        Compute (cross) correlations of the fluctuations of the work averaged
        over `tau0' between different times.

        This algorithm computes the correlations more quickly by averaging over
        successive couples of initial and final values of the active work.
        Results of this function should then be taken with care as some other
        unwanted low-time correlations could be picked.

        Parameters
        ----------
        workPart1 : string
            Part of the active work to consider at the beginning of the
            interval:
                * 'all': active work,
                * 'force': force part of the active work,
                * 'orientation': orientation part of the active work,
                * 'noise': noise part of the active work.
            (default: None)
            NOTE: if workPart1 == None, then self.workArray is taken.
        workPart2 : string
            Part of the active work to consider at the end of the interval.
            (default: None)
            NOTE: if workPart2 == None, then self.workArray is taken.
        tau0 : int
            Number of consecutive individual active works on which to average
            it. (default: 1)
        n_max : int
            Maximum number of values at which to evaluate the correlation.
            (default: 100)
        int_max : int or None
            Maximum number of different intervals to consider in order to
            compute the mean which appears in the correlation expression.
            (default: None)
            NOTE: if int_max == None, then a maximum number of intervals will
                  intervals will be considered.
        max : int or None
            Maximum value at which to compute the correlation in units of tau0.
            (default: None)
            NOTE: if max == None, the maximum number of values is computed.
        log : bool
            Logarithmically space values at which the correlations are
            computed. (default: True)

        Returns
        -------
        cor : (3, *) numpy array
            Array of:
                (0) value at which the correlation is computed,
                (1) mean of the computed correlation,
                (2) standard error of the computed correlation.
        """

        work1 = (self.workArray if workPart1 == None else
            self.workDict[workPart1])
        work2 = (self.workArray if workPart2 == None else
            self.workDict[workPart2])

        if log: space = logspace
        else: space = linspace

        if int_max == None: int_max = (self.numberWork - self.skip)//tau0
        Nsample = int(np.min(
            [(self.numberWork - self.skip)//tau0, int_max*tau0]))   # size of the sample of consecutive normalised rates of active work to consider
        workArray1 = np.array(list(map(                             # array of consecutive normalised rate of active work averaged of time tau0
            lambda t: work1[
                self.skip + t*tau0:self.skip + t*tau0 + tau0].mean(),
            range(Nsample))))
        workArray1 -= workArray1.mean()                             # only considering fluctuations to the mean
        if workPart1 == workPart2: workArray2 = workArray1
        else:
            workArray2 = np.array(list(map(                         # array of consecutive normalised rate of active work averaged of time tau0
                lambda t: work2[
                    self.skip + t*tau0:self.skip + t*tau0 + tau0].mean(),
                range(Nsample))))
            workArray2 -= workArray2.mean()                         # only considering fluctuations to the mean

        lagTimes = space( # array of lag times considered
            1,
            (Nsample - 1) if max == None else int(np.min([max, Nsample - 1])),
            n_max)

        cor = list(map(
            lambda dt: [
                tau0*dt,
                *mean_sterr(
                    (workArray1*np.roll(workArray2, -dt))[:Nsample - dt])],
            lagTimes))

        return np.array(cor)

    def varWorkFromCorWork(self, tau0=1, n=100, int_max=None, bruteForce=True):
        """
        Compute variance of the active work from its "instantaneous"
        fluctuations correlations.

        This function is primarily for consistency testing of
        the correlations functions.

        Parameters
        ----------
        tau0 : int
            Number of consecutive individual active works on which to average
            it, and for which correlations will be computed. (default: 1)
        n : int
            Compute variance for tau = i*tau0 with i in {1, ..., n}.
            (default: 100)
        int_max : int or None
            Maximum number of different intervals to consider in order to
            compute the mean which appears in the correlation expression.
            (default: None)
            NOTE: if int_max == None, then a maximum number of intervals will
                  intervals will be considered (joint if bruteForce else
                  disjoint).
        bruteForce : bool
            Use self.corWorkWorkInsBruteForce rather than self.corWorkWorkIns.
            (default: True)

        Returns
        -------
        var : (3, *) numpy array
            Array of:
                (0) value at which the variance is computed,
                (1) mean of the computed variance,
                (2) standard error of the computed variance.
        """

        if bruteForce: corWorkWorkIns = self.corWorkWorkInsBruteForce
        else: corWorkWorkIns = self.corWorkWorkIns

        if bruteForce:
            cor = self.corWorkWorkInsBruteForce(tau0,
                n_max=n, int_max=int_max, max=n - 1, log=False)
        else:
            cor = self.corWorkWorkIns(tau0,
                n_max=n, int_max=int_max, min=tau0, max=(n - 1)*tau0, log=False)

        var0 = mean_sterr((lambda l: (l - l.mean())**2)
            (self.nWork(tau0, int_max=int_max)))

        var = []
        for n0 in range(1, n + 1):
            var += [[n0*tau0, var0[0]/n0, (var0[1]/n0)**2]]
            for i in range(1, n0):
                var[-1][1] += 2*(n0 - i)*cor[i - 1, 1]/(n0**2)
                var[-1][2] += (2*(n0 - i)*cor[i - 1, 2]/(n0**2))**2
            var[-1][2] = np.sqrt(var[-1][2])

        return np.array(var)

    def corWorkOrderAve(self,
        n_max=100, int_max=None, min=None, max=None, log=False):
        """
        Compute correlations of the fluctuations of the work averaged over an
        interval of length `tau' and the fluctuations of the order parameter norm
        at the beginning of the interval.

        Parameters
        ----------
        n_max : int
            Maximum number of values at which to evaluate the correlation.
            (default: 100)
        int_max : int or None
            Maximum number of different intervals to consider in order to
            compute the mean which appears in the correlation expression.
            (default: None)
            NOTE: if int_max == None, then a maximum number of disjoint
                     intervals will be considered.
        min : int or None
            Minimum value at which to compute the correlation. (default: None)
            NOTE: this value is passed as `min' to self.n.
        max : int or None
            Maximum value at which to compute the correlation. (default: None)
            NOTE: this value is passed as `max' to self.n.
        log : bool
            Logarithmically space values at which the correlations are
            computed. (default: True)

        Returns
        -------
        cor : (5, *) numpy array
            Array of:
                (0) value at which the correlation is computed,
                (1) mean of the computed correlation,
                (2) standard error of the computed correlation,
                (3) standard deviation of the active work,
                (4) standard deviation of the order parameter norm.
        """

        cor = []
        for n in self._n(n_max=n_max, min=min, max=max, log=log):
            works = (lambda l: l - np.mean(l))(self.nWork(n, int_max=int_max))  # fluctuations of the active wok on intervals of size n
            orders = (lambda l: l - np.mean(l))(np.array(list(map(              # fluctuations of the order parameter norm at the beginning of these intervals
                lambda t: self.getOrderParameter(t, norm=True),
                self.framesWork*self._time0(n, int_max=int_max)))))
            workOrder = works*orders
            cor += [[n, *mean_sterr(workOrder),
                np.std(works), np.std(orders)]]

        return np.array(cor)

    def corWorkOrderIns(self,
        tau0=1, n_max=100, int_max=None, min=None, max=None, log=False):
        """
        Compute correlations of the fluctuations of the order parameter norm
        at a given time and the fluctuations of the active work averaged over
        a time `tau0' at a later time.

        Parameters
        ----------
        tau0 : int
            Number of consecutive individual active works on which to average
            it. (default: 1)
        n_max : int
            Maximum number of values at which to evaluate the correlation.
            (default: 100)
        int_max : int or None
            Maximum number of different intervals to consider in order to
            compute the mean which appears in the correlation expression.
            (default: None)
            NOTE: if int_max == None, then a maximum number of disjoint
                     intervals will be considered.
        min : int or None
            Minimum value at which to compute the correlation. (default: None)
            NOTE: if min == None then min = `tau0', otherwise the minimum of
                  `tau0' and `min' is taken.
        max : int or None
            Maximum value at which to compute the correlation. (default: None)
            NOTE: this value is passed as `max' to self.n.
        log : bool
            Logarithmically space values at which the correlations are
            computed. (default: True)

        Returns
        -------
        cor : (5, *) numpy array
            Array of:
                (0) value at which the correlation is computed,
                (1) mean of the computed correlation,
                (2) standard error of the computed correlation,
                (3) standard deviation of the active work,
                (4) standard deviation of the order parameter norm.
        """

        cor = []
        for n in self._n(n_max=n_max, log=log,
            min=(2*tau0 if min == None else np.max([tau0 + min, 2*tau0])),
            max=(None if max == None else tau0 + max)):
            ordersIni = (lambda l: l - np.mean(l))(
                list(map(
                    lambda t: self.getOrderParameter(t, norm=True),         # fluctuations of the active work averaged between t0 and t0 + tau0
                    self._time0(n, int_max=int_max))))
            worksFin = (lambda l: l - np.mean(l))(
                list(map(
                    lambda t: self.workArray[t + n - tau0:t + n].mean(),    # fluctuations of the active work averaged between t0 and t0 + tau0
                    self._time0(n, int_max=int_max))))
            workOrder = ordersIni*worksFin
            cor += [[n - tau0, *mean_sterr(workOrder),
                np.std(ordersIni), np.std(worksFin)]]

        return np.array(cor)

    def corOrderOrder(self,
        n_max=100, int_max=100, max=None, norm=False, log=False):
        """
        Compute autocorrelations of the fluctuations of the order parameter.

        Parameters
        ----------
        n_max : int
            Maximum number of values at which to evaluate the correlation.
            (default: 100)
        int_max : int
            Maximum number of different intervals to consider in order to
            compute the mean which appears in the correlation expression.
            (default: 100)
        max : int or None
            Maximum value at which to compute the correlation. (default: None)
            NOTE: if max == None, then max = self.frames - 1.
        norm : bool
            Consider the norm of the order parameter rather than its vector
            form. (default: False)
        log : bool
            Logarithmically space values at which the correlations are
            computed. (default: True)

        Returns
        -------
        cor : (3, *) numpy array
            Array of:
                (0) value at which the correlation is computed,
                (1) mean of the computed correlation,
                (2) standard error of the computed correlation.
        """

        if max == None: max = self.frames - 1

        if log: space = logspace
        else: space = linspace

        cor = []
        for tau in space(1, max, n_max):
            time0 = list(OrderedDict.fromkeys(np.linspace(
                self.skip*self.framesWork, self.frames - tau - 1, int_max,
                endpoint=True, dtype=int)))
            ordersIni = (lambda l: np.array(l) - np.mean(l, axis=0))(list(map(
                lambda t: self.getOrderParameter(t, norm=norm), time0)))
            ordersFin = (lambda l: np.array(l) - np.mean(l, axis=0))(list(map(
                lambda t: self.getOrderParameter(t + tau, norm=norm), time0)))
            orderOrder = list(map(lambda x, y: np.dot(x, y),
                *(ordersIni, ordersFin)))
            cor += [[tau, *mean_sterr(orderOrder)]]

        return np.array(cor)

    def getWorks(self, tau, n_max=100, init=None):
        """
        [DEPRECATED — Directly manipulate the array self.workArray now.]

        Returns array of normalised active works for periods of `tau' frames,
        with a maximum of `n_max', dumping `init' initial frames.

        Parameters
        ----------
        tau : int
            Number of consecutive individual active works on which to average
            it.
        n_max : int
            Maximum number of values at which to evaluate the active work.
            (default: 100)
        init : int or None
            Number of initial frames to dump. (default: None)
            NOTE: if init == None, then half of the frames of the simulation is
                  dumped.

        Returns
        -------
        activeWork : (*,) float numpy array
            Array of active works.
        """

        if init == None: init = int(self.frames/2)

        time0 = np.array(list(OrderedDict.fromkeys(
            np.linspace(init, self.frames - tau - 1, n_max,
                endpoint=True, dtype=int))))

        return np.array(list(map(
            lambda t0: self.getWork(t0, t0 + int(tau)),
            time0)))

    def _time0(self, n, int_max=None):
        """
        Returns list of initial times to coarse-grain the list of active work
        sums in packs of `n'.

        Parameters
        ----------
        n : int
            Size of the packs of "instantaneous" active work.
        int_max : int or None
            Maximum number of initial times to return. (default: None)
            NOTE: if int_max == None, a maximum number of them are returned.

        Returns
        -------
        time0 : (*,) int numpy array
            Array of initial times.
        """

        time0 = np.linspace(
            self.skip, self.numberWork, int((self.numberWork - self.skip)//n),
            endpoint=False, dtype=int)
        if int_max == None: return time0
        indexes = list(OrderedDict.fromkeys(
            np.linspace(0, time0.size, int_max, endpoint=False, dtype=int)))
        return np.array(itemgetter(*indexes)(time0), ndmin=1)

    def _n(self, n_max=100, min=None, max=None, log=False):
        """
        Returns integers linearly or logarithmically scaled between `min' or 1
        and `max' or int((self.numberWork - self.skip)/2) with `n_max' maximum
        of them.

        Parameters
        ----------
        n_max : int
            Maximum number of integers to return. (default: 100)
        min : int or None
            Minimum integer. (default: None)
            NOTE: if min == None, then min = 1.
        max : int or None
            Maximum integer. (default: None)
            NOTE: if max == None, then max =
                  int((self.numberWork - self.skip)/2).
        log : bool
            Logarithmically space integers. (default: False)

        Returns
        -------
        space : (*,) int numpy array
            Array of spaced integers.
        """

        if max == None: max = int((self.numberWork - self.skip)/2)
        if min == None: min = 1

        n_max = int(n_max)
        max = int(max)
        min = int(min)

        if log: space = logspace
        else: space = linspace

        return space(min, max, n_max)

    def _setWorkPart(self, workPart='all'):
        """
        Set part of the active work to consider in computations.

        Parameters
        ----------
        workPart : string
            Part of the active work to consider in computations:
                * 'all': active work,
                * 'force': force part of the active work,
                * 'orientation': orientation part of the active work,
                * 'noise': noise part of the active work.
            (default: 'all')
        """

        try: self.workArray = self.workDict[workPart]
        except KeyError: raise ValueError(
            'Part \'%s\' is not known.' % workPart)

class FittingWorkForceVariance(CurveFit):
    """
    Fit work force variance at different number of particles and persistence
    lengths.
    """

    def __init__(self, Fx, Fy, gamma=None):
        """
        Fit curve to data points.

        Parameters
        ----------
        Fx : float array-like
            Product of rotational diffusivity and lag time, D_r \\tau.
        Fy : float array-like
            Product of number of particles, square root of diffusivity, lag time
            and variance, N \\sqrt{D_r} \\tau <\\delta w_f^2(t_0; \\tau)>.
        gamma : float or None
            Arbitrarily fix the value of gamma. (default: None)
            NOTE: if gamma == None then its value is fitted.
        """

        self.Fx = np.array(Fx)
        self.Fy = np.array(Fy)

        if gamma is None:
            super().__init__(
                self._masterFunc,
                self.Fx, self.Fy,
                jac=self._masterJac)
            self.gamma = self.popt[1]
        else:
            super().__init__(
                lambda DrTau, A: self._masterFunc(DrTau, A, gamma),
                self.Fx, self.Fy,
                jac=lambda DrTau, A: self._masterJac(DrTau, A, gamma)[:1])
            self.gamma = gamma

        self.A = self.popt[0]

    def _masterFunc(self, DrTau, A, gamma):
        """
        Fitting function.

        Parameters
        ----------
        DrTau : float
            Product of rotational diffusivity and lag time.
        A : float
            Long-time variance parameter.
        gamma : float
            Time-rescaling parameter.

        Returns
        -------
        F : float
            Evaluated fitting function.
        """

        return (2*A/gamma)*(1 - (1/(gamma*DrTau))*(1 - np.exp(-gamma*DrTau)))

    def _masterJac(self, DrTau, A, gamma):
        """
        Jacobian matrix of fitting function with respect to parameters.

        Parameters
        ----------

        DrTau : float
            Product of rotational diffusivity and lag time.
        A : float
            Long-time variance parameter.
        gamma : float
            Time-rescaling parameter.

        Returns
        -------
        J : (2,) float Numpy array
            Evaluated Jacobian matrix.
        """

        return np.array([
            (2/gamma)*(1 - (1/(gamma*DrTau))*(1 - np.exp(-gamma*DrTau))),
            -(2*A/(gamma**2))*(1 + np.exp(-gamma*DrTau))
                + (2*A/((gamma**3)*DrTau))*(1 - np.exp(-gamma*DrTau))])

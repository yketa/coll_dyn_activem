"""
Module workorder provides classes to conjointly and analyse active work and
order parameter.
"""

import numpy as np

from operator import itemgetter

from coll_dyn_activem.work import ActiveWork
from coll_dyn_activem.maths import JointDistribution

class WorkOrder(ActiveWork):
    """
    Conjointly compute and analyse active work and order parameter.

    (see coll_dyn_activem.work.ActiveWork)
    (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#ABP%20work%20and%20order%20LDP)
    """

    def __init__(self, filename, workPart='all', skip=1):
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
        """

        super().__init__(filename, workPart=workPart, skip=skip)    # initialise with super class

    def nWorkOrder(self, n, int_max=None):
        """
        Returns normalised rate of active work and order parameter averaged on
        packs of size `n' of consecutive individual active works and order
        parameters.

        NOTE: Individual active work refers to the normalised rate of active
              work and order parameter on self.dumpPeriod*self.framesWork
              consecutive frames and stored as element of self.workArray and
              self.orderParameter.

        Parameters
        ----------
        n : int
            Size of packs on which to average active work and order parameter.
        int_max : int or None
            Maximum number of packs consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of packs.
                  int_max cannot exceed the maximum number of nonoverlapping
                  packs.

        Returns
        -------
        workAveraged : float numpy array
            Array of computed active works.
        orderAveraged : float numpy array
            Array of computed order parameters.
        """

        workAveraged = []
        orderAveraged = []
        for i in self._time0(n, int_max=int_max):
            workAveraged += [self.workArray[i:i + n].mean()]
            orderAveraged += [self.orderParameter[i:i + n].mean()]

        return np.array(workAveraged), np.array(orderAveraged)

    def SCGF(self, *s, n=1, int_max=None, percentageW=None):
        """
        Returns scaled cumulant generating function from active work averaged on
        packs of size `n' of consecutive individual measures at biasing
        parameter `s'.

        (see self._biasedAverages)

        Parameters
        ----------
        s : float
            Biasing parameter.
        n : int
            Size of packs on which to average active work.
            (default: 1)
        int_max : int or None
            Maximum number of packs consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of packs.
                  int_max cannot exceed the maximum number of nonoverlapping
                  packs.
        percentageW : float or None
            Remove from the SCGF the values corresponding to a work averafe
            which is in the lowest or highest `percentageW'% of the original
            active work array.
            (default: None)
            NOTE: If percentageW == None, this operation is not performed.

        Returns
        -------
        tau : float
            Averaging time in absolute dimensionless units.
        psi : float Numpy array
            Scaled cumulant generating function at `s'.
        """

        return self._biasedAverages(
            *s, n=n, int_max=int_max, percentageW=percentageW,
            returns=('tau', 'SCGF'))

    def sWork(self, *s, n=1, int_max=None, percentageW=None):
        """
        Returns averaged active work in biased ensemble from active work
        averaged on packs of size `n' of consecutive individual measures at
        biasing parameter `s'.

        (see self._biasedAverages)

        Parameters
        ----------
        s : float
            Biasing parameter.
        n : int
            Size of packs on which to average active work.
            (default: 1)
        int_max : int or None
            Maximum number of packs consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of packs.
                  int_max cannot exceed the maximum number of nonoverlapping
                  packs.
        percentageW : float or None
            Remove from the work averages the values corresponding to a work
            average which is in the lowest or highest `percentageW'% of the
            original active work array.
            (default: None)
            NOTE: If percentageW == None, this operation is not performed.

        Returns
        -------
        tau : float
            Averaging time in absolute dimensionless units.
        work : float Numpy array
            Averaged active work at `s'.
        """

        return self._biasedAverages(
            *s, n=n, int_max=int_max, percentageW=percentageW,
            returns=('tau', 'work'))

    def sOrder(self, *s, n=1, int_max=None, percentageW=None):
        """
        Returns averaged order parameter in biased ensemble from active work and
        order parameter averaged on packs of size `n' of consecutive individual
        measures at biasing parameter `s'.

        (see self._biasedAverages)

        Parameters
        ----------
        s : float
            Biasing parameter.
        n : int
            Size of packs on which to average active work and order parameter.
            (default: 1)
        int_max : int or None
            Maximum number of packs consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of packs.
                  int_max cannot exceed the maximum number of nonoverlapping
                  packs.
        percentageW : float or None
            Remove from the order averages the values corresponding to a work
            average which is in the lowest or highest `percentageW'% of the
            original active work array.
            (default: None)
            NOTE: If percentageW == None, this operation is not performed.

        Returns
        -------
        tau : float
            Averaging time in absolute dimensionless units.
        order : float Numpy array
            Averaged order parameter at `s'.
        """

        return self._biasedAverages(
            *s, n=n, int_max=int_max, percentageW=percentageW,
            returns=('tau', 'order'))

    def getHistogram3D(self, Nbins, n=1, int_max=None,
        work_min=None, work_max=None, order_min=None, order_max=None):
        """
        Returns 3D histogram of work and order.

        Parameters
        ----------
        Nbins : int or 2-uple-like of int
            Number of histogram bins for active work and order parameter.
        n : int
            Size of packs on which to average active work and order parameter.
        int_max : int or None
            Maximum number of packs consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of packs.
                  int_max cannot exceed the maximum number of nonoverlapping
                  packs.
        work_min : float or None
            Minimum value for the active work. (default: None)
            NOTE: if work_min == None, then the minimum value in the array is
                  taken.
        work_max : float or None
            Maximum value for the active work. (default: None)
            NOTE: if work_max == None, then the maximum value in the array is
                  taken.
        order_min : float or None
            Minimum value for the order parameter. (default: None)
            NOTE: if order_min == None, then the minimum value in the array is
                  taken.
        order_max : float or None
            Maximum value for the order parameter. (default: None)
            NOTE: if order_max == None, then the maximum value in the array is
                  taken.

        Returns
        -------
        hist : (Nbins.prod(), 3) float Numpy array
            Values of the histogram:
                (0) Active work bin.
                (1) Order parameter bin.
                (2) Proportion.
        """

        return JointDistribution(*self.nWorkOrder(n, int_max=int_max)).hist(
            Nbins,
            vmin1=work_min, vmax1=work_max, vmin2=order_min, vmax2=order_max)

    def getHistogram3DSC(self, n=1, int_max=None):
        """
        Returns 3D histogram computed via self-consistent density estimation.
        (see coll_dyn_activem.scde)

        Parameters
        ----------
        n : int
            Size of packs on which to average active work and order parameter.
        int_max : int or None
            Maximum number of packs consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of packs.
                  int_max cannot exceed the maximum number of nonoverlapping
                  packs.

        Returns
        -------
        hist : (*, 3) float Numpy array
            Values of the histogram:
                (0) Active work bin.
                (1) Order parameter bin.
                (2) Proportion.
            NOTE: This histogram is rather a probability density function,
                  therefore the integral over the bins is equal to 1 and thus
                  the values should be interpreted differently than a simple
                  proporition of observations.
        """

        return JointDistribution(*self.nWorkOrder(n, int_max=int_max)).pdf()

    def meanStdCor(self, n=1, int_max=None):
        """
        Returns means anf standard deviations of active work and order
        parameter, and their Pearson correlation coefficient.

        (see self._biasedAverages)

        Parameters
        ----------
        n : int
            Size of packs on which to average active work and order parameter.
        int_max : int or None
            Maximum number of packs consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of packs.
                  int_max cannot exceed the maximum number of nonoverlapping
                  packs.

        Returns
        -------
        meanWork : float
            Mean of active work.
        meanOrder : float
            Mean of order parameter.
        stdWork : float
            Standard deviation of active work.
        stdOrder : float
            Standard deviation of order parameter.
        corWorkOrder : float
            Pearson correlation coefficient of active work and order parameter.
        """

        return self._biasedAverages(
            n=n, int_max=int_max,
            returns=('meanStdCor',))

    def _biasedAverages(self, *s, n=1, int_max=None, percentageW=None,
        returns=('tau', 's', 'SCGF', 'work', 'order', 'meanStdCor')):
        """
        Returns scaled cumulant generating function, averaged active work and
        averaged order parameter, from active work and order parameter averaged
        on packs of size `n' of consecutive individual measures, at biasing
        parameter `s'.

        (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#ABP%20work%20and%20order%20LDP)

        NOTE: This big master function is designed to avoid computing several
              times the same work and order arrays, which is time consuming.

        Parameters
        ----------
        s : float
            Biasing parameter.
            NOTE: This is only relevant if computing the SCGF or work or order
                  averages.
        n : int
            Size of packs on which to average active work and order parameter.
            (default: 1)
        int_max : int or None
            Maximum number of packs consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of packs.
                  int_max cannot exceed the maximum number of nonoverlapping
                  packs.
        percentageW : float or None
            Remove from the biasing parameter list, the work averages and order
            averages the values corresponding to a work average which is in the
            lowest or highest `percentageW'% of the original active work array.
            (default: None)
            NOTE: If percentageW == None, this operation is not performed.
        returns : tuple-like of strings
            Quantities to return:
                'tau' : float
                    Dimensionless time over which the work and order are
                    averaged.
                's' : float Numpy array
                    Biasing parameters.
                'SCGF' : float Numpy array
                    Scaled cumulant generating function.
                'work' : float Numpy array
                    Averaged active work.
                'order' : float Numpy array
                    Averaged order parameter.
                'meanStdCor' : (5,) float tuple
                    Mean active work, mean order parameter, standard deviation
                    of active work, standard deviation of order parameter, work
                    and order correlation.
            (default: ('tau', 's', 'SCGF', 'work', 'order', 'meanStdCor'))
            NOTE: Quantities are returned as a tuple which follows the requested
                  order.

        Returns
        -------
        (according to `returns`)
        """

        out = {}

        out['s'] = np.array(s)

        out['tau'] = self._tau(n)

        # WORK AND ORDER ARRAYS

        if 'order' in returns or 'meanStdCor' in returns:
            workArray, orderArray = self.nWorkOrder(n, int_max=int_max)
        else:
            workArray = super().nWork(n, int_max=int_max)   # only computation of the work is needed

        if 'SCGF' in returns:   # scaled cumulant generating function

            out['SCGF'] = np.array(list(map(
                lambda _s:
                    np.log(np.mean(np.exp(-_s*out['tau']*self.N*workArray)))/(
                        out['tau']*self.N),
                s)))

        if 'work' in returns or percentageW != None:    # averaged active work in biased ensemble

            out['work'] = np.array(list(map(
                lambda _s: (
                    np.mean(
                        workArray*np.exp(-_s*out['tau']*self.N*workArray))/(
                    np.mean(np.exp(-_s*out['tau']*self.N*workArray)))),
                s)))

        if 'order' in returns:  # averaged order parameter in biased ensemble

            out['order'] = np.array(list(map(
                lambda _s: (
                    np.mean(
                        orderArray*np.exp(-_s*out['tau']*self.N*workArray))/(
                    np.mean(np.exp(-_s*out['tau']*self.N*workArray)))),
                s)))

        if 'meanStdCor' in returns: # means, standard deviations, and correlation of active work and order parameter in unbiased ensemble

            meanWork = workArray.mean()
            meanOrder = orderArray.mean()

            stdWork = workArray.std()
            stdOrder = orderArray.std()

            corWorkOrder = np.cov(
                np.stack((workArray, orderArray),
                    axis=0))[0, 1]/(stdWork*stdOrder)

            out['meanStdCor'] = (
                meanWork, meanOrder, stdWork, stdOrder, corWorkOrder)

        # CROP STATISTICALLY INSIGNIFICANT VALUES

        if percentageW != None:

            rangeS = (
                (out['work'] >= np.percentile(
                    workArray, percentageW, interpolation='higher'))
                *(out['work'] <= np.percentile(
                    workArray, 100 - percentageW, interpolation='lower')))

            out['s'] = out['s'][rangeS]

            if 'SCGF' in returns: out['SCGF'] = out['SCGF'][rangeS]
            if 'work' in returns: out['work'] = out['work'][rangeS]
            if 'order' in returns: out['order'] = out['order'][rangeS]

        # RETURNS

        return itemgetter(*returns)(out)

    def _tau(self, n):
        """
        Returns dimensionless time corresponding to `n' consecutive measures.

        Parameters
        ----------
        n : int
            Number of consecutive measures.

        Returns
        -------
        tau : float
            Corresponding dimensionless time.
        """

        return n*self.dt*self.dumpPeriod*self.framesWork

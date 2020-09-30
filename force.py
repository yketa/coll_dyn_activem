"""
Module force provides classes to compute and analyse forces autocorrelations and
correlations with orientation.
"""

import numpy as np

from collections import OrderedDict

from coll_dyn_activem.read import Dat
from coll_dyn_activem.maths import linspace, logspace, mean_sterr

class Force(Dat):
    """
    Compute and analyse force from simulation data.
    """

    def __init__(self, filename, skip=1):
        """
        Loads file.

        Parameters
        ----------
        filename : string
            Name of input data file.
        skip : int
            Skip the `skip' first computed values of the active work in the
            following calculations. (default: 1)
            NOTE: This can be changed at any time by setting self.skip.
        """

        super().__init__(filename, loadWork=False)  # initialise with super class

        self.skip = skip    # skip the `skip' first measurements of the active work in the analysis

    def getForce(self, time):
        """
        Returns forces exerted on every particles at time `time'.

        Parameters
        ----------
        time : int
            Index of the frame.

        Returns
        -------
        forces : (self.N, 2) float numpy array
            Forces exerted on every particles.
        """

        forces = np.full((self.N, 2), fill_value=0, dtype='float64')
        for i in range(self.N):
            for j in range(1, self.N):
                force = self._WCA(time, i, j)
                forces[i] += force
                forces[j] -= force

        return forces

    def corForceForce(self,
        n_max=100, int_max=None, min=1, max=None, log=False):
        """
        Returns fluctuations of the force autocorrelations.

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
            Minimum value at which to compute the correlation. (default: 1)
        max : int or None
            Maximum value at which to compute the correlation. (default: None)
            NOTE: if max == None, then n_max = self.frames - self.skip - 1.
        log : bool
            Logarithmically space values at which the correlations are
            computed. (default: False)

        Returns
        -------
        cor : (3, *) numpy array
            Array of:
                (0) value at which the correlation is computed,
                (1) mean of the computed correlation,
                (2) standard error of the computed correlation,
                (3) mean of the squared norm of the flucuations of the force.
        """

        min = 1 if min == None else int(min)
        max = self.frames - self.skip - 1 if max == None else int(max)
        int_max = ((self.frames - self.skip - 1)//max if int_max == None
            else int(int_max))

        if log: space = logspace
        else: space = linspace

        dt = space(min, max, n_max)
        time0 = linspace(self.skip, self.frames - max - 1, int_max)

        cor = []
        forcesIni = (
            (lambda l: (np.array(l)
                - np.mean(l, axis=1).reshape(time0.size, 1)).flatten())(    # fluctuations of the force scalar the orientation at t0
                list(map(
                    lambda t: self._ForceOrientation(t),
                    time0))))
        for tau in dt:
            forcesFin = (
                (lambda l:
                    np.array(l) - np.mean(l, axis=1).reshape(time0.size, 1, 2))(    # fluctuations of the force at t0 + tau
                list(map(lambda t: self.getForce(t + tau), time0)))).reshape(
                    (self.N*time0.size, 2))
            forcesForces = list(map(lambda x, y: np.dot(x, y),
                *(forcesIni, forcesFin)))
            forcesNormSq = (list(map(lambda x, y: np.dot(x, y),
                *(forcesIni, forcesIni)))
                + list(map(lambda x, y: np.dot(x, y),
                    *(forcesFin, forcesFin))))
            cor += [[tau, *mean_sterr(forcesForces), np.mean(forcesNormSq)]]

        return np.array(cor)

    def corForceVelocity(self,
        n_max=100, int_max=None, min=1, max=None, log=False):
        """
        Returns correlations of the scalar product of force and velocity.

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
            Minimum value at which to compute the correlation. (default: 1)
        max : int or None
            Maximum value at which to compute the correlation. (default: None)
            NOTE: if max == None, then n_max = self.frames - self.skip - 1.
        log : bool
            Logarithmically space values at which the correlations are
            computed. (default: False)

        Returns
        -------
        cor : (3, *) numpy array
            Array of:
                (0) value at which the correlation is computed,
                (1) mean of the computed correlation,
                (2) standard error of the computed correlation,
                (3) product of the standard deviations of the scalar product
                    at initial and final times.
        """

        min = 1 if min == None else int(min)
        max = self.frames - self.skip - 1 if max == None else int(max)
        int_max = ((self.frames - self.skip - 1)//max if int_max == None
            else int(int_max))

        if log: space = logspace
        else: space = linspace

        dt = space(min, max, n_max)
        time0 = linspace(self.skip, self.frames - max - 1, int_max)

        cor = []
        forcesVelocitesIni = (
            (lambda l:
                np.array(l) - np.mean(l, axis=1).reshape(time0.size, 1))(   # fluctuations of the scalar product of force and velocity at t0
                list(map(
                    lambda t: list(map(
                        lambda f, v: np.dot(f, v),
                        *(self.getForce(t),
                            self.getVelocities(t, norm=False)))),
                    time0)))).flatten()
        print(forcesVelocitesIni)
        for tau in dt:
            forcesVelocitesFin = (
                (lambda l:
                    np.array(l) - np.mean(l, axis=1).reshape(time0.size, 1))(   # fluctuations of the scalar product of force and velocity at t0 + tau
                    list(map(
                        lambda t: list(map(
                            lambda f, v: np.dot(f, v),
                            *(self.getForce(t),
                                self.getVelocities(t, norm=False)))),
                        time0 + tau)))).flatten()
            print(forcesVelocitesFin)
            cor += [[
                tau,
                *mean_sterr(forcesVelocitesIni*forcesVelocitesFin),
                forcesVelocitesIni.std()*forcesVelocitesFin.std()]]

        return np.array(cor)

    def corForceForceOrientation(self,
        n_max=100, int_max=None, min=1, max=None, log=False):
        """
        Returns fluctuations of the force scalar orientation autocorrelations.

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
            Minimum value at which to compute the correlation. (default: 1)
        max : int or None
            Maximum value at which to compute the correlation. (default: None)
            NOTE: if max == None, then n_max = self.frames - self.skip - 1.
        log : bool
            Logarithmically space values at which the correlations are
            computed. (default: False)

        Returns
        -------
        cor : (3, *) numpy array
            Array of:
                (0) value at which the correlation is computed,
                (1) mean of the computed correlation,
                (2) standard error of the computed correlation,
                (3) product of the standard deviations of the force scalar
                    orientation at initial and final times.
        """

        min = 1 if min == None else int(min)
        max = self.frames - self.skip - 1 if max == None else int(max)
        int_max = ((self.frames - self.skip - 1)//max if int_max == None
            else int(int_max))

        if log: space = logspace
        else: space = linspace

        dt = space(min, max, n_max)
        time0 = linspace(self.skip, self.frames - max - 1, int_max)

        cor = []
        forcesIni = (
            (lambda l: (np.array(l)
                - np.mean(l, axis=1).reshape(time0.size, 1)).flatten())(    # fluctuations of the force scalar the orientation at t0
                list(map(
                    lambda t: self._ForceOrientation(t),
                    time0)))).flatten()
        for tau in dt:
            forcesFin = (
                (lambda l: (np.array(l)
                    - np.mean(l, axis=1).reshape(time0.size, 1)).flatten())(    # fluctuations of the force scalar the orientation at t0 + tau
                    list(map(
                        lambda t: self._ForceOrientation(t + tau),
                        time0)))).flatten()
            forcesForces = forcesIni*forcesFin
            cor += [[tau, *mean_sterr(forcesForces),
                forcesIni.std()*forcesFin.std()]]

        return np.array(cor)

    def varForceOrientation(self, int_max=100):
        """
        Returns variance of the scalar product of force and particle direction.

        Parameters
        ----------
        int_max : float
            Number of times at which to compute the scalar product of force
            and particle direction. (default: 100)

        Returns
        -------
        var : float
            Computed variance.
        """

        time0 = np.array(list(OrderedDict.fromkeys(
            np.linspace(
                self.skip, self.frames - 1, int(int_max),
                endpoint=False, dtype=int))))

        forceOrientation = np.array(list(map(
            self._ForceOrientation, time0)))

        return forceOrientation.var()

    def _ForceOrientation(self, time):
        """
        Returns scalar products of force and particle directions at a given
        time.

        Parameters
        ----------
        time : int
            Frame index.

        Returns
        -------
        forceOrientation : (self.N,) float numpy array
            Array of scalar products.
        """

        return np.array(list(map(
            lambda x, y: np.dot(x, y),
            *(self.getForce(time),
                self.getDirections(time)))))

    def _WCA(self, time, particle0, particle1):
        """
        Returns force derived from WCA potential applied on `particle0' by
        `particle1' at time `time'.

        Parameters
        ----------
        time : int
            Index of the frame.
        particle0 : int
            Index of the first particle.
        particle1 : int
            Index of the second particle.

        Returns
        -------
        force : (2,) float numpy Array
        """

        force = np.array([0, 0])

        if particle0 == particle1: return force # same particle

        dist, pos0, pos1 = self.getDistancePositions(time, particle0, particle1)

        sigma = (self.diameters[particle0] + self.diameters[particle1])/2

        if dist/sigma >= 2**(1./6.): return force # distance greater than cut-off

        force = ((48/((dist/sigma)**14) - 24/((dist/sigma)**8))/(sigma**2))*(
            np.array([
                self._diffPeriodic(pos1[0], pos0[0]),
                self._diffPeriodic(pos1[1], pos0[1])]))
        return force

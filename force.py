"""
Module force provides classes to compute and analyse forces autocorrelations and
correlations with orientation.
"""

import numpy as np

from collections import OrderedDict
from operator import itemgetter

from coll_dyn_activem.read import Dat
from coll_dyn_activem.maths import linspace, logspace, mean_sterr, wo_mean,\
    normalise1D
from coll_dyn_activem._pycpp import getWCA, getWCAForces, getRAForces,\
    getRAHessian

class Force(Dat):
    """
    Compute and analyse force from simulation data.
    """

    def __init__(self, filename, skip=1, from_velocity=True, corruption=None):
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
            NOTE: This does not apply to .datN files.
        from_velocity : bool
            Use the velocity as a proxy to the force by substracting it the
            self-propulsion (this is exact if there is no translational noise).
            (default: True)
        corruption : str or None
            Pass corruption test for given file type (see
            coll_dyn_activem.read.Dat). (default: None)
            NOTE: if corruption == None, then the file has to pass corruption
                  tests.
        """

        super().__init__(filename, loadWork=False, corruption=corruption)   # initialise with super class

        self.skip = skip                    # skip the `skip' first measurements of the active work in the analysis
        self.from_velocity = from_velocity  # compute force from velocity

    def getForce(self, time, *particle, norm=False, Heun=False):
        """
        Returns WCA forces exerted on particles at time `time'.

        Parameters
        ----------
        time : int
            Index of the frame.
        particle : int
            Indexes of particles.
            NOTE: if none is given, then all particles are returned.
        norm : bool
            Return norm of forces rather than 2D force. (default: False)
        Heun : bool
            Compute correction to forces from Heun integration. (default: False)
            NOTE: Translational noise is neglected.

        Returns
        -------
        forces : [not(norm)] (self.N, 2) float numpy array
                 [norm] (self.N,) float numpy array
            Forces exerted on particles.
        """

        if particle == (): particle = range(self.N)

        if self.from_velocity:
            forces = self.getVelocities(time) - self.getPropulsions(time)
        else:
            forces = self.epsilon*getWCAForces(
                self.getPositions(time), self.diameters, self.L)

        if Heun:    # Heun correction

            newPositions = self.getPositions(time) + self.dt*(
                self.getPropulsions(time, norm=False) + forces)
            newForces = self.epsilon*getWCAForces(
                newPositions, self.diameters, self.L)

            forces = (forces + newForces)/2

        forces = np.array(itemgetter(*particle)(forces))

        if norm: return np.sqrt((forces**2).sum(axis=-1))
        return forces

    def getRAForce(self, time, *particle, norm=False, a=12, rcut=1.25):
        """
        Return regularised 1/r^a forces exerted on particles at time `time'.

        Parameters
        ----------
        time : int
            Index of the frame.
        particle : int
            Indexes of particles.
            NOTE: if none is given, then all particles are returned.
        norm : bool
            Return norm of forces rather than 2D force. (default: False)
        a : float
            Potential exponent. (default: 12)
        rcut : float
            Potential cut-off radius. (default: 1.25)

        Returns
        -------
        forces : [not(norm)] (self.N, 2) float numpy array
                 [norm] (self.N,) float numpy array
            Forces exerted on particles.
        """

        if particle == (): particle = range(self.N)
        f = self.epsilon*np.array(itemgetter(*particle)(
            getRAForces(
                self.getPositions(time), self.diameters, self.L, a, rcut)))

        if norm: return np.sqrt((f**2).sum(axis=-1))
        return f

    def getRAHessian(self, time, a=12, rcut=1.25):
        """
        Return regularised 1/r^a-potential Hessian matrix at time `time'.

        Parameters
        ----------
        time : int
            Index of the frame.
        a : float
            Potential exponent. (default: 12)
        rcut : float
            Potential cut-off radius. (default: 1.25)

        Returns
        -------
        forces : (2*self.N, 2*self.N)
            Hessian matrix.
        """

        return self.epsilon*getRAHessian(
            self.getPositions(time), self.diameters, self.L,
            a=a, rcut=rcut)

    def getRAResidualForce(self, time0, time1, a=12, rcut=1.25):
        """
        Return residual force between ADD inherent states using regularised
        1/r^a-potential Hessian matrix at time `time'.

        Parameters
        ----------
        time0 : int
            Initial frame.
        time1 : int
            Final frame.
        a : float
            Potential exponent. (default: 12)
        rcut : float
            Potential cut-off radius. (default: 1.25)

        Returns
        -------
        r_forces : (self.N, 2)
            Residual forces.
        """

        disp = self.getDisplacements(time0, time1, remove_cm=True)
        d_prop = wo_mean(
            self.getPropulsions(time1) - self.getPropulsions(time0))
        hessian = self.getRAHessian(time0, a=12, rcut=1.25)

        return (
            np.reshape(np.matmul(hessian, disp.flatten()), (self.N, 2))
            - d_prop)

    def getPotential(self, time):
        """
        Compute WCA potential.

        Parameters
        ----------
        time : int
            Index of the frame.

        Returns
        -------
        U : float
            WCA potential energy.
        """

        return getWCA(self.getPositions(time), self.diameters, self.L)

    def corForce(self,
        n_max=100, int_max=None, min=None, max=None, log=False, Heun=False):
        """
        Returns correlations of the force fluctuations.

        Parameters
        ----------
        n_max : int
            Maximum number of lag times. (default: 100)
        int_max : int or None
            Maximum number of different initial times. (default: None)
            NOTE: if int_max == None, then a maximum number of (disjoint)
                  intervals will be considered.
        min : int or None
            Minimum lag time. (default: None)
            NOTE: if min == None, then min = 1.
        max : int or None
            Maximum lag time. (default: None)
            NOTE: if max == None, then max is taken to be the maximum according
                  to the choice of int_max.
        log : bool
            Logarithmically spaced lag times. (default: False)
            NOTE: This does not apply to .datN files.
        Heun : bool
            Compute correction to forces from Heun integration. (default: False)
            NOTE: Translational noise is neglected.

        Returns
        -------
        cor : (*, 3) float numpy array
            Array of:
                (0) value at which the correlation is computed,
                (1) mean of the computed correlation,
                (2) standard error of the computed correlation.
        F0sq : (**,) float numpy array
            Variance of forces at initial times.
        """

        time0, dt = self._dt(
            n_max=n_max, int_max=int_max, min=min, max=max, log=log)

        cor = []
        forcesIni = np.array(list(map(                                  # fluctuations at initial times
            lambda t: wo_mean(self.getForce(t, Heun=Heun), axis=-2),
            time0)))
        F0sq = (forcesIni**2).sum(axis=-1).mean(axis=-1)                # average over particles
        for tau in dt:
            forcesFin = np.array(list(map(                              # fluctuations at initial times + lag time
                lambda t: wo_mean(self.getForce(t + tau, Heun=Heun), axis=-2),
                time0)))
            qProd = (forcesIni*forcesFin).sum(axis=-1).mean(axis=-1)    # average over particles
            cor += [[tau, *mean_sterr(qProd/F0sq)]]                     # average over times

        return np.array(cor), F0sq

    def corForceDotVelocity(self,
        n_max=100, int_max=None, min=None, max=None, log=False, Heun=False):
        """
        Returns correlations of the scalar product of force and velocity
        fluctuations.

        Parameters
        ----------
        n_max : int
            Maximum number of lag times. (default: 100)
        int_max : int or None
            Maximum number of different initial times. (default: None)
            NOTE: if int_max == None, then a maximum number of (disjoint)
                  intervals will be considered.
        min : int or None
            Minimum lag time. (default: None)
            NOTE: if min == None, then min = 1.
        max : int or None
            Maximum lag time. (default: None)
            NOTE: if max == None, then max is taken to be the maximum according
                  to the choice of int_max.
        log : bool
            Logarithmically spaced lag times. (default: False)
            NOTE: This does not apply to .datN files.
        Heun : bool
            Compute correction to forces from Heun integration. (default: False)
            NOTE: Translational noise is neglected.

        Returns
        -------
        cor : (*, 3) numpy array
            Array of:
                (0) value at which the correlation is computed,
                (1) mean of the computed correlation,
                (2) standard error of the computed correlation.
        Fv0sq : (**,) float numpy array
            Variance of the scalar product of force and velocity at initial
            times.
        """

        time0, dt = self._dt(
            n_max=n_max, int_max=int_max, min=min, max=max, log=log)

        cor = []
        forcesVelocitesIni = np.array(list(map(                             # fluctuations at initial times
            lambda t: wo_mean(
                (self.getForce(t, Heun=Heun)
                    *self.getVelocities(t, norm=False)).sum(axis=-1),
                axis=-1),
            time0)))
        Fv0sq = (forcesVelocitesIni**2).mean(axis=-1)                       # average over particles
        for tau in dt:
            forcesVelocitesFin = np.array(list(map(                         # fluctuations at initial times + lag time
                lambda t: wo_mean(
                    (self.getForce(t + tau, Heun=Heun)
                        *self.getVelocities(t + tau, norm=False)).sum(axis=-1),
                    axis=-1),
                time0)))
            qProd = (forcesVelocitesIni*forcesVelocitesFin).mean(axis=-1)   # average over particles
            cor += [[tau, *mean_sterr(qProd/Fv0sq)]]                        # average over times

        return np.array(cor), Fv0sq

    def corForceDotDirection(self,
        n_max=100, int_max=None, min=None, max=None, log=False):
        """
        Returns correlations of the scalar product of force and self-propulsion
        direction fluctuations.

        Parameters
        ----------
        n_max : int
            Maximum number of lag times. (default: 100)
        int_max : int or None
            Maximum number of different initial times. (default: None)
            NOTE: if int_max == None, then a maximum number of (disjoint)
                  intervals will be considered.
        min : int or None
            Minimum lag time. (default: None)
            NOTE: if min == None, then min = 1.
        max : int or None
            Maximum lag time. (default: None)
            NOTE: if max == None, then max is taken to be the maximum according
                  to the choice of int_max.
        log : bool
            Logarithmically spaced lag times. (default: False)
            NOTE: This does not apply to .datN files.

        Returns
        -------
        cor : (*, 3) numpy array
            Array of:
                (0) value at which the correlation is computed,
                (1) mean of the computed correlation,
                (2) standard error of the computed correlation.
        Fu0sq : (**,) float numpy array
            Variance of the scalar product of force and self-propulsion
            direction at initial times.
        """

        time0, dt = self._dt(
            n_max=n_max, int_max=int_max, min=min, max=max, log=log)

        cor = []
        forcesDirIni = np.array(list(map(                       # fluctuations at initial times
            lambda t: wo_mean(self._ForceOrientation(t), axis=-1),
            time0)))
        Fu0sq = (forcesDirIni**2).mean(axis=-1)                 # average over particles
        for tau in dt:
            forcesDirFin = np.array(list(map(                   # fluctuations at initial times + lag time
                lambda t: wo_mean(self._ForceOrientation(t + tau), axis=-1),
                time0)))
            qProd = (forcesDirIni*forcesDirFin).mean(axis=-1)   # average over particles
            cor += [[tau, *mean_sterr(qProd/Fu0sq)]]            # average over times

        return np.array(cor), Fu0sq

    def corPropulsionForce(self,
        n_max=100, int_max=None, min=None, max=None, log=False, Heun=False):
        """
        Returns correlations between (i) the fluctuations of the propulsion at
        initial times and the fluctuations of the force at later times, and (ii)
        the fluctuations of the force at initial times and the fluctations of
        the propulsion at later times.

        Parameters
        ----------
        n_max : int
            Maximum number of lag times. (default: 100)
        int_max : int or None
            Maximum number of different initial times. (default: None)
            NOTE: if int_max == None, then a maximum number of (disjoint)
                  intervals will be considered.
        min : int or None
            Minimum lag time. (default: None)
            NOTE: if min == None, then min = 1.
        max : int or None
            Maximum lag time. (default: None)
            NOTE: if max == None, then max is taken to be the maximum according
                  to the choice of int_max.
        log : bool
            Logarithmically spaced lag times. (default: False)
            NOTE: This does not apply to .datN files.
        Heun : bool
            Compute correction to forces from Heun integration. (default: False)
            NOTE: Translational noise is neglected.

        Returns
        -------
        corPF : (*, 3) numpy array
            Array of:
                (0) value at which the correlation (i) is computed,
                (1) mean of the computed correlation (i),
                (2) standard error of the computed correlation (i).
        corFP : (*, 3) numpy array
            Array of:
                (0) value at which the correlation (ii) is computed,
                (1) mean of the computed correlation (ii),
                (2) standard error of the computed correlation (ii).
        pF0sq : (**,) float numpy array
            Product of the standard deviations of the propulsion and the force
            at initial times.
        """

        time0, dt = self._dt(
            n_max=n_max, int_max=int_max, min=min, max=max, log=log)

        corPF, corFP = [], []
        propIni = np.array(list(map(                                # fluctuations of the propulsion at initial times
            lambda t: wo_mean(self.getPropulsions(t, norm=False), axis=-2),
            time0)))
        forcesIni = np.array(list(map(                              # fluctuations of the force at initial times
            lambda t: wo_mean(self.getForce(t, Heun=Heun), axis=-2),
            time0)))
        pF0sq = np.sqrt(
            (propIni**2).sum(axis=-1).mean(axis=-1)                 # average over particles
            *(forcesIni**2).sum(axis=-1).mean(axis=-1))             # average over particles
        for tau in dt:
            propFin = np.array(list(map(                            # fluctuations of the propulsion at initial times + lag time
                lambda t: wo_mean(
                    self.getPropulsions(t + tau, norm=False),
                    axis=-2),
                time0)))
            forcesFin = np.array(list(map(                          # fluctuations of the force at initial times + lag time
                lambda t: wo_mean(self.getForce(t + tau, Heun=Heun), axis=-2),
                time0)))
            qProd = (propIni*forcesFin).sum(axis=-1).mean(axis=-1)  # average over particles
            corPF += [[tau, *mean_sterr(qProd/pF0sq)]]              # average over times
            qProd = (forcesIni*propFin).sum(axis=-1).mean(axis=-1)  # average over particles
            corFP += [[tau, *mean_sterr(qProd/pF0sq)]]              # average over times

        return np.array(corPF), np.array(corFP), pF0sq

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

        if self._type == 'datN':
            time0 = self.time0
        else:
            time0 = np.array(range(self.skip, self.frames))
        if int_max != None:
            indexes = list(OrderedDict.fromkeys(
                np.linspace(0, time0.size, int_max, endpoint=False, dtype=int)))
            time0 = np.array(itemgetter(*indexes)(time0), ndmin=1)

        forceOrientation = np.array(list(map(self._ForceOrientation, time0)))

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

        return self._ForcePropulsion(time)[1]

    def _ForcePropulsion(self, time, Heun=False):
        """
        Returns scalar products of force and (i) particle propulsions and (ii)
        particle directions, at a given time.

        Parameters
        ----------
        time : int
            Frame index.
        Heun : bool
            Compute correction to forces from Heun integration. (default: False)
            NOTE: Translational noise is neglected.

        Returns
        -------
        forcePropulsion : (self.N,) float numpy array
            Array of (i).
        forceOrientation : (self.N,) float numpy array
            Array of (ii).
        """

        force = wo_mean(self.getForce(time, Heun=Heun))
        propulsion = wo_mean(self.getPropulsions(time))

        forcePropulsion = (force*propulsion
            ).sum(axis=-1)
        forceOrientation = (force*np.array(list(map(normalise1D, propulsion)))
            ).sum(axis=-1)

        return forcePropulsion, forceOrientation

    def _WCA(self, time, particle0, particle1, positions=None):
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
        positions : (self.N, 2) float array-like or None
            Custom positions from which to compute distances. (default: None)
            NOTE: if positions == None, actual positions of particles at `time'
                  are considered.

        Returns
        -------
        force : (2,) float numpy Array
        """

        force = np.array([0, 0])

        if particle0 == particle1: return force # same particle

        dist, pos0, pos1 = self.getDistancePositions(
            time, particle0, particle1, positions=positions)

        sigma = (self.diameters[particle0] + self.diameters[particle1])/2

        if dist/sigma >= 2**(1./6.): return force # distance greater than cut-off

        force = ((48/((dist/sigma)**14) - 24/((dist/sigma)**8))/(sigma**2))*(
            np.array([
                self._diffPeriodic(pos1[0], pos0[0]),
                self._diffPeriodic(pos1[1], pos0[1])]))
        return self.epsilon*force

    def _1ra(self, time, particle0, particle1, positions=None, a=12, rcut=1.25):
        """
        Returns force derived from regularised 1/r^a potential applied on
        `particle0' by `particle1' at time `time'.

        Parameters
        ----------
        time : int
            Index of the frame.
        particle0 : int
            Index of the first particle.
        particle1 : int
            Index of the second particle.
        positions : (self.N, 2) float array-like or None
            Custom positions from which to compute distances. (default: None)
            NOTE: if positions == None, actual positions of particles at `time'
                  are considered.
        a : float
            Inverse power of the potential. (default: 12)
        rcut : float
            Potential cut-off radius. (default: 1.25)

        Returns
        -------
        force : (2,) float numpy Array
        """

        force = np.array([0, 0])

        if particle0 == particle1: return force # same particle

        dist, pos0, pos1 = self.getDistancePositions(
            time, particle0, particle1, positions=positions)

        sigma = ((self.diameters[particle0] + self.diameters[particle1])/2
            *(1 - 0.2*
                np.abs(self.diameters[particle0] - self.diameters[particle1])))

        # c0 = -(8 + a*(a + 6))/(8*(rcut**a))
        c1 = (a*(a + 4))/(4*(rcut**(a + 2)))
        c2 = -(a*(a + 2))/(8*(rcut**(a + 4)))

        if dist/sigma >= rcut: return force # distance greater than cut-off

        force = (a*((sigma/dist)**a)
            - 2*c1*((dist/sigma)**2) - 4*c2*((dist/sigma)**4))*(
                np.array([
                    self._diffPeriodic(pos1[0], pos0[0]),
                    self._diffPeriodic(pos1[1], pos0[1])]))/(dist**2)
        return self.epsilon*force

    def _dt(self, n_max=100, int_max=None, min=None, max=None, log=False):
        """
        Returns initial times and lag times for intervals of time between `min'
        and `max'.

        Parameters
        ----------
        n_max : int
            Maximum number of lag times. (default: 100)
        int_max : int or None
            Maximum number of different initial times. (default: None)
            NOTE: if int_max == None, then a maximum number of (disjoint)
                  intervals will be considered.
        min : int or None
            Minimum lag time. (default: None)
            NOTE: if min == None, then min = 1.
        max : int or None
            Maximum lag time. (default: None)
            NOTE: if max == None, then max is taken to be the maximum according
                  to the choice of int_max.
        log : bool
            Logarithmically spaced lag times. (default: False)
            NOTE: This does not apply to .datN files.

        Returns
        -------
        time0 : (*,) int numpy array
            Array of initial times.
        dt : (**,) int numpy array
            Array of lag times.
        """

        min = 1 if min == None else int(min)
        if self._type == 'datN':
            max = self.deltat.max() if max == None else int(max)
            dt = self.deltat[(self.deltat >= min)*(self.deltat <= max)]
            dt = itemgetter(*linspace(0, len(dt) - 1, n_max, endpoint=True))(dt)
            time0  = self.time0
        else:
            max = self.frames - self.skip - 1 if max == None else int(max)
            if log: space = logspace
            else: space = linspace
            dt = space(min, max, n_max)
            time0 = np.array(range(self.skip, self.frames - max))
            int_max = ((self.frames - self.skip - 1)//max if int_max == None
                else int(int_max))
        if int_max != None:
            indexes = list(OrderedDict.fromkeys(
                np.linspace(0, time0.size, int_max, endpoint=False, dtype=int)))
            time0 = np.array(itemgetter(*indexes)(time0), ndmin=1)

        return np.array(time0), np.array(dt)

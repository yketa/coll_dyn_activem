"""
Module flow provides classes to compute and analyse positions to characterise
the structure of systems of ABPs.

(see https://yketa.github.io/PhD_Wiki/#Structure%20characteristics)
(see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#ABP%20structure%20characteristics)
"""

from coll_dyn_activem.read import Dat
from coll_dyn_activem.maths import pycpp, g2Dto1D, wave_vectors_2D, DictList,\
    angle, linspace, Histogram, mean_sterr, relative_positions

import numpy as np

from operator import itemgetter

from multiprocessing import Pool

from freud.locality import Voronoi
from freud.box import Box

class Positions(Dat):
    """
    Compute and analyse positions from simulation data.

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

    def getParticleDensity(self, time, nBoxes=None):
        """
        Returns particle density at `time' as grid where each box is equal to
        the number of particles in the corresponding region of space divided by
        the surface of this region.

        Parameters
        ----------
        time : int
            Frame index.
        nBoxes : int
            Number of grid boxes in each direction. (default: None)
            NOTE: if nBoxes == None, then nBoxes = int(sqrt(self.N)).

        Returns
        -------
        rho : (nBoxes, nBoxes) float Numpy array
            Particle density grid.
        """

        time = int(time)

        if nBoxes == None: nBoxes = np.sqrt(self.N)
        nBoxes = int(nBoxes)
        dV = (self.L/nBoxes)**2

        return self.toGrid(time,
            np.full((self.N,), fill_value=1),
            nBoxes=nBoxes, box_size=self.L, centre=(0, 0), average=False)/dV

    def getLocalDensity(self, time, nBoxes=None):
        """
        Returns local packing fraction defined as the ratio of particles'
        volume to the volume of the corresponding box, where the system has been
        divided in `nBoxes'x`nBoxes' linearly spaced square boxes of identical
        size.

        NOTE: particle volumes are computed with a factor 2^(1/6) on diameters.

        Parameters
        ----------
        time : int
            Frame index.
        nBoxes : int
            Number of grid boxes in each direction. (default: None)
            NOTE: if nBoxes == None, then nBoxes = int(sqrt(self.N)).

        Returns
        -------
        localPhi : (nBoxes**2,) float numpy array
            Array of local packing fraction.
        """

        time = int(time)

        if nBoxes == None: nBoxes = np.sqrt(self.N)
        nBoxes = int(nBoxes)
        dV = (self.L/nBoxes)**2

        surfaces = self.toGrid(time,
            (np.pi/4.)*(((2**(1./6.))*self.diameters)**2),
            nBoxes=nBoxes, box_size=self.L, centre=(0, 0), average=False)
        return surfaces.flatten()/dV

    def getLocalParticleDensity(self, time, a):
        """
        Returns local packing fraction for each particle defined as the ratio of
        particles' volume to the volume of the square box of size `a' around
        each particle.

        NOTE: particle volumes are computed with a factor 2^(1/6) on diameters.

        Parameters
        ----------
        time : int
            Frame index.
        a : float
            Size of the box in which to compute densities.

        Returns
        -------
        localPhi : (self.N,) float numpy array
            Array of local packing fraction.
        """

        # positions = self.getPositions(time)
        #
        # surfaces = np.zeros((self.N,))
        # for particle in range(self.N):
        #     surfaces[particle] = ((np.pi/4.)*(((2**(1./6.))*self.diameters[
        #         (np.abs(
        #             relative_positions(positions, positions[particle], self.L))
        #         < a/2).all(axis=-1)])**2)).sum()
        #
        # return surfaces/(a**2)

        return pycpp.getLocalParticleDensity(
            a, self.getPositions(time), self.L, self.diameters)

    def getLocalDensityVoronoi(self, time, phi=True):
        """
        Returns local packing fraction defined as the ratio of particles'
        volume to the volume of the corresponding voronoi cells.

        NOTE: particle volumes are computed with a factor 2^(1/6) on diameters.

        (see self._voronoi)
        (see https://yketa.github.io/PhD_Wiki/#Structure%20characteristics)

        Parameters
        ----------
        time : int
            Frame index.
        phi : bool
            Returns local packing fraction instead of inverse local volume.
            (default: True)

        Returns
        -------
        localPhi : (self.N,) float numpy array
            Array of local packing fraction.
        """

        volumes = self._voronoi(time).volumes
        if phi: return ((np.pi/4.)*(((2**(1./6.))*self.diameters)**2))/volumes
        else: return 1./volumes

    def getNeighbourList(self, time):
        """
        Get list of neighbours and bond length for particles in the system at
        `time' from Voronoi tesselation.

        (see self._voronoi)

        Parameters
        ----------
        time : int
            Frame index.

        Returns
        -------
        neighbours : coll_dyn_activem.maths.DictList
            Neighbour list :
            (key) particle index,
            (0)   neighbour index,
            (1)   neighbour distance.
        """

        neighbours = DictList()
        voro = self._voronoi(time)
        for ((i, j), d) in zip(voro.nlist[:], voro.nlist.distances):
            neighbours[i] += [[j, d]]

        return neighbours

    def getBondOrderParameter(self, time, *particle):
        """
        Get bond orientational order parameter at `time'.

        (see https://yketa.github.io/PhD_Wiki/#Structure%20characteristics)

        Parameters
        ----------
        time : int
            Frame index.
        particle : int
            Indexes of particles.
            NOTE: if none is given, then all particles are returned.

        Returns
        -------
        psi : (self.N,) float numpy array
            Bond orientational order parameter.
        """

        neighbours = self.getNeighbourList(time)
        positions = self.getPositions(time)
        psi = np.zeros((self.N,), dtype=complex)
        for i in range(self.N):
            for j, _ in neighbours[i]:
                psi[i] += np.exp(1j*6*angle(
                    self._diffPeriodic(positions[i][0], positions[j][0]),
                    self._diffPeriodic(positions[i][1], positions[j][1])))
            psi[i] /= len(neighbours[i])

        if particle == (): return psi
        return np.array(itemgetter(*particle)(psi))

    def nPositions(self, int_max=None):
        """
        Returns array of positions.

        Parameters
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.

        Returns
        -------
        positions : (*, self.N) float numpy array
            Array of computed positions.
        """

        return np.array(list(map(
            lambda time0: self.getPositions(time0),
            self._time0(int_max=int_max))))

    def nParticleDensity(self, int_max=None, nBoxes=None):
        """
        Returns array of particle density as grids.

        Parameters
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.
        nBoxes : int
            Number of grid boxes in each direction. (default: None)
            NOTE: if nBoxes==None, then None is passed to
                  self.getParticleDensity.

        Returns
        -------
        rho : (*, nBoxes, nBoxes) float numpy array
            Array of particle density grids.
        """

        return np.array(list(map(
            lambda time0: self.getParticleDensity(time0, nBoxes=nBoxes),
            self._time0(int_max=int_max))))

    def nDistances(self, int_max=None, scale_diameter=False):
        """
        Returns distances between the particles of the system.

        Parameters
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.
        scale_diameter : bool
            Divide the distance between pairs of particles by the sum of the
            radii of the particles in the pair. (default: False)

        Returns
        -------
        distances : (*, self.N(self.N - 1)/2) float Numpy array
            Array of computed distances.
        """

        return np.array(
            [pycpp.getDistances(self.getPositions(t), self.L,
                diameters=(self.diameters if scale_diameter else None))
            for t in self._time0(int_max=int_max)])

    def structureFactor(self, nBins, kmin=None, kmax=None,
        int_max=None, nBoxes=None):
        """
        Returns static structure factor averaged along directions of space
        (assuming isotropy) as a histogram.

        Parameters
        ----------
        nBins : int
            Number of histogram bins.
        kmin : float or None
            Minimum (included) wavevector norm in the histogram. (default: None)
            NOTE: if kmin == None then None is passed to pycpp.g2Dto1Dgridhist.
        kmax : float or None
            Maximum (excluded) wavevector norm in the histogram. (default: None)
            NOTE: if kmax == None then None is passed to pycpp.g2Dto1Dgridhist.
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.
        nBoxes : int
            Number of grid boxes in each direction. (default: None)
            NOTE: if nBoxes==None, then nBoxes = int(sqrt(self.N)).

        Returns
        -------
        S : (*, 3) float Numpy array
            Array of (k, S(k), Sstd(k)) with S(k) the cylindrically averaged
            structure factor at minimum wavevector k of corresponding bin, and
            Sstd(k) the standard deviation on this measure.
        """

        particleDensity = self.nParticleDensity(int_max=int_max, nBoxes=nBoxes)
        nBoxes = particleDensity.shape[1]

        # _S2D = np.array(list(map(
        #     lambda _rho:
        #         (lambda FFT: np.real(np.conj(FFT)*FFT))
        #             (np.fft.fft2(_rho)),
        #     particleDensity)))/self.N
        S2D = np.zeros((nBoxes, nBoxes))
        for rho in particleDensity:
            S2D += (lambda FFT: np.real(np.conj(FFT)*FFT))(np.fft.fft2(rho))/(
                self.N*len(particleDensity))

        k2D = np.sqrt(
            (wave_vectors_2D(nBoxes, nBoxes, self.L/nBoxes)**2).sum(axis=-1))

        # S = pycpp.g2Dto1Dgridhist(_S2D.mean(axis=0), k2D, nBins,
        S = pycpp.g2Dto1Dgridhist(S2D, k2D, nBins,
            vmin=kmin, vmax=kmax)
        S[:, 2] /= np.sqrt(len(particleDensity))    # change standard deviation to standard error
        return S

    def densityCorrelation(self, int_max=None, nBoxes=None):
        """
        Returns particle spacial density averaged along directions of space
        (assuming isotropy).

        NOTE: Correlations are computed with FFT.
              (see https://yketa.github.io/PhD_Wiki/#Field%20correlation)

        Parameters
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.
        nBoxes : int
            Number of grid boxes in each direction. (default: None)
            NOTE: if nBoxes==None, then None is passed to
                  self.getParticleDensity.

        Returns
        -------
        G : float Numpy array
            Array of (r, G(r)) with G(r) the averaged density correlation at
            radius r.
        """

        particleDensity = self.nParticleDensity(int_max=int_max, nBoxes=nBoxes)
        nBoxes = particleDensity.shape[1]

        _G2D = np.array(list(map(
            lambda _rho:
                (lambda G2D: G2D*(self.rho/(G2D[0, 0]*((self.L/nBoxes)**2))))(
                    (lambda FFT: np.real(np.fft.ifft2(np.conj(FFT)*FFT)))
                        (np.fft.fft2(_rho - self.N/(nBoxes**2)))),
            particleDensity)))

        return g2Dto1D(_G2D.mean(axis=0), self.L)

    def pairDistribution(self, Nbins, min=None, max=None, int_max=None,
        scale_diameter=False):
        """
        Returns pair distribution function as an histogram of distances between
        pairs of particles.

        Parameters
        ----------
        Nbins : int
            Number of histogram bins.
        min : float or None
            Minimum included value for histogram bins. (default: None)
            NOTE: if min == None then 0 is taken.
            NOTE: values lesser than to min will be ignored.
        max : float or None
            Maximum excluded value for histogram bins. (default: None)
            NOTE: if max == None then self.L/2 is taken.
            NOTE: values greater than or equal to max will be ignored.
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.
        scale_diameter : bool
            Divide the distance between pairs of particles by the sum of the
            radii of the particles in the pair. (default: False)

        Returns
        -------
        gp : float Numpy array
            Array of (r, gp(r), errgp(r)) with gp(r) the proportion of pairs at
            distance r and errgp(r) the standard error on this measure.
        """

        if min == None: min = 0
        if max == None: max = self.L/2

        hist = np.array(list(map(
            # lambda t: (lambda dist: pycpp.getHistogramLinear(dist,
            #     Nbins, min, max)/dist.size)(
            #         pycpp.getDistances(self.getPositions(t), self.L,
            #             diameters=(
            #                 self.diameters if scale_diameter else None))),
            lambda t: pycpp.pairDistribution(
                Nbins, min, max, self.getPositions(t), self.L,
                diameters=(
                    self.diameters if scale_diameter else None)),
            self._time0(int_max=int_max))))

        bins = np.array([min + b*(max - min)/Nbins for b in range(1, Nbins)])
        histErr = np.array([mean_sterr(h) for h in np.transpose(hist)])[1:]

        histErr *= (self.L**2)/((max - min)/Nbins)

        return np.array([[b, *h/(2*np.pi*b)] for b, h in zip(bins, histErr)])

    def _time0(self, int_max=None):
        """
        Returns array of frames at which to compute positions.

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

    def _voronoi(self, time, centre=None):
        """
        Compute Voronoi tesselation of the system at `time'.

        Parameters
        ----------
        time : int
            Frame index.
        centre : (2,) float array-like or None
            Centre of the box to consider. (default: None)
            NOTE: if centre == None, then centre = (self.L/2, self.L/2).

        Returns
        -------
        voro : freud.locality.Voronoi
            Voronoi tesselation.
        """

        if type(centre) is type(None): centre = (self.L/2., self.L/2.)

        voro = Voronoi()
        voro.compute((
            Box.square(self.L),
            np.concatenate(
                (self.getPositions(time, centre=centre),
                np.zeros((self.N, 1))),
                axis=-1)))

        return voro

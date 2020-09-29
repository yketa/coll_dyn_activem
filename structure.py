"""
Module flow provides classes to compute and analyse positions to characterise
the structure of systems of ABPs.

(see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#ABP%20structure%20characteristics)
"""

from coll_dyn_activem.read import Dat
from coll_dyn_activem.maths import g2Dto1D, g2Dto1Dgrid, wave_vectors_2D,\
    DictList, angle

import numpy as np

from freud.locality import Voronoi
from freud.box import Box

class Positions(Dat):
    """
    Compute and analyse positions from simulation data.

    (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#Active%20Brownian%20particles)
    """

    def __init__(self, filename, skip=1):
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
        """

        super().__init__(filename, loadWork=False) # initialise with super class

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
            NOTE: if nBoxes==None, then nBoxes = int(sqrt(self.N)).

        Returns
        -------
        rho : (nBoxes, nBoxes) float Numpy array
            Particle density grid.
        """

        time = int(time)

        if nBoxes == None: nBoxes = np.sqrt(self.N)
        nBoxes = int(nBoxes)

        return self.toGrid(time,
            np.full((self.N,), fill_value=1),
            nBoxes=nBoxes, box_size=self.L, centre=(0, 0), average=False)

    def getLocalDensity(self, time):
        """
        Returns local packing fraction defined as the ratio of particles'
        volume to the volume of the corresponding voronoi cells.

        (see self._voronoi)
        (see https://yketa.github.io/PhD_Wiki/#Structure%20characteristics)

        Parameters
        ----------
        time : int
            Frame index.

        Returns
        -------
        localPhi : (self.N,) float numpy array
            Array of local packing fraction.
        """

        return ((np.pi/4.)*(self.diameters**2))/self._voronoi(time).volumes

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

    def getBondOrderParameter(self, time):
        """
        Get bond orientational order parameter at `time'.

        (see https://yketa.github.io/PhD_Wiki/#Structure%20characteristics)

        Parameters
        ----------
        time : int
            Frame index.

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

        return psi

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
        rho : (*, self.N) float numpy array
            Array of computed positions.
        """

        return np.array(list(map(
            lambda time0: self.getParticleDensity(time0, nBoxes=nBoxes),
            self._time0(int_max=int_max))))

    def structureFactor(self, int_max=None, nBoxes=None):
        """
        Returns static structure factor averaged along directions of space
        (assuming isotropy).

        Parameters
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.
        nBoxes : int
            Number of grid boxes in each direction. (default: None)
            NOTE: if nBoxes==None, then nBoxes = int(sqrt(self.N)).

        Returns
        -------
        S : (*, 2) float Numpy array
            Array of (k, S(k)) with S(k) the cylindrically averaged structure
            factor at wavevector k.
        """

        if nBoxes == None: nBoxes = np.sqrt(self.N)
        nBoxes = int(nBoxes)

        particleDensity = self.nParticleDensity(int_max=int_max, nBoxes=nBoxes)

        _S2D = np.array(list(map(
            lambda _rho:
                (lambda FFT: np.real(np.conj(FFT)*FFT))
                    (np.fft.fft2(_rho)),
            particleDensity)))/self.N

        k2D = np.sqrt(
            (wave_vectors_2D(nBoxes, nBoxes, self.L/nBoxes)**2).sum(axis=-1))

        return g2Dto1Dgrid(_S2D.mean(axis=0), k2D)

    def densityCorrelation(self, int_max=None, nBoxes=None):
        """
        Returns particle spacial density averaged along directions of space
        (assuming isotropy).

        NOTE: Correlations are computed with FFT.
              (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#Fourier%20transform%20field%20correlation)

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

        _G2D = np.array(list(map(
            lambda _rho:
                (lambda FFT: np.real(np.fft.ifft2(np.conj(FFT)*FFT)))
                    (np.fft.fft2(_rho - self.rho)),
            particleDensity)))/(nBoxes**2)

        return g2Dto1D(_G2D.mean(axis=0), self.L)

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
        time0 : (*,) float numpy array
            Array of frames.
        """

        if int_max == None: return np.array(range(self.skip, self.frames - 1))
        return np.linspace(
            self.skip, self.frames - 1, int(int_max), endpoint=False, dtype=int)

    def _voronoi(self, time):
        """
        Compute Voronoi tesselation of the system at `time'.

        Parameters
        ----------
        time : int
            Frame index.

        Returns
        -------
        voro : freud.locality.Voronoi
            Voronoi tesselation.
        """

        voro = Voronoi()
        voro.compute((
            Box.square(self.L),
            np.concatenate(
                (self.getPositions(time, centre=(self.L/2., self.L/2.)),
                np.zeros((self.N, 1))),
                axis=-1)))

        return voro

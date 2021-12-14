"""
Module pycpp provides interfaces between python and C++.

(see coll_dyn_activem/pycpp.cpp)
(see https://docs.python.org/3/library/ctypes.html)
(see https://numpy.org/doc/stable/user/basics.types.html)
"""

##########

import ctypes
import os
import numpy as np

# C++ library
_pycpp = ctypes.CDLL(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),    # project directory path
    '_pycpp.so'))                                   # C++ library share object

##########

def pointerArray(array):
    """
    Returns array of pointer to double array.

    Parameters
    ----------
    array : (*, **) array-like
        Input array.

    Returns
    -------
    pointer : LP_c_double_Array_*
        Output array of pointer.
    """

    return (ctypes.POINTER(ctypes.c_double)*len(array))(
        *[np.ctypeslib.as_ctypes(value) for value in array])

# HISTOGRAMS

def getHistogram(values, bins):
    """
    Build an histogram counting the occurences of `values' in the
    `len(bins) - 1' intervals of values in `bins'.

    Parameters
    ----------
    values : float array-like
        Values to count.
    bins : float array-like
        Limits of the bins.

    Returns
    -------
    histogram : (len(bins) - 1,) float Numpy array
        Histogram.
    """

    nValues = len(values)
    values = np.array(values, dtype=np.double)
    assert values.shape == (nValues,)
    nBins = len(bins) - 1
    bins = np.array(bins, dtype=np.double)
    bins.sort()
    assert bins.shape == (nBins + 1,)
    histogram = np.empty((nBins,), dtype=np.double)

    if nBins < 1: raise ValueError("Number of bins must be greater than 1.")

    _pycpp.getHistogram.argtypes = [
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
    _pycpp.getHistogram(
        nValues,
        np.ascontiguousarray(values),
        nBins,
        np.ascontiguousarray(bins),
        np.ascontiguousarray(histogram))

    return histogram

def getHistogramLinear(values, nBins, vmin, vmax):
    """
    Build an histogram counting the occurences of `values' in the `nBins'
    intervals of values between `vmin' and `vmax'.

    Parameters
    ----------
    values : float array-like
        Values to count.
    nBins : int
        Number of bins.
    vmin : float
        Minimum value of the bins (included).
    vmax : float
        Maximum value of the bins (excluded).

    Returns
    -------
    histogram : (nBins,) float Numpy array
        Histogram.
    """

    nValues = len(values)
    values = np.array(values, dtype=np.double)
    assert values.shape == (nValues,)
    nBins = int(nBins)
    histogram = np.empty((nBins,), dtype=np.double)

    if nBins < 1: raise ValueError("Number of bins must be greater than 1.")

    _pycpp.getHistogramLinear.argtypes = [
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
    _pycpp.getHistogramLinear(
        nValues,
        np.ascontiguousarray(values),
        nBins,
        vmin,
        vmax,
        np.ascontiguousarray(histogram))

    return histogram

# DISTANCES

def pairIndex(i, j, N):
    """
    For `N' particles, return a unique pair index for the couples (`i', `j')
    and (`j', `i') in {0, ..., N(N + 1)/2 - 1}.

    Parameters
    ----------
    i : int
        Index of first particle.
    j : int
        Index of second particle.
    N : int
        Number of particles.

    Returns
    -------
    index : int
        Unique index.
    """

    N = int(N)
    assert N > 0
    i = int(i)
    assert i < N
    j = int(j)
    assert j < N

    _pycpp.pairIndex.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

    return _pycpp.pairIndex(i, j, N)

def invPairIndex(index, N):
    """
    For `N' particles, return the pair (`i', `j') corresponding to the unique
    pair index `index'. (see pycpp.pairIndex)

    Parameters
    ----------
    index : int
        Unique index.
    N : int
        Number of particles.

    Returns
    -------
    i : int
        Index of first particle.
    j : int
        Index of second particle.
    """

    row = np.floor(np.sqrt(2*index + 1./4.) - 1./2.)
    i = index - row*(row + 1)/2
    j = N - 1 + index - row*(row + 3)/2
    assert index == pairIndex(i, j, N)

    return int(i), int(j)

def getDifferences(positions, L, diameters=None):
    """
    Compute position differences between the particles with `positions' of a
    system of size `L'. Differences are rescaled by the sum of the radii of the
    particles in the pair if `diameters' != None.

    Parameters
    ----------
    positions : (*, 2) float array-like
        Positions of the particles.
    L : float
        Size of the system box.
    diameters : (*,) float array-like or None
        Diameters of the particles. (default: None)

    Returns
    -------
    differences : (*, 2) float Numpy array
        Array of position differences between pairs.
    """

    positions = np.array(positions, dtype=np.double)
    N = len(positions)
    assert positions.shape == (N, 2)
    scale_diameter = not(type(diameters) is type(None))
    if scale_diameter:
        diameters = np.array(diameters, dtype=np.double)
    else:
        diameters = np.empty((N,), dtype=np.double)
    assert diameters.shape == (N,)
    differences_x = np.empty((int(N*(N - 1)/2),), dtype=np.double)
    differences_y = np.empty((int(N*(N - 1)/2),), dtype=np.double)

    _pycpp.getDifferences.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_bool]
    _pycpp.getDifferences(
        N,
        L,
        np.ascontiguousarray(positions[:, 0]),
        np.ascontiguousarray(positions[:, 1]),
        np.ascontiguousarray(diameters),
        np.ascontiguousarray(differences_x),
        np.ascontiguousarray(differences_y),
        scale_diameter)

    return np.concatenate(
        (
            differences_x.reshape(len(differences_x), 1),
            differences_y.reshape(len(differences_y), 1)),
        axis=-1)

def getDistances(positions, L, diameters=None):
    """
    Compute distances between the particles with `positions' of a system of size
    `L'. Distances are rescaled by the sum of the radii of the particles in the
    pair if `diameters' != None.

    Parameters
    ----------
    positions : (*, 2) float array-like
        Positions of the particles.
    L : float
        Size of the system box.
    diameters : (*,) float array-like or None
        Diameters of the particles. (default: None)

    Returns
    -------
    distances : float Numpy array
        Array of distances between pairs.
    """

    positions = np.array(positions, dtype=np.double)
    N = len(positions)
    assert positions.shape == (N, 2)
    scale_diameter = not(type(diameters) is type(None))
    if scale_diameter:
        diameters = np.array(diameters, dtype=np.double)
    else:
        diameters = np.empty((N,), dtype=np.double)
    assert diameters.shape == (N,)
    distances = np.empty((int(N*(N - 1)/2),), dtype=np.double)

    _pycpp.getDistances.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_bool]
    _pycpp.getDistances(
        N,
        L,
        np.ascontiguousarray(positions[:, 0]),
        np.ascontiguousarray(positions[:, 1]),
        np.ascontiguousarray(diameters),
        np.ascontiguousarray(distances),
        scale_diameter)

    return distances

def getOrientationNeighbours(A1, L, diameters, positions, *displacements):
    """
    Computer for each particle the number of other particles at distance lesser
    than `A1' relative to their average diameter with the same orientation of
    `displacements'.

    Parameters
    ----------
    A1 : float
        Distance relative to their diameters below which particles are
        considered bonded.
    L : float
        Size of the system box.
    diameters : (*,) float array-like
        Array of diameters.
    positions : (*, 2) float array-like
        Initial positions.
    displacements : (*, 2) float array-like
        Displacements of the particles.

    Returns
    -------
    oneigbours : (**, *) int Numpy array
        Number of neighbours with same displacement orientation with:
            *  : the number of particles,
            ** : the number of `displacements' provided.
    """

    positions = np.array(positions, dtype=np.double)
    N = len(positions)
    assert positions.shape == (N, 2)
    displacements = np.array(displacements, dtype=np.double)
    nDisp = len(displacements)
    assert displacements.shape == (nDisp, N, 2)
    diameters = np.array(diameters, dtype=np.double)
    assert diameters.shape == (N,)
    _oneighbours = np.empty((N,), dtype=np.intc)
    oneighbours = []

    distances = getDistances(positions, L, diameters=None)

    _pycpp.getOrientationNeighbours.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.intc, ndim=1, flags='C_CONTIGUOUS')]

    for i in range(nDisp):

        _pycpp.getOrientationNeighbours(
            N,
            A1,
            np.ascontiguousarray(diameters),
            np.ascontiguousarray(distances),
            np.ascontiguousarray(displacements[i][:, 0]),
            np.ascontiguousarray(displacements[i][:, 1]),
            np.ascontiguousarray(_oneighbours))
        oneighbours += [_oneighbours.tolist()]

    return np.array(oneighbours)

def getBrokenBonds(A1, A2, L, diameters, positions0, *positions1, pairs=False):
    """
    Compute the number of broken bonds for particles from `positions0' to
    `positions1'.

    Parameters
    ----------
    A1 : float
        Distance relative to their diameters below which particles are
        considered bonded.
    A2 : float
        Distance relative to their diameters above which particles are
        considered unbonded.
    L : float
        Size of the system box.
    diameters : (*,) float array-like
        Array of diameters.
    positions0 : (*, 2) float array-like
        Initial positions.
    positions1 : (*, 2) float array-like
        Final positions.
    pairs : bool
        Return array of broken pairs (see pycpp.pairIndex for indexing).
        (default: False)

    Returns
    -------
    brokenBonds : (**, *) int Numpy array
        Number of broken bonds between `positions0' and `positions1' with:
            *  : the number of particles,
            ** : the number of `positions1' provided.
    [pairs] brokenPairs : (**, *(* - 1)/2) bool Numpy array
        Broken bond between particles of pair truth values with:
            *  : the number of particles,
            ** : the number of `positions1' provided.
    """

    positions0 = np.array(positions0, dtype=np.double)
    N = len(positions0)
    assert positions0.shape == (N, 2)
    positions1 = np.array(positions1, dtype=np.double)
    nPos = len(positions1)
    assert positions1.shape == (nPos, N, 2)
    diameters = np.array(diameters, dtype=np.double)
    assert diameters.shape == (N,)
    _brokenBonds = np.empty((N,), dtype=np.intc)
    brokenBonds = []
    _brokenPairs = np.empty((int(N*(N - 1)/2),), dtype=np.bool_)
    brokenPairs = []

    distances0 = getDistances(positions0, L, diameters=None)

    _pycpp.getBrokenBonds.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.intc, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.bool_, ndim=1, flags='C_CONTIGUOUS')]

    for i in range(nPos):

        distances1 = getDistances(positions1[i], L, diameters=None)
        _pycpp.getBrokenBonds(
            N,
            A1,
            A2,
            np.ascontiguousarray(diameters),
            np.ascontiguousarray(distances0),
            np.ascontiguousarray(distances1),
            np.ascontiguousarray(_brokenBonds),
            np.ascontiguousarray(_brokenPairs))
        brokenBonds += [_brokenBonds.tolist()]
        brokenPairs += [_brokenPairs.tolist()]

    brokenBonds = np.array(brokenBonds)
    brokenPairs = np.array(brokenPairs)
    if pairs: return brokenBonds, brokenPairs
    return brokenBonds

def getVanHoveDistances(positions, displacements, L):
    """
    Compte van Hove distances between particles of a system of size `L', with
    `positions' and `displacements'.

    Parameters
    ----------
    positions : (*, 2) float array-like
        Positions of the particles.
    displacements : (*, 2) float array-like
        Displacements of the particles.
    L : float
        Size of the system box.

    Returns
    -------
    distances : (*^2,) float Numpy array
        Van Hove distances.
    """

    positions = np.array(positions, dtype=np.double)
    N = len(positions)
    assert positions.shape == (N, 2)
    displacements = np.array(displacements, dtype=np.double)
    assert displacements.shape == (N, 2)
    distances = np.empty((N**2,), dtype=np.double)

    _pycpp.getVanHoveDistances.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
    _pycpp.getVanHoveDistances(
        N,
        L,
        np.ascontiguousarray(positions[:, 0]),
        np.ascontiguousarray(positions[:, 1]),
        np.ascontiguousarray(displacements[:, 0]),
        np.ascontiguousarray(displacements[:, 1]),
        np.ascontiguousarray(distances))

    return distances

def nonaffineSquaredDisplacement(positions0, positions1, L, A1, diameters):
    """
    Compute nonaffine squared displacements for particles in a system of size
    `L' between positions `positions0' and `positions1'.

    Parameters
    ----------
    positions0 : (*, 2) float array-like
        Initial positions of the particles.
    positions1 : (*, 2) float array-like
        Final positions of the particles.
    L : float
        Size of the system box.
    A1 : float
        Distance relative to their diameters below which particles are
        considered bonded.
    diameters : (*,) float array-like
        Array of diameters.

    Returns
    -------
    D2min : (*,) float Numpy array
        Nonaffine squared displacements.
    """

    positions0 = np.array(positions0, dtype=np.double)
    N = len(positions0)
    assert positions0.shape == (N, 2)
    positions1 = np.array(positions1, dtype=np.double)
    assert positions1.shape == (N, 2)
    diameters = np.array(diameters, dtype=np.double)
    assert diameters.shape == (N,)
    D2min = np.full((N,), fill_value=0, dtype=np.double)

    if (positions1 == positions0).all(): return D2min

    _pycpp.nonaffineSquaredDisplacement.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
    _pycpp.nonaffineSquaredDisplacement(
        N,
        L,
        np.ascontiguousarray(positions0[:, 0]),
        np.ascontiguousarray(positions0[:, 1]),
        np.ascontiguousarray(positions1[:, 0]),
        np.ascontiguousarray(positions1[:, 1]),
        A1,
        diameters,
        np.ascontiguousarray(D2min))

    return D2min

def pairDistribution(nBins, vmin, vmax, positions, L, diameters=None):
    """
    Compute pair distribution function as histogram with `nBins' intervals of
    values between `vmin' and `vmax' from the distances between the particles
    with `positions' of a system of size `L'. Distances are rescaled by the sum
    of the radii of the particles in the pair if `diameters' != None.

    Parameters
    ----------
    nBins : int
        Number of bins.
    vmin : float
        Minimum value of the bins (included).
    vmax : float
        Maximum value of the bins (excluded).
    positions : (*, 2) float array-like
        Positions of the particles.
    L : float
        Size of the system box.
    diameters : (*,) float array-like or None
        Diameters of the particles. (default: None)

    Returns
    -------
    histogram : (nBins,) float Numpy array
        Pair distribution function.
    """

    nBins = int(nBins)
    if nBins < 1: raise ValueError("Number of bins must be greater than 1.")
    histogram = np.empty((nBins,), dtype=np.double)
    positions = np.array(positions, dtype=np.double)
    N = len(positions)
    assert positions.shape == (N, 2)
    scale_diameter = not(type(diameters) is type(None))
    if scale_diameter:
        diameters = np.array(diameters, dtype=np.double)
    else:
        diameters = np.empty((N,), dtype=np.double)
    assert diameters.shape == (N,)

    _pycpp.pairDistribution.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_bool]
    _pycpp.pairDistribution(
        nBins,
        vmin,
        vmax,
        histogram,
        N,
        L,
        np.ascontiguousarray(positions[:, 0]),
        np.ascontiguousarray(positions[:, 1]),
        np.ascontiguousarray(diameters),
        scale_diameter)

    return histogram

def S4Fs(filename, time0, dt, q, k):
    """
    Compute four-point structure factor.

    Parameters
    ----------
    filename : string
        Name of .datN data file.
    time0 : (*,) int array-like
        Array of initial times.
    dt : (**,) int array-like
        Array of lag times.
    q : (***, 2) float array-like
        Wave-vectors along which to compute four-point structure factor.
    k : (****, 2) float array-like
        Wave-vectors at which to compute self-intermediate scattering function.

    Returns
    -------
    S4 : (**,) float numpy array
        Mean four-point structure factor along wave-vectors.
    S4var : (**,) float
        Variance on the four-point structure factor along wave-vectors.
    """

    time0 = np.array(time0, dtype=np.intc)
    nTime0 = len(time0)
    assert time0.shape == (nTime0,)
    dt = np.array(dt, dtype=np.intc)
    nDt = len(dt)
    assert dt.shape == (nDt,)
    q = np.array(q, dtype=np.double)
    nq = len(q)
    assert q.shape == (nq, 2)
    k = np.array(k, dtype=np.double)
    nk = len(k)
    assert k.shape == (nk, 2)

    S4 = np.empty((nDt,), dtype=np.double)
    S4var = np.empty((nDt,), dtype=np.double)

    _pycpp.S4Fs.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.intc, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.intc, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
    _pycpp.S4Fs(
        filename.encode('utf-8'),
        nTime0,
        np.ascontiguousarray(time0),
        nDt,
        np.ascontiguousarray(dt),
        nq,
        np.ascontiguousarray(q[:, 0]),
        np.ascontiguousarray(q[:, 1]),
        nk,
        np.ascontiguousarray(k[:, 0]),
        np.ascontiguousarray(k[:, 1]),
        np.ascontiguousarray(S4),
        np.ascontiguousarray(S4var))

    return S4, S4var

def getLocalParticleDensity(a, positions, L, diameters):
    """
    Returns local packing fraction for each particle.

    Parameters
    ----------
    a : float
        Size of the box in which to compute local packing fractions.
    positions : (*, 2) float array-like
        Positions of the particles.
    L : float
        Size of the system box.
    diameters : (*,) float array-like
        Diameters of the particles.

    Returns
    -------
    densities : (*,) float Numpy array
        Array of local packing fractions.
    """

    positions = np.array(positions, dtype=np.double)
    N = len(positions)
    assert positions.shape == (N, 2)
    diameters = np.array(diameters, dtype=np.double)
    assert diameters.shape == (N,)
    densities = np.empty((N,), dtype=np.double)

    _pycpp.getLocalParticleDensity.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
    _pycpp.getLocalParticleDensity(
        N,
        L,
        a,
        np.ascontiguousarray(positions[:, 0]),
        np.ascontiguousarray(positions[:, 1]),
        np.ascontiguousarray(diameters),
        np.ascontiguousarray(densities))

    return densities

def isNotInBubble(philim, dlim, positions, L, phi):
    """
    Returns particles which are not within distance `dlim' of particles which
    packing fractions `phi' are below `philim'.

    Parameters
    ----------
    philim : float
        Packing fraction below which particles are considered in a bubble.
    dlim : float
        Distance from bubble within which to discard particles.
    positions : (*, 2) float array-like
        Positions of the particles.
    L : float
        Size of the system box.
    phi : (*,) float array-like
        Local packing fractions of particles.

    Returns
    -------
    notInBubble : (*,) bool Numpy array
        Particles not in bubbles.
    """

    positions = np.array(positions, dtype=np.double)
    N = len(positions)
    assert positions.shape == (N, 2)
    phi = np.array(phi, dtype=np.double)
    assert phi.shape == (N,)
    notInBubble = np.empty((N,), dtype=np.bool_)

    _pycpp.isNotInBubble.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.bool_, ndim=1, flags='C_CONTIGUOUS')]
    _pycpp.isNotInBubble(
        N,
        L,
        philim,
        dlim,
        np.ascontiguousarray(positions[:, 0]),
        np.ascontiguousarray(positions[:, 1]),
        np.ascontiguousarray(phi),
        np.ascontiguousarray(notInBubble))

    return notInBubble

# GRIDS

def toGrid(positions, L, values, nBoxes, average=False):
    """
    Maps square (sub-)system of particles at `positions' centred around 0 and
    of (cropped) size `L' to a square (`nBoxes', `nBoxes') grid, and associates
    to each box the sum or averaged value of the (*, **)-array `values'.

    Parameters
    ----------
    positions : (*, 2) float array-like
        Positions of the particles.
        NOTE: These positions must be centred around 0, as when obtained with
              coll_dyn_activem.read.Dat.getPositions with argument
              centre != None.
    L : float
        (Cropped) size of the system box.
    values : (*, **)  float array-like
        Values to be put on the grid.
    nBoxes : int
        Number of boxes in each direction of the grid.

    Returns
    -------
    grid : (nBoxes, nBoxes, **) float Numpy array
        Computed grid.
    """

    positions = np.array(positions, dtype=np.double)
    N = len(positions)
    assert positions.shape == (N, 2)
    values = np.array(values, dtype=np.double)
    assert values.shape[0] == N
    assert values.ndim <= 2
    if values.ndim == 1: values = values.reshape(values.shape + (1,))
    dim = values.shape[1]
    nBoxes = int(nBoxes)
    grid = [np.empty((nBoxes**2,), dtype=np.double) for d in range(dim)]

    _pycpp.toGrid.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_bool]
    for d in range(dim):
        _pycpp.toGrid(
            N,
            L,
            np.ascontiguousarray(positions[:, 0]),
            np.ascontiguousarray(positions[:, 1]),
            np.ascontiguousarray(values[:, d]),
            nBoxes,
            np.ascontiguousarray(grid[d]),
            average)
    grid = np.transpose(grid)

    if dim == 1: grid = np.reshape(grid, (nBoxes, nBoxes))
    else: grid = np.reshape(grid, (nBoxes, nBoxes, dim))
    return grid

def g2Dto1Dgrid(g2D, grid):
    """
    Returns cylindrical average of square 2D grid with values of radii given
    by other parameter grid.

    Parameters
    ----------
    g2D : (*, *) float array-like
        Square 2D grid.
    grid : (*, *) float-array like
        Array of radii.

    Returns
    -------
    g1D : Numpy array
        Array of (r, g1D(r)) with g1D(r) the averaged 2D grid at radius r.
    """

    g2D = np.array(g2D, dtype=np.double)
    nBoxes = len(g2D)
    assert g2D.shape == (nBoxes, nBoxes)
    g2D = g2D.reshape((nBoxes**2,))
    grid = np.array(grid, dtype=np.double)
    assert grid.shape == (nBoxes, nBoxes)
    grid = grid.reshape((nBoxes**2,))
    g1D = np.empty((nBoxes**2,), dtype=np.double)
    radii = np.empty((nBoxes**2,), dtype=np.double)
    nRadii = ctypes.c_int()

    _pycpp.g2Dto1Dgrid.argtypes = [
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.POINTER(ctypes.c_int)]
    _pycpp.g2Dto1Dgrid(
        nBoxes,
        np.ascontiguousarray(g2D),
        np.ascontiguousarray(grid),
        np.ascontiguousarray(g1D),
        np.ascontiguousarray(radii),
        nRadii)
    nRadii = int(nRadii.value)
    g1D = g1D[:nRadii]
    radii = radii[:nRadii]

    return np.concatenate(
        (radii.reshape(radii.shape + (1,)), g1D.reshape(g1D.shape + (1,))),
        axis=-1)

def g2Dto1Dgridhist(g2D, grid, nBins, vmin=None, vmax=None):
    """
    Returns cylindrical average of square 2D grid with values of radii given
    by other parameter grid as histogram of `nBins' between `vmin' and `vmax'.

    Parameters
    ----------
    g2D : (*, *) float array-like
        Square 2D grid.
    grid : (*, *) float-array like
        Array of radii.
    nBins : int
        Number of histogram bins.
    vmin : float or None
        Minimum (included) radii in the histogram. (default: None)
        NOTE: if vmin == None then vmin = grid.min().
    vmax : float or None
        Maximum (excluded) radii in the histogram. (default: None)
        NOTE: if vmax == None then vmax = grid.max().

    Returns
    -------
    g1D : Numpy array
        Array of (r, g1D(r), g1Dstd(r)) with g1D(r) the averaged 2D grid at bin
        corresponding to minimum radius r, and g1Dstd(r) the standard deviation
        on this measure.
    """

    g2D = np.array(g2D, dtype=np.double)
    nBoxes = len(g2D)
    assert g2D.shape == (nBoxes, nBoxes)
    g2D = g2D.reshape((nBoxes**2,))
    grid = np.array(grid, dtype=np.double)
    assert grid.shape == (nBoxes, nBoxes)
    grid = grid.reshape((nBoxes**2,))
    nBins = int(nBins)
    vmin = grid.min() if vmin == None else vmin
    vmax = grid.max() if vmax == None else vmax
    bins  = np.linspace(vmin, vmax, nBins, endpoint=False, dtype=np.double)
    g1D = np.empty((nBins,), dtype=np.double)
    g1Dstd = np.empty((nBins,), dtype=np.double)

    _pycpp.g2Dto1Dgridhist.argtypes = [
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
    _pycpp.g2Dto1Dgridhist(
        nBoxes,
        np.ascontiguousarray(g2D),
        np.ascontiguousarray(grid),
        nBins,
        vmin,
        vmax,
        np.ascontiguousarray(g1D),
        np.ascontiguousarray(g1Dstd))

    return np.concatenate(
        (bins.reshape(bins.shape + (1,)),
            g1D.reshape(g1D.shape + (1,)),
            g1Dstd.reshape(g1Dstd.shape + (1,))),
        axis=-1)

# CORRELATIONS

def getRadialCorrelations(positions, L, values, nBins, min=None, max=None,
    rescale_pair_distribution=False, values2=None):
    """
    Compute radial correlations between `values' (and `values2') associated to
    each of the `positions' of a system of size `L'.

    Parameters
    ----------
    positions : (*, 2) float array-like
        Positions of the particles.
    L : float
        Size of the system box.
    values : (*, **)  float array-like
        Values to compute the correlations of.
        NOTE: if these values are 1D arrays, the sum of the correlation on each
              axis is returned.
    nBins : int
        Number of intervals of distances on which to compute the correlations.
    min : float or None
        Minimum distance (included) at which to compute the correlations.
        (default: None)
        NOTE: if min == None then min = 0.
    max : float or None
        Maximum distance (excluded) at which to compute the correlations.
        (default: None)
        NOTE: if max == None then max = L/2.
    rescale_pair_distribution : bool
        Rescale correlations by pair distribution function. (default: False)
    values2 : None or (*, **)  float array-like
        Values to compute the correlations of `values' with.
        NOTE: if values2 == None then the autocorrelations of `values' are
              computed.

    Returns
    -------
    correlations : (nBins, 2) float Numpy array
        Array of (r, C(r)) where r is the lower bound of the bin and C(r) the
        radial correlation computed for this bin.
    """

    positions = np.array(positions, dtype=np.double)
    N = len(positions)
    assert positions.shape == (N, 2)
    assert values.shape[0] == N
    assert values.ndim <= 2
    if values.ndim == 1: values = values.reshape(values.shape + (1,))
    if type(values2) == type(None): values2 = values
    if values2.ndim == 1: values2 = values2.reshape(values2.shape + (1,))
    assert values.shape == values2.shape
    nBins = int(nBins)
    min = 0 if min == None else min
    max = L/2 if max == None else max
    correlations = np.empty((nBins,), dtype=np.double)

    _pycpp.getRadialCorrelations.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_bool]
    _pycpp.getRadialCorrelations(
        N,
        L,
        np.ascontiguousarray(positions[:, 0]),
        np.ascontiguousarray(positions[:, 1]),
        values.shape[1],
        pointerArray(values),
        pointerArray(values2),
        nBins,
        min,
        max,
        np.ascontiguousarray(correlations),
        rescale_pair_distribution)

    return np.array([[min + bin*(max - min)/nBins, correlations[bin]]
        for bin in range(nBins)])

def getVelocitiesOriCor(positions, L, velocities, sigma=1):
    """
    Compute radial correlation of orientations of `velocities' associated to
    each of the `positions' of a system of size `L'.

    Parameters
    ----------
    positions : (*, 2) float array-like
        Positions of the particles.
    L : float
        Size of the system box.
    velocities : (*, 2)  float array-like
        Velocities of the particles.
    sigma : float
        Mean radius. (default: 1)

    Returns
    -------
    correlations : (*, 2) float Numpy array
        Array of (r, C(r)) where r is the upper bound of the bin and C(r) the
        radial correlation of orientations of velocities computed for this bin.
    """

    positions = np.array(positions, dtype=np.double)
    N = len(positions)
    assert positions.shape == (N, 2)
    velocities = np.array(velocities, dtype=np.double)
    assert velocities.shape == (N, 2)
    nBins = int((L/2)/sigma);
    correlations = np.empty((nBins,), dtype=np.double)

    _pycpp.getVelocitiesOriCor.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_double]
    _pycpp.getVelocitiesOriCor(
        N,
        L,
        np.ascontiguousarray(positions[:, 0]),
        np.ascontiguousarray(positions[:, 1]),
        np.ascontiguousarray(velocities[:, 0]),
        np.ascontiguousarray(velocities[:, 1]),
        np.ascontiguousarray(correlations),
        sigma)

    return np.array([[1 + bin*sigma, correlations[bin]]
        for bin in range(nBins)])

# READ

def readDouble(filename, targets):
    """
    Returns an array of double values at `targets' read from file `filename'.

    Parameters
    ----------
    filename : string
        File from which to read.
    targets : (*,) int array-like
        Stream position byte offsets.

    Returns
    -------
    out : (*,) float Numpy array
        Values from file.
    """

    filename = str(filename)
    targets = np.array(targets, dtype=np.int_)
    assert targets.ndim == 1
    nTargets = len(targets)
    out = np.empty(targets.shape, dtype=np.double)

    _pycpp.readDouble.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.int_, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')]
    _pycpp.readDouble(
        filename.encode('utf-8'),
        nTargets,
        np.ascontiguousarray(targets),
        np.ascontiguousarray(out))

    return out

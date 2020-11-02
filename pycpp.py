"""
Module _pycpp provides interfaces between python and C++.

(see coll_dyn_activem/pycpp.cpp)
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
    values = np.array(values, dtype=np.float64)
    assert values.shape == (nValues,)
    nBins = len(bins) - 1
    bins = np.array(bins, dtype=np.float64)
    bins.sort()
    assert bins.shape == (nBins + 1,)
    histogram = np.empty((nBins,), dtype=np.float64)

    if nBins < 1: raise ValueError("Number of bins must be greater than 1.")

    _pycpp.getHistogram.argtypes = [
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')]
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
    values = np.array(values, dtype=np.float64)
    assert values.shape == (nValues,)
    nBins = int(nBins)
    histogram = np.empty((nBins,), dtype=np.float64)

    if nBins < 1: raise ValueError("Number of bins must be greater than 1.")

    _pycpp.getHistogramLinear.argtypes = [
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')]
    _pycpp.getHistogramLinear(
        nValues,
        np.ascontiguousarray(values),
        nBins,
        vmin,
        vmax,
        np.ascontiguousarray(histogram))

    return histogram

# DISTANCES

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
    diameters : float array-like or None
        Diameters of the particles. (default: None)

    Returns
    -------
    distances : float Numpy array
        Array of distances between pairs.
    """

    positions = np.array(positions, dtype=np.float64)
    N = len(positions)
    assert positions.shape == (N, 2)
    scale_diameter = not(type(diameters) is type(None))
    if scale_diameter:
        diameters = np.array(diameters, dtype=np.float64)
    else:
        diameters = np.empty((N,), dtype=np.float64)
    assert diameters.shape == (N,)
    distances = np.empty((int(N*(N - 1)/2),), dtype=np.float64)

    _pycpp.getDistances.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
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

# CORRELATIONS

def getRadialCorrelations(positions, L, values, nBins, min=None, max=None):
    """
    Compute radial correlations between `values' associated to each of the
    `positions' of a system of size `L'.

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

    Returns
    -------
    correlations : (nBins, 2) float Numpy array
        Array of (r, C(r)) where r is the lower bound of the bin and C(r) the
        radial correlation computed for this bin.
    """

    positions = np.array(positions, dtype=np.float64)
    N = len(positions)
    assert positions.shape == (N, 2)
    assert values.shape[0] == N
    assert values.ndim <= 2
    if values.ndim == 1: values = values.reshape(values.shape + (1,))
    nBins = int(nBins)
    min = 0 if min == None else min
    max = L/2 if max == None else max
    correlations = np.empty((nBins,), dtype=np.float64)

    _pycpp.getRadialCorrelations.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')]
    _pycpp.getRadialCorrelations(
        N,
        L,
        np.ascontiguousarray(positions[:, 0]),
        np.ascontiguousarray(positions[:, 1]),
        values.shape[1],
        (ctypes.POINTER(ctypes.c_double)*N)(
            *[np.ctypeslib.as_ctypes(value) for value in values]),
        nBins,
        min,
        max,
        np.ascontiguousarray(correlations))

    return np.array([[min + bin*(max - min)/nBins, correlations[bin]]
        for bin in range(nBins)])

def getVelocitiesOriCor(positions, L, velocities, sigma=1):
    """
    Compute radial correlation of orientations of `velocities' associated to
    each of the `positions' of a system of size `L'.

    (see https://yketa.github.io/PhD_Wiki/#Flow%20characteristics)

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

    positions = np.array(positions, dtype=np.float64)
    N = len(positions)
    assert positions.shape == (N, 2)
    velocities = np.array(velocities, dtype=np.float64)
    assert velocities.shape == (N, 2)
    nBins = int((L/2)/sigma);
    correlations = np.empty((nBins,), dtype=np.float64)

    _pycpp.getVelocitiesOriCor.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
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

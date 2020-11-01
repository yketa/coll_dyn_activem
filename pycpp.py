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

def getDistances(positions, L, diameters=None):
    """
    Compute distances between the particles with `positions' of a system of size
    `L'. Distances are rescaled by the sum of the radii of the particles in the
    pair if `diameters' != None.

    Parameters
    ----------
    positions : float array-like
        Positions of the particles.
    L : float
        Size of the system box.
    diameters : float array-like or None
        Diameters of the particles.

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

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

    positions = np.array(positions, dtype=np.float64)
    N = len(positions)
    assert positions.shape == (N, 2)
    values = np.array(values, dtype=np.float64)
    assert values.shape[0] == N
    assert values.ndim <= 2
    if values.ndim == 1: values = values.reshape(values.shape + (1,))
    dim = values.shape[1]
    nBoxes = int(nBoxes)
    grid = [np.empty((nBoxes**2,), dtype=np.float64) for d in range(dim)]

    _pycpp.toGrid.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
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

    g2D = np.array(g2D, dtype=np.float64)
    nBoxes = len(g2D)
    assert g2D.shape == (nBoxes, nBoxes)
    g2D = g2D.reshape((nBoxes**2,))
    grid = np.array(grid, dtype=np.float64)
    assert grid.shape == (nBoxes, nBoxes)
    grid = grid.reshape((nBoxes**2,))
    g1D = np.empty((nBoxes**2,), dtype=np.float64)
    radii = np.empty((nBoxes**2,), dtype=np.float64)
    nRadii = ctypes.c_int()

    _pycpp.g2Dto1Dgrid.argtypes = [
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
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

    g2D = np.array(g2D, dtype=np.float64)
    nBoxes = len(g2D)
    assert g2D.shape == (nBoxes, nBoxes)
    g2D = g2D.reshape((nBoxes**2,))
    grid = np.array(grid, dtype=np.float64)
    assert grid.shape == (nBoxes, nBoxes)
    grid = grid.reshape((nBoxes**2,))
    nBins = int(nBins)
    vmin = grid.min() if vmin == None else vmin
    vmax = grid.max() if vmax == None else vmax
    bins  = np.linspace(vmin, vmax, nBins, endpoint=False, dtype=np.float64)
    g1D = np.empty((nBins,), dtype=np.float64)
    g1Dstd = np.empty((nBins,), dtype=np.float64)

    _pycpp.g2Dto1Dgridhist.argtypes = [
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')]
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
    rescale_pair_distribution=False):
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
    rescale_pair_distribution : bool
        Rescale correlations by pair distribution function. (default: False)

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
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_bool]
    _pycpp.getRadialCorrelations(
        N,
        L,
        np.ascontiguousarray(positions[:, 0]),
        np.ascontiguousarray(positions[:, 1]),
        values.shape[1],
        pointerArray(values),
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

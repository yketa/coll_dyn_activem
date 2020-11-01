#ifndef PYCPP_HPP
#define PYCPP_HPP

////////////////
// PROTOTYPES //
////////////////

extern "C" void getHistogram(
  int nValues, double* values,
  int nBins, double* bins, double* histogram);
  // Put the `nValues' `values' in an `histogram' with `nBins' bins defined by
  // the `nBins' + 1 values in `bins'.

extern "C" void getHistogramLinear(
  int nValues, double* values,
  int nBins, double vmin, double vmax, double* histogram);
  // Put the `nValues' `values' in an `histogram' with `nBins' bins defined by
  // linearly spaced between `vmin' and `vmax'.

extern "C" void getDistances(
  int N, double L, double* x, double* y, double* diameters, double *distances,
  bool scale_diameter = false);
  // Compute distances between the `N' particles of a system of size `L', with
  // x-axis positions given by `x' and y-axis positions given by `y'.
  // Distances are rescaled by the sum of the radii of the particles in the pair
  // if `scale_diameter'.
  // NOTE: distances must have (at least) N(N - 1)/2 entries.

#endif

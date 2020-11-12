#ifndef PYCPP_HPP
#define PYCPP_HPP

////////////////
// PROTOTYPES //
////////////////

// HISTOGRAMS

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

// DISTANCES

double getDistance(
  double const& x0, double const& y0, double const& x1, double const& y1,
  double const& L);
  // Compute distance between (`x0', `y0') and (`x1', `y1') in a periodic square
  // system of size L.

extern "C" void getDistances(
  int N, double L, double* x, double* y, double* diameters, double *distances,
  bool scale_diameter = false);
  // Compute distances between the `N' particles of a system of size `L', with
  // x-axis positions given by `x' and y-axis positions given by `y'.
  // Distances are rescaled by the sum of the radii of the particles in the pair
  // if `scale_diameter'.
  // NOTE: distances must have (at least) N(N - 1)/2 entries.

// CORRELATIONS

extern "C" void getRadialCorrelations(
  int N, double L, double* x, double* y, int dim, double** values,
  int nBins, double rmin, double rmax, double* correlations,
  bool rescale_pair_distribution = false);
  // Compute radial correlations between the (`dim',) float arrays `values'
  // associated to each of the `N' particles of a system of size `L', with
  // x-axis positions given by `x' and y-axis positions given by `y'.
  // Correlations are computed on the interval between `rmin' (included) and
  // `rmax' (excluded) with `nBins' bins.
  // Correlations are rescaled by pair distribution function (for bins > 0) if
  // `rescale_pair_distribution'.

extern "C" void getVelocitiesOriCor(
  int N, double L, double* x, double* y, double* vx, double* vy,
  double* correlations,
  double sigma);
  // Compute radial correlations of orientation of velocities (`vx', `vy')
  // associated to each of the `N' particles of a system of size `L', with
  // x-axis positions given by `x' and y-axis positions given by `y', and mean
  // diameter `sigma'.
  // (see https://yketa.github.io/PhD_Wiki/#Flow%20characteristics)

#endif

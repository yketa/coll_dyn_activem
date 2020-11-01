#include <math.h>
#include <vector>

#include "maths.hpp"
#include "pycpp.hpp"

#include <iostream>

extern "C" void getHistogram(
  int nValues, double* values,
  int nBins, double* bins, double* histogram) {
  // Put the `nValues' `values' in an `histogram' with `nBins' bins defined by
  // the `nBins' + 1 values in `bins'.

  std::vector<double> vecValues(values, values + nValues);
  std::sort(vecValues.begin(), vecValues.end());

  for (int i=0; i < nBins; i++) { histogram[i] = 0; }

  int bin = 0;
  for (auto value = vecValues.begin(); value != vecValues.end(); value++) {
    if ( *value < bins[0] ) { continue; }
    while ( *value >= bins[bin + 1] ) {
      bin++;
      if ( bin == nBins ) { return; }
    }
    histogram[bin] += 1;
  }
}

extern "C" void getHistogramLinear(
  int nValues, double* values,
  int nBins, double vmin, double vmax, double* histogram) {
  // Put the `nValues' `values' in an `histogram' with `nBins' bins defined by
  // linearly spaced between `vmin' and `vmax'.

  for (int i=0; i < nBins; i++) { histogram[i] = 0; }

  int bin;
  double dbin = (vmax - vmin)/nBins;
  for (int i=0; i < nValues; i++) {
    bin = (values[i] - vmin)/dbin;
    if ( bin < 0 || bin >= nBins ) { continue; }
    histogram[bin] += 1;
  }
}

extern "C" void getDistances(
  int N, double L, double* x, double* y, double* diameters, double *distances,
  bool scale_diameter) {
  // Compute distances between the `N' particles of a system of size `L', with
  // x-axis positions given by `x' and y-axis positions given by `y'.
  // Distances are rescaled by the sum of the radii of the particles in the pair
  // if `scale_diameter'.
  // NOTE: distances must have (at least) N(N - 1)/2 entries.

  double dist;
  int pair = 0;
  for (int i=0; i < N; i++) {
    for (int j=i + 1; j < N; j++) {
      dist = sqrt(
        pow(algDistPeriod(x[i], x[j], L), 2)
        + pow(algDistPeriod(y[i], y[j], L), 2));
      if ( scale_diameter ) {
        distances[pair] = dist/((diameters[i] + diameters[j])/2);
      }
      else {
        distances[pair] = dist;
      }
      pair++;
    }
  }
}

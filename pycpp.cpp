#include <math.h>
#include <vector>

#include "maths.hpp"
#include "pycpp.hpp"

// HISTOGRAMS

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

// DISTANCES

double getDistance(
  double const& x0, double const& y0, double const& x1, double const& y1,
  double const& L) {
  // Compute distance between (`x0', `y0') and (`x1', `y1') in a periodic square
  // system of size L.

  return sqrt(
    pow(algDistPeriod(x0, x1, L), 2)
    + pow(algDistPeriod(y0, y1, L), 2));
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
      dist = getDistance(x[i], y[i], x[j], y[j], L);
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

// GRIDS

extern "C" void toGrid(
  int N, double L, double* x, double *y, double* values,
  int nBoxes, double* grid, bool average) {
  // Maps square (sub-)system of `N' particles with positions (`x', `y') centred
  // around 0 and of (cropped) size `L' to a flattened square `grid' of size
  // `nBoxes'^2, and associates to each box the sum or averaged value of the
  // (`N',)-array `values'.

  int nBoxesSq = pow(nBoxes, 2);

  for (int bin=0; bin < nBoxesSq; bin++) { grid[bin] = 0; }

  int bin;
  double dbox = L/nBoxes;
  std::vector<int> occupancy(nBoxesSq, 0);
  for (int i=0; i < N; i++) {
    bin = nBoxes*((int) ((x[i] + L/2)/dbox)) + ((int) ((y[i] + L/2)/dbox));
    if ( bin < 0 || bin >= nBoxesSq ) { continue; }
    grid[bin] += values[i];
    occupancy[bin] += 1;
  }

  if ( average ) {
    for (int bin=0; bin < nBoxesSq; bin++) {
      if ( occupancy[bin] > 0 ) { grid[bin] /= occupancy[bin]; }
    }
  }
}

extern "C" void g2Dto1Dgrid(
  int nBoxes, double* g2D, double* grid,
  double* g1D, double* radii, int* nRadii) {
  // Returns cylindrical average of flattened square grid `g2D' of size
  // `nBoxes'^2 with values of radii given by flatten square grid `grid' of same
  // size, as `nRadii'[0] values of `g1D' at corresponding `radii'.

  int nBoxesSq = pow(nBoxes, 2);

  nRadii[0] = 0;
  std::map<double, double> g1Ddic;
  std::map<double, int> occupancy;

  // compute 1D grid
  for (int i=0; i < nBoxesSq; i++) {
    if ( g1Ddic.find(grid[i]) == g1Ddic.end() ) {
      g1Ddic[grid[i]] = g2D[i];
      occupancy[grid[i]] = 1;
    }
    else {
      g1Ddic[grid[i]] += g2D[i];
      occupancy[grid[i]] += 1;
    }
  }

  // add result to input arguments
  std::vector<double> radii_(0);
  for(auto it = g1Ddic.begin(); it != g1Ddic.end(); it++) {
    radii_.push_back(it->first);
  }
  std::sort(radii_.begin(), radii_.end());
  nRadii[0] = radii_.size();
  for (int bin=0; bin < nRadii[0]; bin++) {
    g1D[bin] = g1Ddic[radii_[bin]]/occupancy[radii_[bin]];
    radii[bin] = radii_[bin];
  }
}

extern "C" void g2Dto1Dgridhist(
  int nBoxes, double* g2D, double* grid,
  int nBins, double vmin, double vmax, double* g1D, double* g1Dstd) {
  // Returns cylindrical average of flattened square grid `g2D' of size
  // `nBoxes'^2 with values of radii given by flatten square grid `grid' of same
  // size, as histogram `g1D' with `nBins' between `vmin' and `vmax' and
  // standard variation on this measure `g1Dstd'.

  int bin;
  for (bin=0; bin < nBins; bin++) { g1D[bin] = 0; g1Dstd[bin] = 0; }

  double dbin = (vmax - vmin)/nBins;
  std::vector<int> occupancy(nBins, 0);
  for (int i=0; i < pow(nBoxes, 2); i++) {
    bin = (grid[i] - vmin)/dbin;
    if ( bin < 0 || bin >= nBins ) { continue; }
    g1D[bin] += g2D[i];
    g1Dstd[bin] += pow(g2D[i], 2);
    occupancy[bin] += 1;
  }

  for (bin=0; bin < nBins; bin++) {
    if ( occupancy[bin] > 0 ) {
      g1D[bin] /= occupancy[bin];
      g1Dstd[bin] = sqrt(g1Dstd[bin]/occupancy[bin] - pow(g1D[bin], 2));
    }
  }
}

// CORRELATIONS

extern "C" void getRadialCorrelations(
  int N, double L, double* x, double* y, int dim, double** values,
  int nBins, double rmin, double rmax, double* correlations,
  bool rescale_pair_distribution) {
  // Compute radial correlations between the (`dim',) float arrays `values'
  // associated to each of the `N' particles of a system of size `L', with
  // x-axis positions given by `x' and y-axis positions given by `y'.
  // Correlations are computed on the interval between `rmin' (included) and
  // `rmax' (excluded) with `nBins' bins.
  // Correlations are rescaled by pair distribution function (for bins > 0) if
  // `rescale_pair_distribution'.

  for (int i=0; i < nBins; i++) { correlations[i] = 0; }
  std::vector<int> occupancy(nBins, 0);
  int nPairs = 0; // number of pairs

  int bin;
  double dbin = (rmax - rmin)/nBins;
  for (int i=0; i < N; i++) {
    for (int j=i; j < N; j++) {
      if ( i != j ) { nPairs++; }
      bin = (getDistance(x[i], y[i], x[j], y[j], L) - rmin)/dbin;
      if ( bin < 0 || bin >= nBins ) { continue; }
      for (int d=0; d < dim; d++) {
        correlations[bin] += values[i][d]*values[j][d];
      }
      occupancy[bin] += 1;
    }
  }

  for (int i=0; i < nBins; i++) {
    if ( occupancy[i] > 0 ) {
      // mean over computed values
      correlations[i] /= occupancy[i];
      // correction by pair distribution function
      if ( ! rescale_pair_distribution ) { continue; }
      if ( i == 0 && rmin == 0 ) { continue; } // do not consider 0th bin
      correlations[i] /=
        ((double) occupancy[i]/nPairs) // histogram value
        *pow(L, 2)/((rmax - rmin)/nBins) // normalisation
        /(2*M_PI*(rmin + i*(rmax - rmin)/nBins)); // radial projection
    }
  }
}

extern "C" void getVelocitiesOriCor(
  int N, double L, double* x, double* y, double* vx, double* vy,
  double* correlations,
  double sigma) {
  // Compute radial correlations of orientation of velocities (`vx', `vy')
  // associated to each of the `N' particles of a system of size `L', with
  // x-axis positions given by `x' and y-axis positions given by `y', and mean
  // diameter `sigma'.
  // (see https://yketa.github.io/PhD_Wiki/#Flow%20characteristics)

  int nBins = (L/2)/sigma;

  std::vector<std::vector<double>>
    angularDistances(N, std::vector<double>(nBins, 0));
  std::vector<std::vector<int>>
    occupancy(N, std::vector<int>(nBins, 0));

  int bin;
  double abspsiij, dij;
  for (int i=0; i < N; i++) {
    for (int j=i + 1; j < N; j++) {
      bin = getDistance(x[i], y[i], x[j], y[j], L)/sigma;
      if ( bin >= nBins ) { continue; }
      abspsiij =
        abs(getAngleVector(vx[i], vy[i]) - getAngleVector(vx[j], vy[j]));
      dij = std::min(abspsiij, 2*M_PI - abspsiij);
      angularDistances[i][bin] += dij;
      occupancy[i][bin] += 1;
      angularDistances[j][bin] += dij;
      occupancy[j][bin] += 1;
    }
  }

  for (int bin=0; bin < nBins; bin++) {
    correlations[bin] = 1;
    for (int i=0; i < N; i++) {
      if ( occupancy[i][bin] == 0 ) { continue; }
      correlations[bin] -= 2*angularDistances[i][bin]/occupancy[i][bin]/M_PI/N;
    }
  }
}

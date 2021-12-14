#include <math.h>
#include <vector>

#include "dat.hpp"
#include "maths.hpp"
#include "pycpp.hpp"
#include "readwrite.hpp"

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
  // Put the `nValues' `values' in an `histogram' with `nBins' bins linearly
  // spaced between `vmin' and `vmax'.

  for (int i=0; i < nBins; i++) { histogram[i] = 0; }

  int bin;
  double dbin = (vmax - vmin)/nBins;
  for (int i=0; i < nValues; i++) {
    bin = (values[i] - vmin)/dbin;
    if ( bin < 0 || bin >= nBins ) { continue; }
    histogram[bin] += 1;
  }
}

// POSITIONS AND DISPLACEMENTS
// Compute vectors of positions and displacements for faster computation in C++
// from .datN files.

std::vector<std::vector<std::vector<double>>> getPositions(
  std::string filename,
  int const& nTime0, int* const& time0) {
  // Compute positions at the `nTime0' initial times `time0' from the .datN file
  // `filename'.

  DatN dat(filename);
  const int N = dat.getNumberParticles();
  std::vector<std::vector<std::vector<double>>> positions
    (nTime0,
      std::vector<std::vector<double>>(N,
        std::vector<double>(2)));

  for (int t0=0; t0 < nTime0; t0++) {
    for (int i=0; i < N; i++) {
      for (int dim=0; dim < 2; dim++) {
        positions[t0][i][dim] = dat.getPosition(time0[t0], i, dim, false);
      }
    }
  }

  return positions;
}

std::vector<std::vector<std::vector<std::vector<double>>>> getDisplacements(
  std::string filename,
  int const& nTime0, int* const& time0, int const& nDt, int* const& dt,
  bool remove_cm) {
  // Compute displacements from the `nTime0' initial times `time0' with the
  // `nDt' lag times `dt' from the .datN file `filename'.
  // Remove centre of mass displacement if `remove_cm'.

  DatN dat(filename);
  const int N = dat.getNumberParticles();
  std::vector<std::vector<std::vector<std::vector<double>>>> displacements
    (nTime0,
      std::vector<std::vector<std::vector<double>>>(nDt,
        std::vector<std::vector<double>>(N,
          std::vector<double>(2))));

  std::vector<double> dispCM(2);
  for (int t0=0; t0 < nTime0; t0++) {
    for (int t=0; t < nDt; t++) {
      for (int i=0; i < N; i++) {
        for (int dim=0; dim < 2; dim++) {
          displacements[t0][t][i][dim] =
            dat.getPosition(time0[t0] + dt[t], i, dim, true)
              - dat.getPosition(time0[t0], i, dim, true);
        }
      }
      if ( remove_cm ) {
        // compute centre of mass displacement
        for (int dim=0; dim < 2; dim++) {
          dispCM[dim] = 0;
          for (int i=0; i < N; i++) {
            dispCM[dim] += displacements[t0][t][i][dim];
          }
          dispCM[dim] /= N;
        }
        // remove centre of mass displacement
        for (int i=0; i < N; i++) {
          for (int dim=0; dim < 2; dim++) {
            displacements[t0][t][i][dim] -= dispCM[dim];
          }
        }
      }
    }
  }

  return displacements;
}

// DISTANCES

double getDistance(
  double const& x0, double const& y0, double const& x1, double const& y1,
  double const& L) {
  // Compute distance between (`x0', `y0') and (`x1', `y1') in a periodic square
  // system of size `L'.

  return sqrt(
    pow(algDistPeriod(x0, x1, L), 2)
    + pow(algDistPeriod(y0, y1, L), 2));
}

bool withinDistance(
  double const& A, double* const& diameters, int const& i, int const& j,
  double const& distance) {
  // Compute if particles `i' and `j' separated by `distance' are within
  // distance `A' relative to their diameters.

  return ( distance <= A*((diameters[i] + diameters[j])/2) );
}

extern "C" int pairIndex(int i, int j, int N) {
  // For `N' particles, return a unique pair index for the couples (`i', `j')
  // and (`j', `i') in {0, ..., `N'(`N' + 1)/2 - 1}.
  // (adapted from Paul Mangold)

  int min = std::min(i, j);
  int max = std::max(i, j);

  return (min + N - 1 - max)*(min + N - max)/2 + min;
}

extern "C" void getDifferences(
  int N, double L, double* x, double* y, double *diameters,
  double* differences_x, double* differences_y,
  bool scale_diameter) {
  // Compute position differences between the `N' particles of a system of size
  // `L', with x-axis positions given by `x' and y-axis positions given by `y'.
  // Differences are rescaled by the sum of the radii of the particles in the
  // pair if `scale_diameter'.
  // NOTE: `differences_x' and `differences_y' must have (at least)
  //       `N'(`N' - 1)/2 entries.

  double diff_x, diff_y;
  int pair;
  for (int i=0; i < N; i++) {
    for (int j=i + 1; j < N; j++) {
      diff_x = algDistPeriod(x[i], x[j], L);
      diff_y = algDistPeriod(y[i], y[j], L);
      pair = pairIndex(i, j, N);
      if ( scale_diameter ) {
        differences_x[pair] = diff_x/((diameters[i] + diameters[j])/2);
        differences_y[pair] = diff_y/((diameters[i] + diameters[j])/2);
      }
      else {
        differences_x[pair] = diff_x;
        differences_y[pair] = diff_y;
      }
    }
  }
}

extern "C" void getDistances(
  int N, double L, double* x, double* y, double* diameters, double* distances,
  bool scale_diameter) {
  // Compute distances between the `N' particles of a system of size `L', with
  // x-axis positions given by `x' and y-axis positions given by `y'.
  // Distances are rescaled by the sum of the radii of the particles in the pair
  // if `scale_diameter'.
  // NOTE: `distances' must have (at least) `N'(`N' - 1)/2 entries.

  double dist;
  int pair;
  for (int i=0; i < N; i++) {
    for (int j=i + 1; j < N; j++) {
      dist = getDistance(x[i], y[i], x[j], y[j], L);
      pair = pairIndex(i, j, N);
      if ( scale_diameter ) {
        distances[pair] = dist/((diameters[i] + diameters[j])/2);
      }
      else {
        distances[pair] = dist;
      }
    }
  }
}

extern "C" void getOrientationNeighbours(
  int N, double A1, double* diameters, double* distances,
  double* dx, double* dy,
  int* oneighbours) {
  // Compute for each of the `N' particles the number of other particles, at
  // distance lesser than `A1' relative to their average diameter in
  // `distances', with the same orientation of displacement (`dx', `dy').
  // NOTE: `distances' must have at least `N'(`N' - 1)/2 entries and follow
  //       the indexing of pairs given by pairIndex (such as returned by
  //       getDistances).
  // NOTE: `oneighbours' must have at least `N' entries.

  for (int i=0; i < N; i++) { oneighbours[i] = 0; }

  double prod;
  for (int i=0; i < N; i++) {
    for (int j=i + 1; j < N; j++) {
      if (
        withinDistance(A1, diameters, i, j, distances[pairIndex(i, j, N)]) ) {
        prod = (dx[i]*dx[j] + dy[i]*dy[j])/
          sqrt(
            (pow(dx[i], 2.0) + pow(dy[i], 2.0))
            *(pow(dx[j], 2.0) + pow(dy[j], 2.0)));
        if ( prod >= 0.5 ) {
          oneighbours[i]++;
          oneighbours[j]++;
        }
      }
    }
  }
}

extern "C" void getBrokenBonds(
  int N, double A1, double A2, double* diameters,
  double* distances0, double* distances1,
  int* brokenBonds, bool* brokenPairs) {
  // Compute for each of `N' particles the number of other particles which are
  // at distance lesser than `A1' in `distances0' and distance greater than `A2'
  // in `distances1' relative to their average diameter.
  // Broken pair indices are flagged as true in `brokenPairs'.
  // NOTE: `distances0' and `distances1' must have at least `N'(`N' - 1)/2
  //       entries and follow the indexing of pairs given by pairIndex (such as
  //       returned by getDistances).
  // NOTE: `brokenBonds' must have at least `N' entries.
  // NOTE: `brokenPairs' must have at least `N(N - 1)/2' entries.

  for (int i=0; i < N; i++) { brokenBonds[i] = 0; }
  for (int index=0; index < N*(N- 1)/2; index++) { brokenPairs[index] = false; }

  int index;
  for (int i=0; i < N; i++) {
    for (int j=i + 1; j < N; j++) {
      index = pairIndex(i, j, N);
      if (
        withinDistance(
          A1, diameters, i, j, distances0[index])     // bonded at time0
        && !withinDistance(
          A2, diameters, i, j, distances1[index]) ) { // unbonded at time1
        brokenBonds[i]++;
        brokenBonds[j]++;
        brokenPairs[index] = true;
      }
    }
  }
}

extern "C" void getVanHoveDistances(
  int N, double L, double* x, double* y, double* dx, double* dy,
  double* distances) {
  // Compute van Hove distances between the `N' particles of a system of size
  // `L', with x-axis positions given by `x' and y-axis positions given by `y',
  // and x-axis displacements given by `dx' and y-axis displacements given by
  // `dy'.

  double dist[2];
  for (int i=0; i < N; i++) {
    for (int j=0; j < N; j++) {
      dist[0] = algDistPeriod(x[i], x[j], L) + dx[j];
      dist[1] = algDistPeriod(y[i], y[j], L) + dy[j];
      distances[i*N + j] = sqrt(pow(dist[0], 2) + pow(dist[1], 2));
    }
  }
}

extern "C" void nonaffineSquaredDisplacement(
  int N, double L, double* x0, double* y0, double* x1, double* y1,
  double A1, double* diameters, double* D2min) {
  // Compute nonaffine squared displacements for `N' particles of a system of
  // size `L', with x-axis initial positions `x0', y-axis initial positions
  // `y0', x-axis final positions `x1', and y-axis final positions `y1',
  // considering that neighbouring particles are within a distance `A1'
  // relative to their `diameters'.

  std::vector<double> distances0(N*(N - 1)/2);
  getDistances(N, L, x0, y0, diameters, &(distances0[0]), false);

  double* r0[2] = {x0, y0};
  double* r1[2] = {x1, y1};

  std::vector<std::vector<std::vector<long double>>> X(N);
  std::vector<std::vector<std::vector<long double>>> Y(N);
  for (int i=0; i < N; i++) {
    X[i] = {{0.0, 0.0}, {0.0, 0.0}};
    Y[i] = {{0.0, 0.0}, {0.0, 0.0}};
  }
  std::vector<std::vector<long double>> Yinv;

  double term;
  for (int i=0; i < N; i++) {
    for (int j=i + 1; j < N; j++) {
      if ( !withinDistance( // initially neighbouring particles
        A1, diameters, i, j, distances0[pairIndex(i, j, N)]) ) { continue; }

      for (int alpha=0; alpha < 2; alpha++) {
        for (int beta=0; beta < 2; beta++) {

          term =
            algDistPeriod(r1[alpha][i], r1[alpha][j], L)
            *algDistPeriod(r0[beta][i], r0[beta][j], L);
          X[i][alpha][beta] += term;
          X[j][alpha][beta] += term;

          term =
            algDistPeriod(r0[alpha][i], r0[alpha][j], L)
            *algDistPeriod(r0[beta][i], r0[beta][j], L);
          Y[i][alpha][beta] += term;
          Y[j][alpha][beta] += term;

        }
      }

    }
  }

  double sqrtterm;
  double dr0;
  for (int i=0; i < N; i++) {
    D2min[i] = 0;
    try {
      Yinv = invert2x2<long double>(Y[i]);
    }
    catch (const std::invalid_argument&) {
      if ( !(X[i][0][0] == 0.0 && X[i][0][1] == 0.0
        && X[i][1][0] == 0.0 && X[i][1][1] == 0.0) ) {
        throw std::invalid_argument("Impossible to compute D2min.");
      }
      else {
        Yinv = {{0, 0}, {0, 0}};
      }
    }
    for (int j=0; j < N; j++) {
      if ( !withinDistance( // initially neighbouring particles
        A1, diameters, i, j, distances0[pairIndex(i, j, N)]) ) { continue; }

      for (int mu=0; mu < 2; mu++) {

        sqrtterm = 0;
        sqrtterm += algDistPeriod(r1[mu][i], r1[mu][j], L);
        for (int nu=0; nu < 2; nu++) {
          dr0 = algDistPeriod(r0[nu][i], r0[nu][j], L);
          for (int gamma=0; gamma < 2; gamma++) {
            sqrtterm -= X[i][mu][gamma]*Yinv[gamma][nu]*dr0;
          }
        }
        D2min[i] += pow(sqrtterm, 2);

      }

    }
  }
}

extern "C" void pairDistribution(
  int nBins, double vmin, double vmax, double* histogram,
  int N, double L, double* x, double* y, double* diameters,
  bool scale_diameter) {
  // Compute pair distribution function as `histogram' with `nBins' bins
  // linearly spaced between `vmin' and `vmax' from the distances between the
  // `N' particles of a system of size `L', with x-axis positions given by `x'
  // and y-axis positions given by `y'.
  // Distances are rescaled by the sum of the radii of the particles in the pair
  // if `scale_diameter'.

  for (int i=0; i < nBins; i++) { histogram[i] = 0; }

  int bin;
  double dbin = (vmax - vmin)/nBins;
  double dist;
  int nPairs = 0;
  for (int i=0; i < N; i++) {
    for (int j=i + 1; j < N; j++) {
      nPairs++;
      dist = getDistance(x[i], y[i], x[j], y[j], L);
      if ( scale_diameter ) { dist /= (diameters[i] + diameters[j])/2; }
      bin = (dist - vmin)/dbin;
      if ( bin < 0 || bin >= nBins ) { continue; }
      histogram[bin] += 1;
    }
  }

  for (int i=0; i < nBins; i++) { histogram[i] /= nPairs; }
}

extern "C" void S4Fs(
  const char* filename,
  int nTime0, int* time0, int nDt, int* dt,
  int nq, double *qx, double *qy, int nk, double *kx, double *ky,
  double *S4, double *S4var) {
  // Compute from the .datN file `filename' the four-point structure factor of
  // the real part of the self-intermediate scattering functions, computed at
  // the `nk' wave-vectors (`kx', `ky'), along the `nq' wave-vectors
  // (`qx', `qy'), from the positions and displacements from the `nTime0'
  // initial times `time0' with the `nDt' lag times `dt'.
  // Means are saved in `S4' and variances are saved in `S4var', for each value
  // of the lag time.

  DatN dat(filename);
  const int N = dat.getNumberParticles();

  const std::vector<std::vector<std::vector<double>>> positions
    = getPositions(filename, nTime0, time0);
  const std::vector<std::vector<std::vector<std::vector<double>>>> displacements
    = getDisplacements(filename, nTime0, time0, nDt, dt, true);

  std::vector<std::vector<double>> s4
    (nDt, std::vector<double>(nq));
  std::vector<std::vector<double>> s4sq
    (nDt, std::vector<double>(nq));
  for (int t=0; t < nDt; t++) {
    S4[t] = 0;
    S4var[t] = 0;
    for (int q=0; q < nq; q++) {
      s4[t][q] = 0;
      s4sq[t][q] = 0;
    }
  }

  std::vector<double> term(2);
  std::vector<double> Fs(N);
  for (int t0=0; t0 < nTime0; t0++) {
    for (int t=0; t < nDt; t++) {
      for (int i=0; i < N; i++) {
        Fs[i] = 0;
        for (int k=0; k < nk; k++) {
          Fs[i] += cos(
            kx[k]*displacements[t0][t][i][0]
            + ky[k]*displacements[t0][t][i][1]);
        }
        Fs[i] /= nk;
      }
      for (int q=0; q < nq; q++) {
        term = {0.0, 0.0};
        for (int part=0; part < 2; part++) {
          for (int i=0; i < N; i++) {
            term[part] += Fs[i]
              *cos(qx[q]*positions[t0][i][0] + qy[q]*positions[t0][i][1]
                - part*M_PI/2);
          }
          term[part] = (pow(term[part], 2.0)*nk)/N;
        }
        s4[t][q] += term[0] + term[1];
        s4sq[t][q] += pow(term[0] + term[1], 2.0);
      }
    }
  }

  for (int t=0; t < nDt; t++) {
    for (int q=0; q < nq; q++) {
      S4[t] += s4[t][q];
      S4var[t] += s4sq[t][q] - s4[t][q]*s4[t][q]/nTime0;
    }
    S4[t] /= nTime0*nq;
    S4var[t] /= nTime0*nq;
  }
}

extern "C" void getLocalParticleDensity(
  int N, double L, double a, double* x, double* y, double* diameters,
  double* densities) {
  // Compute for each of the `N' particles of a system of size `L', with x-axis
  // positions given by `x' and y-axis positions given by `y', the sum of the
  // areas of the particles in a box of size `a' centred on the particle divided
  // by the area of the box.
  // Particle areas are computed with a factor 2^(1/6) on diameters.

  for (int i=0; i < N; i++) { densities[i] = 0; }

  for (int i=0; i < N; i++) {
    densities[i] += (M_PI/4)*pow(pow(2., 1./6.)*diameters[i], 2)/pow(a, 2);
    for (int j=i + 1; j < N; j++) {
      if (
        abs(algDistPeriod(x[i], x[j], L)) <= a/2
        && abs(algDistPeriod(y[i], y[j], L)) <= a/2 ) {
        densities[i] += (M_PI/4)*pow(pow(2., 1./6.)*diameters[j], 2)/pow(a, 2);
        densities[j] += (M_PI/4)*pow(pow(2., 1./6.)*diameters[i], 2)/pow(a, 2);
      }
    }
  }
}

extern "C" void isNotInBubble(
  int N, double L, double philim, double dlim,
  double* x, double* y, double* densities,
  bool* notInBubble) {
  // Determine which of the `N' particles of a system of size `L', with x-axis
  // positions given by `x' and y-axis positions given by `y', are not within
  // distance `dlim' of particles with `densities' below `philim'.

  for (int i=0; i < N; i++) { notInBubble[i] = true; }

  for (int i=0; i < N; i++) {
    if ( densities[i] < philim ) {
      notInBubble[i] = false;
      for (int j=0; j < N; j++) {
        if ( notInBubble[j] ) {
          if ( getDistance(x[i], y[i], x[j], y[j], L) < dlim ) {
            notInBubble[j] = false;
          }
        }
      }
    }
  }
}

// GRIDS

extern "C" void toGrid(
  int N, double L, double* x, double* y, double* values,
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
  int N, double L, double* x, double* y, int dim,
  double** values1, double** values2,
  int nBins, double rmin, double rmax, double* correlations,
  bool rescale_pair_distribution) {
  // Compute radial correlations between the (`dim',) float arrays `values1'
  // and `values2' associated to each of the `N' particles of a system of size
  // `L', with x-axis positions given by `x' and y-axis positions given by `y'.
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
        correlations[bin] +=
          (values1[i][d]*values2[j][d] + values1[j][d]*values2[i][d])/2;
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

// READ

extern "C" void readDouble(
  const char* filename, int nTargets, long int* targets, double* out) {
  // Read `nTargets' doubles in `filename' at `targets' and output in `out'.

  Read read(filename);
  for (int i=0; i < nTargets; i++) {
    read.read<double>(&(out[i]), targets[i]);
  }
  read.close();
}

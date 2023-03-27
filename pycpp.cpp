#include <math.h>
#include <complex>
#include <vector>
#include <tuple>
#include <assert.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <gsl/gsl_linalg.h>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include "add.hpp"
#include "dat.hpp"
#include "maths.hpp"
#include "particle.hpp"
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
    if ( values[i] < vmin || values[i] >= vmax ) { continue; }
    bin = (values[i] - vmin)/dbin;
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

  std::vector<double> dispCM(2, 0);
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

pybind11::array_t<double> getDisplacementsPYB(
  std::string const& filename,
  pybind11::array_t<int> time0, pybind11::array_t<int> deltat,
  bool const& remove_cm) {
  // Compute displacements from the initial times `time0' with the lag times
  // `dt' from the .datN file `filename'.
  // Remove centre of mass displacement if `remove_cm'.

  pybind11::buffer_info t0_buf = time0.request();
  pybind11::buffer_info dt_buf = deltat.request();

  std::vector<std::vector<std::vector<std::vector<double>>>> displacements
    = getDisplacements(
      filename,
      (int) t0_buf.shape[0], (int*) t0_buf.ptr,
      (int) dt_buf.shape[0], (int*) dt_buf.ptr,
      remove_cm);

  pybind11::array_t<double> disp({
    (int) t0_buf.shape[0],
    (int) dt_buf.shape[0],
    (int) displacements[0][0].size(),
    2}); // displacements
  auto d = disp.mutable_unchecked<4>();
  for (int t0=0; t0 < (int) disp.request().shape[0]; t0++) {
    for (int t=0; t < (int) disp.request().shape[1]; t++) {
      for (int i=0; i < (int) disp.request().shape[2]; i++) {
        for (int dim=0; dim < (int) disp.request().shape[3]; dim++) {
          d(t0, t, i, dim) = displacements[t0][t][i][dim];
        }
      }
    }
  }

  return disp;
}

std::vector<std::vector<pybind11::array_t<double>>> getDisplacementsPair(
  std::string const& filename,
  pybind11::array_t<int> time0, pybind11::array_t<int> deltat,
  double const& a1) {
  // Compute differences of displacements of initially paired particles (i.e. at
  // distance lesser than `a1' mean diameter) from the initial times `time0'
  // with the lag times `dt' from the .datN file `filename'.

  DatN dat(filename, false, true);
  const int N = dat.getNumberParticles();
  const double L = dat.getSystemSize();
  const std::vector<double> sigma = dat.getDiameters();
  std::vector<std::vector<double>> positions(N, std::vector<double>(2, 0));

  CellList cellList(N, L, a1*(*std::max_element(sigma.begin(), sigma.end())));

  auto t0 = time0.unchecked<1>();
  pybind11::buffer_info t0_buf = time0.request();
  auto t = deltat.unchecked<1>();
  pybind11::buffer_info dt_buf = deltat.request();

  std::vector<pybind11::array_t<double>> displacementsPair(0);
  std::vector<pybind11::array_t<double>> pairs(0);
  std::vector<std::array<int, 2>> neighbours(0);
  std::vector<std::vector<double>> displacements(N, std::vector<double>(2, 0));
  double dist;
  double sigmaij;
  double diff[2];
  for (int it0=0; it0 < t0_buf.shape[0]; it0++) {

    // get positions
    for (int i=0; i < N; i++) {
      for (int dim=0; dim < 2; dim++) {
        positions[i][dim] = dat.getPosition(t0(it0), i, dim, false);
      }
    }

    // get neighbours
    neighbours.clear();
    cellList.listConstructor<std::vector<double>>(positions);
    cellList.pairs(
      [&sigma, &positions, &L, &a1, &dist, &sigmaij, &diff, &neighbours]
      (int const& index1, int const& index2) { // do for each individual pair
        // rescaled diameter
        sigmaij = (sigma[index1] + sigma[index2])/2;
        // distance
        dist = dist2DPeriod(
          &(positions[index1][0]), &(positions[index2][0]), L, &diff[0]);
        if ( dist/sigmaij < a1 ) { // particles are bonded
          neighbours.push_back({index1, index2});
        }
      });
    pairs.push_back(pybind11::array_t<double>(
      {(int) neighbours.size(), 2}));
    auto p = pairs[it0].mutable_unchecked<2>();
    for (int n=0; n < (int) neighbours.size(); n++) {
      for (int i=0; i < 2; i++) {
        p(n, i) = neighbours[n][i];
      }
    }
    displacementsPair.push_back(pybind11::array_t<double>(
      {(int) dt_buf.shape[0], (int) neighbours.size(), 2}));
    auto dP = displacementsPair[it0].mutable_unchecked<3>();

    // for each lag time
    for (int it=0; it < dt_buf.shape[0]; it++) {

      // compute displacements
      for (int i=0; i < N; i++) {
        for (int dim=0; dim < 2; dim++) {
          displacements[i][dim] =
            dat.getPosition(t0(it0) + t(it), i, dim, true)
            - dat.getPosition(t0(it0), i, dim, true);
        }
      }
      // for each pair
      for (int n=0; n < (int) neighbours.size(); n++) {
        for (int dim=0; dim < 2; dim++) {
          dP(it, n, dim) =
            displacements[neighbours[n][1]][dim]
            - displacements[neighbours[n][0]][dim];
        }
      }

    }
  }

  std::vector<std::vector<pybind11::array_t<double>>> out(0);
  out.push_back(displacementsPair);
  out.push_back(pairs);
  return out;
}

std::vector<std::vector<pybind11::array_t<double>>> getSeparationsPair(
  std::string const& filename,
  pybind11::array_t<int> time0, pybind11::array_t<int> deltat,
  double const& a1, bool const& scale) {
  // Compute differences of positions of initially paired particles (i.e. at
  // distance lesser than `a1' mean diameter) from the initial times `time0'
  // with the lag times `dt' from the .datN file `filename'.
  // Scale separation by average diameter if `scale'.

  DatN dat(filename, false, true);
  const int N = dat.getNumberParticles();
  const double L = dat.getSystemSize();
  const std::vector<double> sigma = dat.getDiameters();
  std::vector<std::vector<double>> positions(N, std::vector<double>(2, 0));

  CellList cellList(N, L, a1*(*std::max_element(sigma.begin(), sigma.end())));

  auto t0 = time0.unchecked<1>();
  pybind11::buffer_info t0_buf = time0.request();
  auto t = deltat.unchecked<1>();
  pybind11::buffer_info dt_buf = deltat.request();

  std::vector<pybind11::array_t<double>> separationsPair(0);
  std::vector<std::array<double, 2>> separations(0);
  std::vector<pybind11::array_t<double>> pairs(0);
  std::vector<std::array<int, 2>> neighbours(0);
  std::vector<std::vector<double>> displacements(N, std::vector<double>(2, 0));
  double dist;
  double sigmaij;
  double diff[2];
  for (int it0=0; it0 < t0_buf.shape[0]; it0++) {

    // get positions
    for (int i=0; i < N; i++) {
      for (int dim=0; dim < 2; dim++) {
        positions[i][dim] = dat.getPosition(t0(it0), i, dim, false);
      }
    }

    // get neighbours
    separations.clear();
    neighbours.clear();
    cellList.listConstructor<std::vector<double>>(positions);
    cellList.pairs(
      [&sigma, &positions, &L, &a1, &dist, &sigmaij, &diff,
        &separations, &neighbours]
      (int const& index1, int const& index2) { // do for each individual pair
        // rescaled diameter
        sigmaij = (sigma[index1] + sigma[index2])/2;
        // distance
        dist = dist2DPeriod(
          &(positions[index1][0]), &(positions[index2][0]), L, &diff[0]);
        if ( dist/sigmaij < a1 ) { // particles are bonded
          separations.push_back({diff[0], diff[1]});
          neighbours.push_back({index1, index2});
        }
      });
    pairs.push_back(pybind11::array_t<double>(
      {(int) neighbours.size(), 2}));
    auto p = pairs[it0].mutable_unchecked<2>();
    for (int n=0; n < (int) neighbours.size(); n++) {
      for (int i=0; i < 2; i++) {
        p(n, i) = neighbours[n][i];
      }
    }
    separationsPair.push_back(pybind11::array_t<double>(
      {(int) dt_buf.shape[0], (int) neighbours.size(), 2}));
    auto sP = separationsPair[it0].mutable_unchecked<3>();

    // for each lag time
    for (int it=0; it < dt_buf.shape[0]; it++) {

      // compute displacements
      for (int i=0; i < N; i++) {
        for (int dim=0; dim < 2; dim++) {
          displacements[i][dim] =
            dat.getPosition(t0(it0) + t(it), i, dim, true)
            - dat.getPosition(t0(it0), i, dim, true);
        }
      }
      // for each pair
      for (int n=0; n < (int) neighbours.size(); n++) {
        for (int dim=0; dim < 2; dim++) {
          sP(it, n, dim) =
            separations[n][dim]
            + displacements[neighbours[n][1]][dim]
            - displacements[neighbours[n][0]][dim];
          if ( scale ) {
            sP(it, n, dim) /=
              (sigma[neighbours[n][0]] + sigma[neighbours[n][1]])/2;
          }
        }
      }

    }
  }

  std::vector<std::vector<pybind11::array_t<double>>> out(0);
  out.push_back(separationsPair);
  out.push_back(pairs);
  return out;
}

std::vector<pybind11::array_t<double>> getVelocityCorrelationPair(
  std::string const& filename,
  pybind11::array_t<int> time0, pybind11::array_t<int> deltat,
  double const& a1) {
  // Compute difference between velocity autocorrelation in time and correlation
  // of velocities of initially neighbouring particles.
  // Computes paired particles (i.e. at distance lesser than `a1' mean diameter)
  // from the initial times `time0' with the lag times `dt' from the .datN file
  // `filename'.

  DatN dat(filename, false, true);
  const int N = dat.getNumberParticles();
  const double L = dat.getSystemSize();
  const std::vector<double> sigma = dat.getDiameters();
  std::vector<std::vector<double>> positions(N, std::vector<double>(2, 0));

  CellList cellList(N, L, a1*(*std::max_element(sigma.begin(), sigma.end())));

  auto t0 = time0.unchecked<1>();
  pybind11::buffer_info t0_buf = time0.request();
  auto t = deltat.unchecked<1>();
  pybind11::buffer_info dt_buf = deltat.request();

  std::vector<pybind11::array_t<double>> correlationsPair(0);
  std::vector<std::array<int, 2>> neighbours(0);
  std::vector<std::vector<double>> velocities0(N, std::vector<double>(2, 0));
  std::vector<std::vector<double>> velocities1(N, std::vector<double>(2, 0));
  double dist;
  double sigmaij;
  double diff[2];
  for (int it0=0; it0 < t0_buf.shape[0]; it0++) {

    // get positions and velocities
    for (int i=0; i < N; i++) {
      for (int dim=0; dim < 2; dim++) {
        positions[i][dim] = dat.getPosition(t0(it0), i, dim, false);
        velocities0[i][dim] = dat.getVelocity(t0(it0), i, dim);
      }
    }

    // get neighbours
    neighbours.clear();
    cellList.listConstructor<std::vector<double>>(positions);
    cellList.pairs(
      [&sigma, &positions, &L, &a1, &dist, &sigmaij, &diff,
        &neighbours]
      (int const& index1, int const& index2) { // do for each individual pair
        // rescaled diameter
        sigmaij = (sigma[index1] + sigma[index2])/2;
        // distance
        dist = dist2DPeriod(
          &(positions[index1][0]), &(positions[index2][0]), L, &diff[0]);
        if ( dist/sigmaij < a1 ) { // particles are bonded
          neighbours.push_back({index1, index2});
        }
      });
    correlationsPair.push_back(pybind11::array_t<double>(
      {(int) dt_buf.shape[0], (int) neighbours.size(), 2}));
    auto cP = correlationsPair[it0].mutable_unchecked<3>();

    // for each lag time
    for (int it=0; it < dt_buf.shape[0]; it++) {

      // compute velocities
      for (int i=0; i < N; i++) {
        for (int dim=0; dim < 2; dim++) {
          velocities1[i][dim] = dat.getVelocity(t0(it0) + t(it), i, dim);
        }
      }
      // for each pair
      for (int n=0; n < (int) neighbours.size(); n++) {
        for (int dim=0; dim < 2; dim++) {
          cP(it, n, dim) =
            velocities0[neighbours[n][0]][dim]
              *velocities1[neighbours[n][0]][dim]
            + velocities0[neighbours[n][1]][dim]
              *velocities1[neighbours[n][1]][dim]
            - velocities0[neighbours[n][0]][dim]
              *velocities1[neighbours[n][1]][dim]
            - velocities0[neighbours[n][1]][dim]
              *velocities1[neighbours[n][0]][dim];
        }
      }

    }
  }

  return correlationsPair;
}

std::vector<pybind11::array_t<double>> getVelocityForcePropulsionCorrelation(
  std::string const& filename,
  pybind11::array_t<int> time0, pybind11::array_t<int> deltat,
  bool Heun=true) {
  // Compute correlations between force and initial velocity, and between
  // propulsion and initial velocity, from the initial times `time0' with the
  // lag times `dt' from the .datN file `filename'.

  DatN dat(filename, false, true);
  const int N = dat.getNumberParticles();
  const double L = dat.getSystemSize();
  const double dt = dat.getTimeStep();
  const double epsilon = dat.getPotentialParameter();
  pybind11::array_t<double> diameters({N});
  auto d = diameters.mutable_unchecked<1>();
  pybind11::array_t<double> positions({N, 2});
  auto r = positions.mutable_unchecked<2>();
  pybind11::array_t<double> propulsions({N, 2});
  auto p = propulsions.mutable_unchecked<2>();
  pybind11::array_t<double> velocities({N, 2});
  auto v = velocities.mutable_unchecked<2>();

  const std::vector<double> s = dat.getDiameters();
  for (int i=0; i < N; i++) { d(i) = s[i]; }

  auto t0 = time0.unchecked<1>();
  pybind11::buffer_info t0_buf = time0.request();
  auto t = deltat.unchecked<1>();
  pybind11::buffer_info dt_buf = deltat.request();

  pybind11::array_t<double> corVelocityForce(
    {t0_buf.shape[0], dt_buf.shape[0]});
  auto cvf = corVelocityForce.mutable_unchecked<2>();
  pybind11::array_t<double> corVelocityPropulsion(
    {t0_buf.shape[0], dt_buf.shape[0]});
  auto cvp = corVelocityPropulsion.mutable_unchecked<2>();
  pybind11::array_t<double> forces;
  double sum_propulsion[2];
  double sum_velocity[2];
  double sum_force[2];
  for (int it0=0; it0 < t0_buf.shape[0]; it0++) {

    // get initial velocity
    for (int dim=0; dim < 2; dim++) {
      sum_velocity[dim] = 0;
    }
    for (int i=0; i < N; i++) {
      for (int dim=0; dim < 2; dim++) {
        v(i, dim) = dat.getVelocity(t0(it0), i, dim);
        sum_velocity[dim] += v(i, dim);
      }
    }
    // remove average
    for (int i=0; i < N; i++) {
      for (int dim=0; dim < 2; dim++) {
        v(i, dim) -= sum_velocity[dim]/N;
      }
    }

    for (int it=0; it < dt_buf.shape[0]; it++) {
      cvf(it0, it) = 0;
      cvp(it0, it) = 0;

      // get positions and propulsions
      for (int dim=0; dim < 2; dim++) {
        sum_propulsion[dim] = 0;
      }
      for (int i=0; i < N; i++) {
        for (int dim=0; dim < 2; dim++) {
          r(i, dim) = dat.getPosition(t0(it0) + t(it), i, dim, false);
          p(i, dim) = dat.getPropulsion(t0(it0) + t(it), i, dim);
        }
      }

      // get forces
      for (int dim=0; dim < 2; dim++) {
        sum_force[dim] = 0;
      }
      forces = getWCAForces(positions, diameters, L);
      auto f = forces.mutable_unchecked<2>();

      // Heun
      if ( Heun ) {
        std::vector<std::vector<double>> forces0(N, std::vector<double>(2, 0));
        for (int i=0; i < N; i++) {
          for (int dim=0; dim < 2; dim++) {
            forces0[i][dim] = f(i, dim);
            r(i, dim) += dt*(epsilon*f(i, dim) + p(i, dim));
          }
        }
        pybind11::array_t<double> forces1 =
          getWCAForces(positions, diameters, L);
        auto f1 = forces1.mutable_unchecked<2>();
        for (int i=0; i < N; i++) {
          for (int dim=0; dim < 2; dim++) {
            f(i, dim) = (f(i, dim) + f1(i, dim))/2;
          }
        }
      }

      // remove averages
      for (int i=0; i < N; i++) {
        for (int dim=0; dim < 2; dim++) {
          sum_force[dim] += f(i, dim);
          sum_propulsion[dim] += p(i, dim);
        }
      }
      for (int i=0; i < N; i++) {
        for (int dim=0; dim < 2; dim++) {
          f(i, dim) = epsilon*(f(i, dim) - sum_force[dim]/N);
          p(i, dim) -= sum_propulsion[dim]/N;
        }
      }

      // compute correlations
      for (int i=0; i < N; i++) {
        for (int dim=0; dim < 2; dim++) {
          cvf(it0, it) += v(i, dim)*f(i, dim);
          cvp(it0, it) += v(i, dim)*p(i, dim);
        }
      }
    }
  }

  // average
  for (int it0=0; it0 < t0_buf.shape[0]; it0++) {
    for (int it=0; it < dt_buf.shape[0]; it++) {
      cvf(it0, it) = cvf(it0, it)/N;
      cvp(it0, it) = cvp(it0, it)/N;
    }
  }

  // output
  std::vector<pybind11::array_t<double>> out(0);
  out.push_back(corVelocityForce);
  out.push_back(corVelocityPropulsion);
  return out;
}

std::tuple<pybind11::array_t<double>, std::vector<pybind11::array_t<double>>>
  getVelocityDifference(
    std::string const& filename, double const& frame,
    int const& nBins, double const& rmin=0, double rmax=0,
    bool const& remove_cm=true) {
  // Compute velocity difference in the radial direction for particles whose
  // distance is in a certain range.

  // INPUT
  DatN dat(filename); // input file
  const int N = dat.getNumberParticles();
  const double L = dat.getSystemSize();
  std::vector<std::vector<double>> positions(N, std::vector<double>(2, 0));
  std::vector<std::vector<double>> velocities(N, std::vector<double>(2, 0));
  double mean_vel[2] = {0, 0};
  for (int i=0; i < N; i++) {
    for (int dim=0; dim < 2; dim++) {
      positions[i][dim] = dat.getPosition(frame, i, dim, false);
      velocities[i][dim] = dat.getVelocity(frame, i, dim);
      mean_vel[dim] += velocities[i][dim]/N;
    }
  }
  if ( remove_cm ) {
    for (int i=0; i < N; i++) {
      for (int dim=0; dim < 2; dim++) {
        velocities[i][dim] -= mean_vel[dim];
      }
    }
  }

  // BINS
  if ( rmax == 0 ) { rmax = L/2; }
  pybind11::array_t<double> bins({nBins});
  auto b = bins.mutable_unchecked<1>();
  for (int bin=0; bin < nBins; bin++) {
    b(bin) = rmin + bin*(rmax - rmin)/nBins;
  }

  // DIFFERENCES
  std::vector<std::vector<double>> diffvijl (nBins, std::vector<double>(0));
  int bin;
  const double dbin = (rmax - rmin)/nBins;
  double dist;
  double diff[2];
  for (int i=0; i < N; i++) {
    for (int j=i + 1; j < N; j++) {
      dist = dist2DPeriod(&(positions[i][0]), &(positions[j][0]), L, &diff[0]);
      if ( dist < rmin || dist >= rmax || dist == 0 ) { continue; }
      bin = (dist - rmin)/dbin;
      diffvijl[bin].push_back(
        (velocities[j][0] - velocities[i][0])*diff[0]/dist
        + (velocities[j][1] - velocities[i][1])*diff[1]/dist);
    }
  }

  // OUT
  std::vector<pybind11::array_t<double>> differences(0);
  double* d;
  for (int bin=0; bin < nBins; bin++) {
    differences.push_back(
      pybind11::array_t<double>({(int) diffvijl[bin].size()}));
    d = (double*) differences[bin].request().ptr;
    for (int p=0; p < (int) diffvijl[bin].size(); p++) {
      d[p] = diffvijl[bin][p];
    }
  }
  return
    std::make_tuple
      <pybind11::array_t<double>, std::vector<pybind11::array_t<double>>>
      (std::move(bins), std::move(differences));
}

std::tuple<std::vector<pybind11::array_t<double>>, pybind11::array_t<double>>
  selfIntScattFunc(
    std::string const& filename,
    pybind11::array_t<int> time0, pybind11::array_t<int> deltat,
    pybind11::array_t<double> qn, double const& dq=0.1, bool remove_cm=true) {
  // Compute self-intermediate scaterring functions for lag times `deltat' from
  // initial times `time0', and with wave vectors `qn' and width of the
  // wave-vector norm interval `dq'.

  DatN dat(filename, false, true);
  const double L = dat.getSystemSize();
  const int N = dat.getNumberParticles();

  auto t0 = time0.unchecked<1>();
  assert ( t0.ndim() == 1 );
  auto dt = deltat.unchecked<1>();
  assert ( dt.ndim() == 1 );

  auto q = qn.unchecked<1>();
  assert ( q.ndim() == 1 );

  // compute the relevant sets of wave-vectors
  int nmax0, nmin, nmax;
  std::vector<std::vector<std::vector<int>>> wn(0);
  for (int iq=0; iq < q.shape(0); iq++) {
    // nmin = std::floor((L/(2*M_PI))*((q(iq) - dq/2)/sqrt(2)));
    nmax0 = std::ceil((L/(2*M_PI))*(q(iq) + dq/2));
    wn.push_back(std::vector<std::vector<int>>(0));
    for (int nx=1; nx <= nmax0; nx++) { // remove (0, n) so it is not redundant with (n, 0)
      nmin = nx > (L/(2*M_PI))*(q(iq) - dq/2) ? 0 :
        std::floor(sqrt(pow((L/(2*M_PI))*(q(iq) - dq/2), 2.) - nx*nx));
      nmax = nx > (L/(2*M_PI))*(q(iq) + dq/2) ? 0 :
        std::floor(sqrt(pow((L/(2*M_PI))*(q(iq) + dq/2), 2.) - nx*nx));
      for (int ny=nmin; ny <= nmax; ny++) {
        if ( abs(q(iq) - (2*M_PI/L)*sqrt(nx*nx + ny*ny)) <= dq/2 ) {
          wn[iq].push_back({nx, ny});
          wn[iq].push_back({-ny, nx});
        }
      }
    }
  }
  std::vector<int> nzeroindices(0);
  for (int iq=0; iq < q.shape(0); iq++) {
    if ( wn[iq].size() > 0 ) {
      nzeroindices.push_back(iq);
    }
  }
  std::vector<pybind11::array_t<double>> wv(0);
  double* wvptr;
  for (int in=0; in < (int) nzeroindices.size(); in++) {
    wv.push_back(
      pybind11::array_t<double>({(int) wn[nzeroindices[in]].size(), 2}));
    wvptr = (double*) wv[in].request().ptr;
    for (int iw=0; iw < wv[in].request().shape[0]; iw++) {
      for (int dim=0; dim < 2; dim++) {
        wvptr[2*iw + dim] = (2*M_PI/L)*wn[nzeroindices[in]][iw][dim];
      }
    }
  }

  // compute self-intermediate scaterring function
  pybind11::array_t<double> Fs(
    {(int) nzeroindices.size(), (int) dt.shape(0), 3});
  auto fsarray = Fs.mutable_unchecked<3>();
  std::vector<std::vector<double>>
    fsvec(t0.shape(0), std::vector<double>((int) nzeroindices.size(), 0));
  std::vector<std::vector<double>>
    fs2vec(t0.shape(0), std::vector<double>((int) nzeroindices.size(), 0));
  std::vector<std::vector<double>> disp(N, std::vector<double>(2, 0));
  double mean_disp[2];
  for (int it=0; it < dt.shape(0); it++) {
    for (int in=0; in < (int) nzeroindices.size(); in++) {
      fsarray(in, it, 0) = dt(it)*dat.getTimeStep();
      fsarray(in, it, 1) = 0;
      fsarray(in, it, 2) = 0;
    }
    for (int it0=0; it0 < t0.shape(0); it0++) {
      // displacements
      for (int dim=0; dim < 2; dim++) {
        mean_disp[dim] = 0;
        for (int i=0; i < N; i++) {
          disp[i][dim] =
            dat.getPosition(t0(it0) + dt(it), i, dim, true)
            - dat.getPosition(t0(it0), i, dim, true);
          mean_disp[dim] += disp[i][dim]/N;
        }
        if ( remove_cm ) {
          for (int i=0; i < N; i++) {
            disp[i][dim] -= mean_disp[dim];
          }
        }
      }
      for (int in=0; in < (int) nzeroindices.size(); in++) {
        fsvec[it0][in] = 0;
        fs2vec[it0][in] = 0;
        wvptr = (double*) wv[in].request().ptr;
        // self-intermediate scattering function
        for (int iw=0; iw < wv[in].request().shape[0]; iw++) {
          for (int i=0; i < N; i++) {
            fsvec[it0][in] +=
              cos(wvptr[2*iw + 0]*disp[i][0] + wvptr[2*iw + 1]*disp[i][1]);
          }
        }
        fsvec[it0][in] /= N*wv[in].request().shape[0];
        fs2vec[it0][in] = fsvec[it0][in]*fsvec[it0][in];
      }
    }
    for (int in=0; in < (int) nzeroindices.size(); in++) {
      for (int it0=0; it0 < t0.shape(0); it0++) {
        fsarray(in, it, 1) +=
          fsvec[it0][in]/t0.shape(0);
        fsarray(in, it, 2) +=
          fs2vec[it0][in]/t0.shape(0)/t0.shape(0);
      }
      fsarray(in, it, 2) =
        sqrt(
          fsarray(in, it, 2)
          - fsarray(in, it, 1)*fsarray(in, it, 1)/t0.shape(0));
    }
  }

  return
    std::make_tuple
      <std::vector<pybind11::array_t<double>>, pybind11::array_t<double>>
      (std::move(wv), std::move(Fs));
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

std::vector<pybind11::array_t<double>> getBondsBrokenBonds(
  pybind11::array_t<double> const& positions,
  pybind11::array_t<double> const& displacements,
  pybind11::array_t<double> const& diameters,
  double const& L, double const& A1=1.15, double const& A2=1.25) {
  // Compute array of initial number of particles with `positions' in a box of
  // size `L' at a distance lesser than `A1' in pair average `diameters' unit
  // for each particle, and array of number of initially bonded particles which
  // are at a distance greater than `A2' after `displacements'.

  pybind11::buffer_info pos_buf = positions.request();
  pybind11::buffer_info dis_buf = displacements.request();
  pybind11::buffer_info dia_buf = diameters.request();
  const int N = pos_buf.shape[0];
  if (dis_buf.shape[0] != N || dia_buf.shape[0] != N) {
    throw std::runtime_error("Input shapes must match.");
  }
  std::vector<double*> r0(N, 0); // initial positions
  std::vector<std::vector<double>> r1(N, std::vector<double>(2, 0)); // final positions
  for (int i=0; i < N; i++) {
    r0[i] = &(((double*) pos_buf.ptr)[2*i]);
    for (int dim=0; dim < 2; dim++) {
      r1[i][dim] = r0[i][dim] + ((double*) dis_buf.ptr)[2*i + dim];
    }
  }
  std::vector<double> sigma((double*) dia_buf.ptr, (double*) dia_buf.ptr + N); // diameters
  double dist, sigmaij;
  double diff[2];

  pybind11::array_t<double> nBonds({N}); // array of initial number of bonds
  double* b = (double*) nBonds.request().ptr;
  pybind11::array_t<double> nBrokenBonds({N}); // array of number of broken bonds
  double *bb = (double*) nBrokenBonds.request().ptr;
  for (int i=0; i < N; i++) {
    // initialise to 0
    b[i] = 0;
    bb[i] = 0;
  }

  CellList cellList(N, L,
    A1*(*std::max_element(sigma.begin(), sigma.end())));
  cellList.listConstructor<double*>(r0);
  cellList.pairs(
    [&b, &bb, &sigma, &r0, &r1, &L, &A1, &A2, &dist, &sigmaij, &diff]
    (int const& index1, int const& index2) { // do for each individual pair
      // rescaled diameter
      sigmaij = (sigma[index1] + sigma[index2])/2;
      // distance
      dist = dist2DPeriod(r0[index1], r0[index2], L, &diff[0]);
      // initial bond
      if ( dist/sigmaij < A1 ) { // particles are bonded
        b[index1] += 1;
        b[index2] += 1;
        // final bond
        dist = dist2DPeriod(&(r1[index1][0]), &(r1[index2][0]), L, &diff[0]);
        if ( dist/sigmaij > A2 ) { // particles are unbonded
          bb[index1] += 1;
          bb[index2] += 1;
        }
      }
    });

  return std::vector<pybind11::array_t<double>>({nBonds, nBrokenBonds});
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
      if ( dist < vmin || dist >= vmax ) { continue; }
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

// POTENTIAL AND FORCES

double getWCA(
  pybind11::array_t<double> const& positions,
  pybind11::array_t<double> const& diameters,
  double const& L) {
  // Compute WCA potential between particles at `positions', with `diameters',
  // in a periodic square box of linear size `L'.

  pybind11::buffer_info pos_buf = positions.request();
  pybind11::buffer_info dia_buf = diameters.request();
  const int N = pos_buf.shape[0];
  if (dia_buf.shape[0] != N) {
    throw std::runtime_error("Input shapes must match.");
  }
  std::vector<double*> r(N, 0);
  for (int i=0; i < N; i++) r[i] = &(((double*) pos_buf.ptr)[2*i]);
  std::vector<double> sigma((double*) dia_buf.ptr, (double*) dia_buf.ptr + N);
  double dist, sigmaij;

  double U = 0;

  CellList cellList(N, L,
    pow(2.0, 1./6.)*(*std::max_element(sigma.begin(), sigma.end())));
  cellList.listConstructor<double*>(r);
  cellList.pairs(
    [&L, &r, &sigma, &sigmaij, &dist, &U](int const& i, int const& j) {
      dist = getDistance(r[i][0], r[i][1], r[j][0], r[j][1], L);
      sigmaij = (sigma[i] + sigma[j])/2;
      if (dist/sigmaij < pow(2., 1./6.)) { // distance lower than cut-off
        U += 4*(pow(sigmaij/dist, 12) - pow(sigmaij/dist, 6) + 1./4.);
      }
    });

  return U;
}

pybind11::array_t<double> getWCAForces(
  pybind11::array_t<double> const& positions,
  pybind11::array_t<double> const& diameters,
  double const& L) {
  // Compute forces from Weeks-Chandler-Andersen potential, between particles at
  // `positions', with `diameters', in a periodic square box of linear size `L'.

  pybind11::buffer_info pos_buf = positions.request();
  pybind11::buffer_info dia_buf = diameters.request();
  const int N = pos_buf.shape[0];
  if (dia_buf.shape[0] != N) {
    throw std::runtime_error("Input shapes must match.");
  }
  std::vector<double*> r(N, 0);
  for (int i=0; i < N; i++) r[i] = &(((double*) pos_buf.ptr)[2*i]); // positions
  std::vector<double> sigma((double*) dia_buf.ptr, (double*) dia_buf.ptr + N); // diameters
  double dist, sigmaij;
  double diff[2];

  pybind11::array_t<double> forces = pybind11::array_t<double>({N, 2}); // array of forces
  double* f = (double*) forces.request().ptr;
  for (int i=0; i < N; i++) {
    for (int dim=0; dim < 2; dim++) {
      f[2*i + dim] = 0; // initialise to 0
    }
  }

  CellList cellList(N, L,
    pow(2.0, 1./6.)*(*std::max_element(sigma.begin(), sigma.end())));
  cellList.listConstructor<double*>(r);
  cellList.pairs(
    [&f, &sigma, &r, &L, &dist, &sigmaij, &diff]
    (int const& index1, int const& index2) { // do for each individual pair
      // rescaled diameter
      sigmaij = (sigma[index1] + sigma[index2])/2;
      // distance
      dist = dist2DPeriod(r[index1], r[index2], L, &diff[0]);
      // potential
      if ( dist/sigmaij < pow(2.0, 1./6.) ) {
        // rescaled distances
        double r6inv = 1./pow((dist/sigmaij), 6);
        // gradient of potential
        for (int dim=0; dim < 2; dim++) {
          f[2*index1 + dim] -=
            (diff[dim]/(dist*dist))*(48*r6inv*r6inv - 24*r6inv);
          f[2*index2 + dim] -=
            -(diff[dim]/(dist*dist))*(48*r6inv*r6inv - 24*r6inv);
        }
      }
    });

  return forces;
}

// pybind11::array_t<double> getWCAtForces(
//   pybind11::array_t<double> const& positions,
//   pybind11::array_t<double> const& velocities,
//   pybind11::array_t<double> const& diameters,
//   double const& L) {
//   // Compute forces from Weeks-Chandler-Andersen potential, between particles at
//   // `positions', with `diameters', in a periodic square box of linear size `L'.
//
//   pybind11::buffer_info pos_buf = positions.request();
//   pybind11::buffer_info vel_buf = velocities.request();
//   pybind11::buffer_info dia_buf = diameters.request();
//   const int N = pos_buf.shape[0];
//   if (vel_buf.shape[0] != N && dia_buf.shape[0] != N) {
//     throw std::runtime_error("Input shapes must match.");
//   }
//   std::vector<double*> r(N, 0);
//   for (int i=0; i < N; i++) r[i] = &(((double*) pos_buf.ptr)[2*i]); // positions
//   std::vector<double> sigma((double*) dia_buf.ptr, (double*) dia_buf.ptr + N); // diameters
//   auto v = velocities.unchecked<2>(); // velocities
//   double dist, sigmaij;
//   double diff[2];
//
//   pybind11::array_t<double> tforces = pybind11::array_t<double>({N, 2}); // array of forces
//   double* tf = (double*) tforces.request().ptr;
//   for (int i=0; i < N; i++) {
//     for (int dim=0; dim < 2; dim++) {
//       tf[2*i + dim] = 0; // initialise to 0
//     }
//   }
//
//   CellList cellList(N, L,
//     pow(2.0, 1./6.)*(*std::max_element(sigma.begin(), sigma.end())));
//   cellList.listConstructor<double*>(r);
//   cellList.pairs(
//     [&tf, &v, &sigma, &r, &L, &dist, &sigmaij, &diff]
//     (int const& index1, int const& index2) { // do for each individual pair
//       // rescaled diameter
//       sigmaij = (sigma[index1] + sigma[index2])/2;
//       // distance
//       dist = dist2DPeriod(r[index1], r[index2], L, &diff[0])/sigmaij;
//       // derivative of potential
//       if ( dist < pow(2.0, 1./6.) ) {
//         // rescaled distances
//         double rinv = 1./(dist/sigmaij);
//         // derivatives of potential
//         double uijp = -48*pow(rinv, 13) + 24*pow(rinv, 7);
//         double uijpp = 624*pow(rinv, 14); - 168*pow(rinv, 8);
//         std::vector<double> d2U(4, 0);
//         double ht;
//         for (int alpha=0; alpha < 2; alpha++) {
//           for (int beta=0; beta < 2; beta++) {
//             ht =
//               (alpha == beta ? uijp/(sigmaij*sigmaij*dist) : 0)
//               + (
//                   (
//                     algDistPeriod(r[index2][alpha], r[index1][alpha], L)
//                     *algDistPeriod(r[index2][beta], r[index1][beta], L))
//                   /(sigmaij*sigmaij*sigmaij*sigmaij))
//               *(uijpp/(dist*dist) - uijp/(dist*dist*dist));
//             h(2*index1 + alpha, 2*index2 + beta) -= ht;
//             h(2*index2 + beta, 2*index1 + alpha) -= ht;
//             h(2*index1 + alpha, 2*index1 + beta) += ht;
//             h(2*index2 + alpha, 2*index2 + beta) += ht;
//           }
//         }
//         // gradient of potential
//         for (int dim=0; dim < 2; dim++) {
//           f[2*index1 + dim] -=
//             (diff[dim]/(dist*dist))*(48*r6inv*r6inv - 24*r6inv);
//           f[2*index2 + dim] -=
//             -(diff[dim]/(dist*dist))*(48*r6inv*r6inv - 24*r6inv);
//         }
//       }
//     });
//
//   return forces;
// }

pybind11::array_t<double> getRAForces(
  pybind11::array_t<double> const& positions,
  pybind11::array_t<double> const& diameters,
  double const& L, double const& a=12, double const& rcut=1.25) {
  // Compute forces from regularised 1/r^`a' potential, with cut-off radius
  // `rcut', between particles at `positions', with `diameters', in a periodic
  // square box of linear size `L'.

  // double const c0 = -(8 + a*(a + 6))/(8*pow(rcut, a)); // constant part of potential
  double const c1 = (a*(a + 4))/(4*pow(rcut, a + 2)); // quadratic part of potential
  double const c2 = -(a*(a + 2))/(8*pow(rcut, a + 4)); // quartic part of potential

  pybind11::buffer_info pos_buf = positions.request();
  pybind11::buffer_info dia_buf = diameters.request();
  const int N = pos_buf.shape[0];
  if (dia_buf.shape[0] != N) {
    throw std::runtime_error("Input shapes must match.");
  }
  std::vector<double*> r(N, 0);
  for (int i=0; i < N; i++) r[i] = &(((double*) pos_buf.ptr)[2*i]); // positions
  std::vector<double> sigma((double*) dia_buf.ptr, (double*) dia_buf.ptr + N); // diameters
  double dist, sigmaij;
  double diff[2];

  pybind11::array_t<double> forces = pybind11::array_t<double>({N, 2}); // array of forces
  double* f = (double*) forces.request().ptr;
  for (int i=0; i < N; i++) {
    for (int dim=0; dim < 2; dim++) {
      f[2*i + dim] = 0; // initialise to 0
    }
  }

  CellList cellList(N, L,
    rcut*(*std::max_element(sigma.begin(), sigma.end())));
  cellList.listConstructor<double*>(r);
  cellList.pairs(
    [&f, &sigma, &r, &L, &a, &c1, &c2, &rcut, &dist, &sigmaij, &diff]
    (int const& index1, int const& index2) { // do for each individual pair
      // rescaled diameter
      sigmaij = (sigma[index1] + sigma[index2])/2
        *(1 - 0.2*fabs(sigma[index1] - sigma[index2]));
      // distance
      dist = dist2DPeriod(r[index1], r[index2], L, &diff[0]);
      // potential
      if ( dist/sigmaij < rcut ) {
        // rescaled distances
        double rAinv = 1./pow((dist/sigmaij), a);
        double r2 = pow((dist/sigmaij), 2);
        double r4 = r2*r2;
        // gradient of potential
        for (int dim=0; dim < 2; dim++) {
          f[2*index1 + dim] -=
            (diff[dim]/(dist*dist))*(a*rAinv - 2*c1*r2 - 4*c2*r4);
          f[2*index2 + dim] -=
            -(diff[dim]/(dist*dist))*(a*rAinv - 2*c1*r2 - 4*c2*r4);
        }
      }
    });

  return forces;
}

pybind11::array_t<double> getRAHessian(
  pybind11::array_t<double> const& positions,
  pybind11::array_t<double> const& diameters,
  double const& L, double const& a=12, double const& rcut=1.25) {
  // Compute regularised 1/r^`a'-potential Hessian matrix, with cut-off radius
  // `rcut', between particles at `positions', with `diameters', in a periodic
  // square box of linear size `L'.

  // double const c0 = -(8 + a*(a + 6))/(8*pow(rcut, a)); // constant part of potential
  double const c1 = (a*(a + 4))/(4*pow(rcut, a + 2)); // quadratic part of potential
  double const c2 = -(a*(a + 2))/(8*pow(rcut, a + 4)); // quartic part of potential

  pybind11::buffer_info pos_buf = positions.request();
  pybind11::buffer_info dia_buf = diameters.request();
  const int N = pos_buf.shape[0];
  if (dia_buf.shape[0] != N) {
    throw std::runtime_error("Input shapes must match.");
  }
  std::vector<double*> r(N, 0);
  for (int i=0; i < N; i++) r[i] = &(((double*) pos_buf.ptr)[2*i]); // positions
  std::vector<double> sigma((double*) dia_buf.ptr, (double*) dia_buf.ptr + N); // diameters
  double dist, sigmaij;
  double diff[2];

  pybind11::array_t<double> hessian = pybind11::array_t<double>({2*N, 2*N}); // hessian
  auto h = hessian.mutable_unchecked<2>();
  for (int i=0; i < 2*N; i++) {
    for (int j=0; j < 2*N; j++) {
      h(i, j) = 0; // initialises to 0
    }
  }

  try {

    // CELL LIST
    CellList cellList(N, L,
      rcut*(*std::max_element(sigma.begin(), sigma.end())));
    cellList.listConstructor<double*>(r);
    cellList.pairs(
      [&h, &sigma, &r, &L, &a, &c1, &c2, &rcut, &dist, &sigmaij, &diff]
      (int const& index1, int const& index2) { // do for each individual pair
        // rescaled diameter
        sigmaij = (sigma[index1] + sigma[index2])/2
          *(1 - 0.2*fabs(sigma[index1] - sigma[index2]));
        // rescaled distance
        dist = dist2DPeriod(r[index1], r[index2], L, &diff[0])/sigmaij;
        // potential
        if ( dist < rcut ) {
          double uijp = -a/pow(dist, a + 1) + 2*c1*dist + 4*c2*dist*dist*dist;
          double uijpp = a*(a + 1)/pow(dist, a + 2) + 2*c1 + 12*c2*dist*dist;
          double ht;
          for (int alpha=0; alpha < 2; alpha++) {
            for (int beta=0; beta < 2; beta++) {
              ht =
                (alpha == beta ? uijp/(sigmaij*sigmaij*dist) : 0)
                + (
                    (
                      algDistPeriod(r[index2][alpha], r[index1][alpha], L)
                      *algDistPeriod(r[index2][beta], r[index1][beta], L))
                    /(sigmaij*sigmaij*sigmaij*sigmaij))
                *(uijpp/(dist*dist) - uijp/(dist*dist*dist));
              h(2*index1 + alpha, 2*index2 + beta) -= ht;
              h(2*index2 + beta, 2*index1 + alpha) -= ht;
              h(2*index1 + alpha, 2*index1 + beta) += ht;
              h(2*index2 + alpha, 2*index2 + beta) += ht;
            }
          }
        }
      });

  }
  catch (std::runtime_error const&) {

    std::cerr << "Using double loop." << std::endl;

    // DOUBLE LOOP
    for (int i=0; i < N; i++) {
      for (int j=0; j < N; j++) {
        if ( i == j ) {
          // DIAGONAL TERM
          for (int k=0; k < N; k++) {
            if ( k == i ) { continue; }
            // rescaled diameter
            sigmaij = (sigma[i] + sigma[k])/2
              *(1 - 0.2*fabs(sigma[i] - sigma[k]));
            // rescaled distance
            dist = dist2DPeriod(r[i], r[k], L, &diff[0])/sigmaij;
            // potential
            if ( dist < rcut ) {
              double uijp =
                -a/pow(dist, a + 1) + 2*c1*dist + 4*c2*dist*dist*dist;
              double uijpp =
                a*(a + 1)/pow(dist, a + 2) + 2*c1 + 12*c2*dist*dist;
              for (int alpha=0; alpha < 2; alpha++) {
                for (int beta=0; beta < 2; beta++) {
                  h(2*i + alpha, 2*i + beta) +=
                    (alpha == beta ? uijp/(sigmaij*sigmaij*dist) : 0)
                    + (
                        (
                          algDistPeriod(r[k][alpha], r[i][alpha], L)
                          *algDistPeriod(r[k][beta], r[i][beta], L))
                        /(sigmaij*sigmaij*sigmaij*sigmaij))
                    *(uijpp/(dist*dist) - uijp/(dist*dist*dist));
                }
              }
            }
          }
        }
        else {
          // rescaled diameter
          sigmaij = (sigma[i] + sigma[j])/2
            *(1 - 0.2*fabs(sigma[i] - sigma[j]));
          // rescaled distance
          dist = dist2DPeriod(r[i], r[j], L, &diff[0])/sigmaij;
          // potential
          if ( dist < rcut ) {
            double uijp =
              -a/pow(dist, a + 1) + 2*c1*dist + 4*c2*dist*dist*dist;
            double uijpp =
              a*(a + 1)/pow(dist, a + 2) + 2*c1 + 12*c2*dist*dist;
            for (int alpha=0; alpha < 2; alpha++) {
              for (int beta=0; beta < 2; beta++) {
                h(2*i + alpha, 2*j + beta) -=
                  (alpha == beta ? uijp/(sigmaij*sigmaij*dist) : 0)
                  + (
                      (
                        algDistPeriod(r[j][alpha], r[i][alpha], L)
                        *algDistPeriod(r[j][beta], r[i][beta], L))
                      /(sigmaij*sigmaij*sigmaij*sigmaij))
                  *(uijpp/(dist*dist) - uijp/(dist*dist*dist));
              }
            }
          }
        }
      }
    }

  }

  return hessian;
}

// std::vector<pybind11::array_t<double>> phononOrderParameter(
//   pybind11::array_t<double> positions, double L,
//   pybind11::array_t<double> eigenvectors) {
//   // Compute phonon order parameter for eigenvectors `ev'.
//
//   auto r = positions.unchecked<2>();
//   auto e = eigenvectors.unchecked<3>();
//   if ( r.shape(0) != e.shape(1) || r.shape(1) != 2 || e.shape(2) != 2 ) {
//     throw std::invalid_argument("Arrays' sizes are not consistent.");
//   }
//   int N = r.shape(0);
//
//   pybind11::ok<double> ok({e.shape(0)});
//   auto o = correlations.mutable_unchecked<1>();
//
//   // PHONONS
//
//   double q[2], norm;
//   std::vector<double> amp(N, 0);
//   std::vector<std::vector<double>> phonons(0);
//   for (int k=1; k <= (int) sqrt(N); k++) {
//     for (int l=1; l <= (int) sqrt(N); l++) {
//       if ( k*k + l*l <= N ) {
//         // wave-vector
//         q[0] = 2*M_PI/L*k;
//         q[1] = 2*M_PI/L*l;
//         norm = sqrt(q[0]*q[0] + q[1]*q[1]);
//         // amplitude
//       }
//     }
//   }
//
// }

// VELOCITIES

std::vector<pybind11::array_t<double>> getVelocityVorticity(
  pybind11::array_t<double> const& positions,
  pybind11::array_t<double> const& velocities,
  double const& L, int const& nBoxes, double const& sigma,
  pybind11::array_t<double> const& centre) {
  // Compute Gaussian-smoothed velocitiy field and vorticity field, using
  // standard deviation `sigma', on a (`nBoxes', `nBoxes')-grid, from
  // `positions' and `velocities', in a system of size `L'.

  double xmin = 0;
  double ymin = 0;
  auto c = centre.unchecked<1>();
  if ( c.shape(0) == 2 ) {
    xmin = c(0) - L/2.;
    ymin = c(1) - L/2.;
  }

  pybind11::buffer_info pos_buf = positions.request();
  pybind11::buffer_info vel_buf = velocities.request();

  if (pos_buf.size != vel_buf.size) {
    throw std::runtime_error("Input shapes must match.");
  }
  const int N = pos_buf.shape[0];

  double* const input_pos = (double*) pos_buf.ptr;
  double* const input_vel = (double*) vel_buf.ptr;

  pybind11::array_t<double> pos_grid = // grid of positions
    pybind11::array_t<double>({nBoxes, nBoxes, 2});
  double* pos = (double*) pos_grid.request().ptr;
  pybind11::array_t<double> vel_grid = // grid of Gaussian-smoothed velocities
    pybind11::array_t<double>({nBoxes, nBoxes, 2});
  double* vel = (double*) vel_grid.request().ptr;
  pybind11::array_t<double> vor_grid = // grid of voriticities from Gaussian-smoothed velocities
    pybind11::array_t<double>({nBoxes, nBoxes});
  double* vor = (double*) vor_grid.request().ptr;
  pybind11::array_t<double> vel_p = // vector of Gaussian-smoothed velocities
    pybind11::array_t<double>({N, 2});
  double* veli = (double*) vel_p.request().ptr;
  pybind11::array_t<double> vor_p = // vecotr of Gaussian-smoothed vorticities
    pybind11::array_t<double>({N});
  double* vori = (double*) vor_p.request().ptr;
  // set positions and set grids to 0
  for (int cx=0; cx < nBoxes; cx++) {
    for (int cy=0; cy < nBoxes; cy++) {
      pos[cx*nBoxes*2 + cy*2 + 0] = L/nBoxes*(cx + 0.5);
      pos[cx*nBoxes*2 + cy*2 + 1] = L/nBoxes*(cy + 0.5);
      for (int dim=0; dim < 2; dim++) {
        vel[cx*nBoxes*2 + cy*2 + dim] = 0;
      }
      vor[cx*nBoxes + cy] = 0;
    }
    for (int i=0; i < N; i++) {
      veli[2*i + 0] = 0;
      veli[2*i + 1] = 0;
      vori[i] = 0;
    }
  }

  // compute
  const double sigma2 = sigma*sigma;
  const double norm = 1./(2*M_PI*sigma2); // normalisation factor
  double dx, dy, gaussian, term;
  for (int i=0; i < N; i++) {
    for (int cx=0; cx < nBoxes; cx++) {
      for (int cy=0; cy < nBoxes; cy++) {
        dx = algDistPeriod( // x - x_i
          std::remainder(input_pos[2*i + 0], L), // x-coordinate of particle i
          std::remainder(xmin + pos[cx*nBoxes*2 + cy*2 + 0], L), // x-coordinate of grid point (cx, cy)
          L);
        dy = algDistPeriod( // y - y_i
          std::remainder(input_pos[2*i + 1], L), // y-coordinate of particle i
          std::remainder(ymin + pos[cx*nBoxes*2 + cy*2 + 1], L), // y-coordinate of grid point (cx, cy)
          L);
        gaussian = exp(-(dx*dx + dy*dy))*norm; // Gaussian factor
        for (int dim=0; dim < 2; dim++) {
          term = gaussian*input_vel[2*i + dim];
          vel[cx*nBoxes*2 + cy*2 + dim] += term;
          vor[cx*nBoxes + cy] += term*(dim == 0 ? dy : -dx)/sigma2;
        }
      }
    }
    for (int j=0; j < N; j++) {
      dx = algDistPeriod( // x - x_i
        std::remainder(input_pos[2*i + 0], L), // x-coordinate of particle i
        std::remainder(input_pos[2*j + 0], L), // x-coordinate of particle j
        L);
      dy = algDistPeriod( // y - y_i
        std::remainder(input_pos[2*i + 1], L), // y-coordinate of particle i
        std::remainder(input_pos[2*j + 1], L), // y-coordinate of particle j
        L);
      gaussian = exp(-(dx*dx + dy*dy))*norm; // Gaussian factor
      for (int dim=0; dim < 2; dim++) {
        term = gaussian*input_vel[2*i + dim];
        veli[2*i + dim] += term;
        vori[i] += term*(dim == 0 ? dy : -dx)/sigma2;
      }
    }
  }

  // return
  std::vector<pybind11::array_t<double>> result;
  result.push_back(pos_grid);
  result.push_back(vel_grid);
  result.push_back(vor_grid);
  result.push_back(vel_p);
  result.push_back(vor_p);
  return result;
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

pybind11::array_t<int> getPolarCharge(pybind11::array_t<double> grid) {
  // Return grid of polar charge.

  auto g = grid.unchecked<3>();
  if ( g.ndim() != 3 || g.shape(2) != 2 ) {
    throw std::invalid_argument("Grid must be (*, **, 2).");
  }

  pybind11::array_t<int> charge({g.shape(0), g.shape(1)});
  auto c = charge.mutable_unchecked<2>();

  double neighbours[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  double sum;
  double angles[4] = {0, 0, 0, 0};
  int in, jn, quot;
  for (int i=0; i < g.shape(0); i++) {
    for (int j=0; j < g.shape(1); j++) {
      for (int n=0; n < 4; n++) {
        in = i + neighbours[n][0];
        if ( in == -1 ) { in = g.shape(0) - 1; }
        if ( in == g.shape(0) ) { in = 0; }
        jn = j + neighbours[n][1];
        if ( jn == -1 ) { jn = g.shape(1) - 1; }
        if ( jn == g.shape(1) ) { jn = 0; }
        angles[n] = atan2(g(in, jn, 1), g(in, jn, 0));
      }
      sum =
        std::remquo(angles[1] - angles[0], 2*M_PI, &quot) // 1 -> 2
        + std::remquo(angles[2] - angles[1], 2*M_PI, &quot) // 2 -> 3
        + std::remquo(angles[3] - angles[2], 2*M_PI, &quot) // 3 -> 4
        + std::remquo(angles[0] - angles[3], 2*M_PI, &quot); // 4 -> 1
      c(i, j) = (int) std::round(sum/2/M_PI);
    }
  }

  return charge;
}

// CORRELATIONS

pybind11::array_t<std::complex<double>> getRadialCorrelations(
  pybind11::array_t<double> positions, double L,
  pybind11::array_t<std::complex<double>> values1,
  pybind11::array_t<std::complex<double>> values2,
  int nBins, double rmin=0, double rmax=0,
  bool rescale_pair_distribution=false) {

  if ( rmax == 0 ) { rmax = L/2; }

  pybind11::array_t<std::complex<double>> correlations({nBins, 2});
  auto c = correlations.mutable_unchecked<2>();
  for (int bin=0; bin < c.shape(0); bin++) {
    c(bin, 0) = rmin + bin*(rmax - rmin)/nBins;
    c(bin, 1) = 0;
  }
  std::vector<int> occupancy(nBins, 0);
  int nPairs = 0; // number of pairs

  auto r = positions.unchecked<2>();
  auto v1 = values1.unchecked<>();
  auto v2 = values2.unchecked<>();
  if ( v1.shape(0) != r.shape(0) ||
    v1.shape(0) != v2.shape(0) || v1.size() != v2.size() ) {
    throw std::invalid_argument("Arrays' sizes are not consistent.");
  }
  if ( v1.ndim() > 2 || v2.ndim() > 2 ) {
    throw std::invalid_argument("Arrays must be 1- or 2-dimensional.");
  }

  int bin;
  double dbin = (rmax - rmin)/nBins;
  double dist;
  for (int i=0; i < r.shape(0); i++) {
    for (int j=i; j < r.shape(0); j++) {
      if ( i != j ) { nPairs++; }
      dist = getDistance(r(i, 0), r(i, 1), r(j, 0), r(j, 1), L);
      if ( dist < rmin || dist >= rmax ) { continue; }
      bin = (dist - rmin)/dbin;
      if ( v1.ndim() == 1 ) {
        c(bin, 1) +=
          (v1(i)*std::conj(v2(j)) + v1(j)*std::conj(v2(i)))/2.;
      }
      else {
        for (int d=0; d < v1.shape(1); d++) {
          c(bin, 1) +=
            (v1(i, d)*std::conj(v2(j, d)) + v1(j, d)*std::conj(v2(i, d)))/2.;
        }
      }
      occupancy[bin] += 1;
    }
  }

  for (int bin=0; bin < nBins; bin++) {
    if ( occupancy[bin] > 0 ) {
      // mean over computed values
      c(bin, 1) /= occupancy[bin];
      // correction by pair distribution function
      if ( ! rescale_pair_distribution ) { continue; }
      if ( bin == 0 && rmin == 0 ) { continue; } // do not consider 0th bin
      c(bin, 1) /=
        ((double) occupancy[bin]/nPairs) // histogram value
        *pow(L, 2)/((rmax - rmin)/nBins) // normalisation
        /(2*M_PI*(rmin + bin*(rmax - rmin)/nBins)); // radial projection
    }
  }

  return correlations;
}

std::vector<pybind11::array_t<std::complex<double>>>
  getRadialDirectonCorrelations(
  pybind11::array_t<double> positions, double L,
  pybind11::array_t<std::complex<double>> values1,
  pybind11::array_t<std::complex<double>> values2,
  int nBins, double rmin=0, double rmax=0,
  bool rescale_pair_distribution=false) {

  if ( rmax == 0 ) { rmax = L/2; }

  std::vector<pybind11::array_t<std::complex<double>>> correlations(0);
  correlations.push_back(pybind11::array_t<std::complex<double>>({nBins, 2}));
  correlations.push_back(pybind11::array_t<std::complex<double>>({nBins, 2}));
  auto clong = correlations[0].mutable_unchecked<2>();
  auto cperp = correlations[1].mutable_unchecked<2>();
  for (int bin=0; bin < nBins; bin++) {
    clong(bin, 0) = rmin + bin*(rmax - rmin)/nBins;
    clong(bin, 1) = 0;
    cperp(bin, 0) = rmin + bin*(rmax - rmin)/nBins;
    cperp(bin, 1) = 0;
  }
  std::vector<int> occupancy(nBins, 0);
  int nPairs = 0; // number of pairs

  auto r = positions.unchecked<2>();
  double* rPTR = (double*) positions.request().ptr;
  auto v1 = values1.unchecked<>();
  auto v2 = values2.unchecked<>();
  if ( v1.shape(0) != r.shape(0) ||
    v1.shape(0) != v2.shape(0) || v1.size() != v2.size() ) {
    throw std::invalid_argument("Arrays' sizes are not consistent.");
  }
  if ( v1.ndim() != 2 || v2.ndim() != 2 ) {
    throw std::invalid_argument("Arrays must 2-dimensional.");
  }
  if ( v1.shape(1) != 2 || v2.shape(1) != 2 ) {
    throw std::invalid_argument("Arrays' shape must be (*, 2).");
  }

  int bin;
  double dbin = (rmax - rmin)/nBins;
  double dist;
  double diff[2];
  std::complex<double> v1ilong, v1iperp, v2ilong, v2iperp;
  std::complex<double> v1jlong, v1jperp, v2jlong, v2jperp;
  for (int i=0; i < r.shape(0); i++) {
    for (int j=i; j < r.shape(0); j++) {
      if ( i != j ) { nPairs++; }
      dist = dist2DPeriod(&rPTR[2*i], &rPTR[2*j], L, &diff[0]);
      if ( dist < rmin || dist >= rmax || dist == 0 ) { continue; }
      bin = (dist - rmin)/dbin;
      v1ilong = (v1(i, 0)*diff[0] + v1(i, 1)*diff[1])/dist;
      v1iperp = (v1(i, 0)*diff[1] - v1(i, 1)*diff[0])/dist;
      v2ilong = (v2(i, 0)*diff[0] + v2(i, 1)*diff[1])/dist;
      v2iperp = (v2(i, 0)*diff[1] - v2(i, 1)*diff[0])/dist;
      v1jlong = (v1(j, 0)*diff[0] + v1(j, 1)*diff[1])/dist;
      v1jperp = (v1(j, 0)*diff[1] - v1(j, 1)*diff[0])/dist;
      v2jlong = (v2(j, 0)*diff[0] + v2(j, 1)*diff[1])/dist;
      v2jperp = (v2(j, 0)*diff[1] - v2(j, 1)*diff[0])/dist;
      clong(bin, 1) +=
        (v1ilong*std::conj(v2jlong) + v1jlong*std::conj(v2ilong))/2.;
      cperp(bin, 1) +=
        (v1iperp*std::conj(v2jperp) + v1jperp*std::conj(v2iperp))/2.;
      occupancy[bin] += 1;
    }
  }

  for (int bin=0; bin < nBins; bin++) {
    if ( occupancy[bin] > 0 ) {
      // mean over computed values
      clong(bin, 1) /= occupancy[bin];
      cperp(bin, 1) /= occupancy[bin];
      // correction by pair distribution function
      if ( ! rescale_pair_distribution ) { continue; }
      if ( bin == 0 && rmin == 0 ) { continue; } // do not consider 0th bin
      clong(bin, 1) /=
        ((double) occupancy[bin]/nPairs) // histogram value
        *pow(L, 2)/((rmax - rmin)/nBins) // normalisation
        /(2*M_PI*(rmin + bin*(rmax - rmin)/nBins)); // radial projection
      cperp(bin, 1) /=
        ((double) occupancy[bin]/nPairs) // histogram value
        *pow(L, 2)/((rmax - rmin)/nBins) // normalisation
        /(2*M_PI*(rmin + bin*(rmax - rmin)/nBins)); // radial projection
    }
  }

  return correlations;
}

pybind11::array_t<std::complex<double>> getRadialDirectionAmplitudes(
  pybind11::array_t<double> positions, double L,
  pybind11::array_t<std::complex<double>> values) {

  pybind11::array_t<std::complex<double>> amplitudes({2});
  auto a2 = amplitudes.mutable_unchecked<1>();
  for (int dim=0; dim < 2; dim++) {
    a2(dim) = 0;
  }

  auto r = positions.unchecked<2>();
  double N = r.shape(0);
  double* rPTR = (double*) positions.request().ptr;
  auto v = values.unchecked<>();
  if ( v.shape(0) != N ) {
    throw std::invalid_argument("Arrays' sizes are not consistent.");
  }
  if ( v.ndim() != 2 ) {
    throw std::invalid_argument("Arrays must 2-dimensional.");
  }
  if ( v.shape(1) != 2 ) {
    throw std::invalid_argument("Arrays' shape must be (*, 2).");
  }

  double dist;
  double diff[2];
  std::complex<double> vilong, viperp, vjlong, vjperp;
  for (int i=0; i < N; i++) {
    for (int j=i + 1; j < N; j++) {
      dist = dist2DPeriod(&rPTR[2*i], &rPTR[2*j], L, &diff[0]);
      vilong = (v(i, 0)*diff[0] + v(i, 1)*diff[1])/dist;
      viperp = (v(i, 0)*diff[1] - v(i, 1)*diff[0])/dist;
      vjlong = (v(j, 0)*diff[0] + v(j, 1)*diff[1])/dist;
      vjperp = (v(j, 0)*diff[1] - v(j, 1)*diff[0])/dist;
      a2(0) += (vilong*std::conj(vilong) + vjlong*std::conj(vjlong))/(N*N);
      a2(1) += (viperp*std::conj(viperp) + vjperp*std::conj(vjperp))/(N*N);
    }
  }

  return amplitudes;
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

class IntegratePropulsions {

  public:

    // CONSTRUCTORS

    IntegratePropulsions(std::string const& inputFilename) :
      input(inputFilename),
      add(
        input.getNumberParticles(),
        input.getSystemSize(),
        input.getDiameters(),
        input.getPropulsionVelocity(),
        input.getTimeStep(),
        0, 1, 0, 1, 1, 0, 0,
        input.getRandomSeed(),
        ""),
      N(add.getNumberParticles()),
      tmp_file_name_rng(strdup("/tmp/tmpfileXXXXXX")),
      tmp_file_name_norm(strdup("/tmp/tmpfileXXXXXX")),
      rg0((add.getRandomGenerator())->getGenerator()),
      propulsions0([&inputFilename](){
        DatN dat(inputFilename);
        std::vector<std::vector<double>>
          p(dat.getNumberParticles(), std::vector<double> (2, 0));
        for (int i=0; i < dat.getNumberParticles(); i++) {
          for (int dim=0; dim < 2; dim++) {
            p[i][dim] = dat.getPropulsion(0, i, dim);
          }
        }
        return p;
        }()) {

        // save random number generator
        if ( !mkstemp(tmp_file_name_rng) ) {
          throw std::runtime_error("Cannot create temporary file.");
        }
        std::ofstream fout_rng(tmp_file_name_rng);
        fout_rng << (add.getRandomGenerator())->getGenerator();
        fout_rng.close();

        // save normal distribution
        if ( !mkstemp(tmp_file_name_norm) ) {
          throw std::runtime_error("Cannot create temporary file.");
        }
        std::ofstream fout_norm(tmp_file_name_norm);
        fout_norm << *(add.getRandomGenerator())->getNormal();
        fout_norm.close();
      }

    // DESTRUCTORS

    ~IntegratePropulsions() {}

    // METHODS

    // pybind11::array_t<double> getPropulsions(int const& frame) {
    //   // Return propulsions at `frame'.
    //
    //   // INITIALISE
    //   // restore random number generator
    //   std::ifstream fin_rng(tmp_file_name_rng);
    //   fin_rng >> rg0;
    //   fin_rng.close();
    //   (add.getRandomGenerator())->setGenerator(rg0);
    //   // restore distribution
    //   std::ifstream fin_norm(tmp_file_name_norm);
    //   fin_norm >> *(add.getRandomGenerator())->getNormal();
    //   fin_norm.close();
    //   // initialise propulsions
    //   for (int i=0; i < N; i++) { // propulsions
    //     for (int dim=0; dim < 2; dim++) {
    //       add.getPropulsion(i)[dim] = propulsions0[i][dim];
    //     }
    //   }
    //   // INTEGRATE
    //   for (int t=0; t < frame - 1; t++) {
    //     add.iteratePropulsion();
    //   }
    //
    //   // ARRAY
    //   pybind11::array_t<double> propulsions({N, 2});
    //   auto p = propulsions.mutable_unchecked<2>();
    //   for (int i=0; i < N; i++) {
    //     for (int dim=0; dim < 2; dim++) {
    //       p(i, dim) = add.getPropulsion(i)[dim];
    //     }
    //   }
    //
    //   return propulsions;
    // }
    pybind11::array_t<double> getPropulsions(pybind11::array_t<int> frames) {
      // Return propulsions at `frame'.

      if ( (add.getRandomGenerator())->getGenerator() != rg0 ) {
        throw std::runtime_error("Cannot be called more than once.");
      }

      // initialise
      for (int i=0; i < N; i++) { // propulsions
        for (int dim=0; dim < 2; dim++) {
          add.getPropulsion(i)[dim] = propulsions0[i][dim];
        }
      }
      auto f = frames.unchecked<1>();

      pybind11::array_t<double> propulsions({(int) f.shape(0), N, 2});
      auto p = propulsions.mutable_unchecked<3>();
      int index = 0;
      if ( f(index) == 0 ) {
        for (int i=0; i < N; i++) {
          for (int dim=0; dim < 2; dim++) {
            p(index, i, dim) = add.getPropulsion(i)[dim];
          }
        }
        index++;
      }
      for (int t=1; t <= f(f.shape(0) - 1); t++) {
        if ( f(index) == t ) {
          for (int i=0; i < N; i++) {
            for (int dim=0; dim < 2; dim++) {
              p(index, i, dim) = add.getPropulsion(i)[dim];
            }
          }
          index++;
        }
        add.iteratePropulsion();
      }

      return propulsions;
    }

  private:

    // ATTRIBUTES

    DatN input; // input file object
    ADD add; // ADD simulation object
    int const N; // number of particles
    char* tmp_file_name_rng; // temporary file name when random number generator is stored
    char* tmp_file_name_norm; // temporary file name when normal distribution is stored
    RNDG rg0; // initial random generator
    std::vector<std::vector<double>> const propulsions0; // initial propulsions

};

// LINEAR ALGEBRA

pybind11::array_t<double>
  invertMatrix(pybind11::array_t<double> const& matrix) {
  // Invert matrix.

  // PYBIND11 ARRAYS

  const int n = matrix.request().shape[0]; // matrix size
  assert (n == matrix.request().shape[1]); // check matrix is square
  if (matrix.request().shape.size() != 2) {
    throw std::invalid_argument("Not a square matrix.");
  }
  auto m = matrix.unchecked<2>();

  pybind11::array_t<double> invMatrix({n, n}); // inverse matrix
  auto invM = invMatrix.mutable_unchecked<2>();

  // GSL MATRICES

  gsl_set_error_handler_off(); // turns off error handler

  gsl_matrix* gsl_m = gsl_matrix_alloc(n, n);
  gsl_matrix* gsl_invM = gsl_matrix_alloc(n, n);
  for (int i=0; i < n; i++) {
    for (int j=0; j < n; j++) {
      gsl_matrix_set(gsl_m, i, j, m(i, j));
      gsl_matrix_set(gsl_invM, i, j, 0);
    }
  }

  gsl_permutation* p = gsl_permutation_alloc(n);

  int s, status;
  status = gsl_linalg_LU_decomp(gsl_m, p, &s);
  if (status) throw std::runtime_error("Cannot be inverted.");
  status = gsl_linalg_LU_invert(gsl_m, p, gsl_invM);
  if (status) throw std::runtime_error("Cannot be inverted.");

  for (int i=0; i < n; i++) {
    for (int j=0; j < n; j++) {
      invM(i, j) = gsl_matrix_get(gsl_invM, i, j);
    }
  }

  gsl_matrix_free(gsl_m);
  gsl_matrix_free(gsl_invM);
  gsl_permutation_free(p);

  return invMatrix;
}

pybind11::array_t<double>
  invertSparseMatrix(pybind11::array_t<double> const& matrix) {
  // Invert matrix.

  // PYBIND11 ARRAYS

  const int n = matrix.request().shape[0]; // matrix size
  assert (n == matrix.request().shape[1]); // check matrix is square
  if (matrix.request().shape.size() != 2) {
    throw std::invalid_argument("Not a square matrix.");
  }
  auto m = matrix.unchecked<2>();

  pybind11::array_t<double> invMatrix({n, n}); // inverse matrix
  auto invM = invMatrix.mutable_unchecked<2>();

  // EIGEN SPARSE MATRICES

  // filling
  Eigen::SparseMatrix<double> mat(n, n); // sparse matrix
  std::vector<Eigen::Triplet<double>> trip; // triplets to fill the sparse matrix
  trip.reserve(n*n); // reserve memory space
  for (int i=0; i < n; i++) {
    for (int j=0; j < n; j++) {
      if ( m(i, j) != 0 ) {
        trip.push_back(Eigen::Triplet<double>(i, j, m(i, j))); // add non-zero value
      }
    }
  }
  mat.setFromTriplets(trip.begin(), trip.end()); // fill sparse matrix from triplet

  // solving
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(mat);
  if(solver.info() != Eigen::Success) {
    throw std::runtime_error("Decomposition failed.");
  }
  Eigen::SparseMatrix<double> I(n, n);
  I.setIdentity();
  Eigen::SparseMatrix<double> invMat = solver.solve(I);
  for (int k=0; k < invMat.outerSize(); k++) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(invMat, k); it; ++it) {
      invM(it.row(), it.col()) = it.value();
    }
  }

  return invMatrix;
}

/////////////////////
// PYBIND11 EXPORT //
/////////////////////

PYBIND11_MODULE(_pycpp, m) {
  m.doc() = "Module pycpp provides interfaces between python and C++.";

  m.def("getDisplacements", &getDisplacementsPYB,
    "Compute displacements.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "filename : str\n"
    "    Path to data file.\n"
    "time0 : (*,) int array-like\n"
    "    Initial frames.\n"
    "deltat : (**,) int array-like\n"
    "    Lag times.\n"
    "remove_cm : bool\n"
    "    Remove centre of mass displacement. (default: True)\n"
    "\n"
    "Returns\n"
    "-------\n"
    "displacements : (*, **, ***, 2) float numpy array\n"
    "    Displacements.",
    pybind11::arg("filename"),
    pybind11::arg("time0"),
    pybind11::arg("deltat"),
    pybind11::arg("remove_cm")=true);

  m.def("getDisplacementsPair", &getDisplacementsPair,
    "Compute relative displacements between initially bonded particles.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "filename : str\n"
    "    Path to data file.\n"
    "time0 : (*,) int array-like\n"
    "    Initial frames.\n"
    "deltat : (**,) int array-like\n"
    "    Lag times.\n"
    "a1 : float\n"
    "    Distance in pair average diameter unit under which particles are\n"
    "    considered bonded. (default: 1.15)\n"
    "\n"
    "Returns\n"
    "-------\n"
    "displacementsPair : (*,) list of (**, ***, 2) float numpy array\n"
    "    Relative displacements between pairs.\n"
    "pairs : (*,) list of (***, 2) float numpy array\n"
    "    Pairs of neighbouring particles at `time0'.",
    pybind11::arg("filename"),
    pybind11::arg("time0"),
    pybind11::arg("deltat"),
    pybind11::arg("a1")=1.15);

  m.def("getSeparationsPair", &getSeparationsPair,
    "Compute separations between initially bonded particles.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "filename : str\n"
    "    Path to data file.\n"
    "time0 : (*,) int array-like\n"
    "    Initial frames.\n"
    "deltat : (**,) int array-like\n"
    "    Lag times.\n"
    "a1 : float\n"
    "    Distance in pair average diameter unit under which particles are\n"
    "    considered bonded. (default: 1.15)\n"
    "scale : bool\n"
    "    Scale separation by average diameter. (default: False)\n"
    "\n"
    "Returns\n"
    "-------\n"
    "separationsPair : (*,) list of (**, ***, 2) float numpy array\n"
    "    Separations between pairs.\n"
    "pairs : (*,) list of (***, 2) float numpy array\n"
    "    Pairs of neighbouring particles at `time0'.",
    pybind11::arg("filename"),
    pybind11::arg("time0"),
    pybind11::arg("deltat"),
    pybind11::arg("a1")=1.15,
    pybind11::arg("scale")=false);

  m.def("getVelocityCorrelationPair", &getVelocityCorrelationPair,
    "Compute difference between velocity autocorrelation in time and\n"
    "correlation of velocities of initially neighbouring particles.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "filename : str\n"
    "    Path to data file.\n"
    "time0 : (*,) int array-like\n"
    "    Initial frames.\n"
    "deltat : (**,) int array-like\n"
    "    Lag times.\n"
    "a1 : float\n"
    "    Distance in pair average diameter unit under which particles are\n"
    "    considered bonded. (default: 1.15)\n"
    "\n"
    "Returns\n"
    "-------\n"
    "correlationsPair : (*,) list of (**, ***, 2) float numpy array\n"
    "    Correlations between pairs.",
    pybind11::arg("filename"),
    pybind11::arg("time0"),
    pybind11::arg("deltat"),
    pybind11::arg("a1")=1.15);

  m.def(
    "getVelocityForcePropulsionCorrelation",
    &getVelocityForcePropulsionCorrelation,
    "Compute correlations between initial velocity and force, and between\n"
    "initial velocity and propulsion.\n"
    "\n"
    "NOTE: Centre-of-mass force and propulsions are removed.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "filename : str\n"
    "    Path to data file.\n"
    "time0 : (*,) int array-like\n"
    "    Initial frames.\n"
    "deltat : (**,) int array-like\n"
    "    Lag times.\n"
    "Heun : bool\n"
    "    Compute correction to forces from Heun integration.\n"
    "    (default: False).\n"
    "    NOTE: Translational noise is neglected.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "corVelocityForce : (*, **) float numpy array\n"
    "    Correlations between initial velocity and force.\n"
    "corVelocityPropulsion : (*, **) float numpy array\n"
    "    Correlations between initial velocity and propulsion.\n",
    pybind11::arg("filename"),
    pybind11::arg("time0"),
    pybind11::arg("deltat"),
    pybind11::arg("Heun")=true);

  m.def("getVelocityDifference", &getVelocityDifference,
    "Compute velocity difference in the radial direction for particles whose\n"
    "distance is in a certain range.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "filename : str\n"
    "    Path to data file.\n"
    "frame : int\n"
    "    Frame.\n"
    "nBins : int\n"
    "    Number of distance ranges to consider.\n"
    "rmin : float\n"
    "    Minimum radius. (default: 0)\n"
    "rmax : float\n"
    "    Maximum radius. (default: 0)\n"
    "    NOTE: if rmax == 0 then rmax = L/2.\n"
    "remove_cm : bool\n"
    "    Remove centre-of-mass displacement. (default: True)\n"
    "\n"
    "Returns\n"
    "-------\n"
    "bins : (nBins,) float numpy array\n"
    "    Distance ranges.\n"
    "differences : (nBins,) tuple of (*,) float numpy array\n"
    "    Velocity differences at each distance range.",
    pybind11::arg("filename"),
    pybind11::arg("frame"),
    pybind11::arg("nBins"),
    pybind11::arg("rmin")=0,
    pybind11::arg("rmax")=0,
    pybind11::arg("remove_cm")=true);

  m.def("selfIntScattFunc", &selfIntScattFunc,
    "Compute self-intermediate scattering function Fs.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "filename : str\n"
    "    Path to data file.\n"
    "time0 : (*,) int array-like\n"
    "    Initial frames.\n"
    "deltat : (**,) int array-like\n"
    "    Lag times.\n"
    "qn : float array-like\n"
    "    Wave-vector norms at which to compute the Fs.\n"
    "dq : float\n"
    "    Width of the wave-vector norm interval. (default: 0.1)\n"
    "remove_cm : bool\n"
    "    Remove centre-of-mass displacement. (default: True)\n"
    "\n"
    "Returns\n"
    "-------\n"
    "wv : (***,) list of (****, 2) float numpy array\n"
    "    Wave vectors at which the Fs is computed.\n"
    "Fs : (***, **, 3) float numpy array\n"
    "    Self-intermediate scattering function:\n"
    "    (0): lag time,\n"
    "    (1): self-intermediate scattering function,\n"
    "    (2): standard error on the self-intermediate scattering function.",
    pybind11::arg("filename"),
    pybind11::arg("time0"),
    pybind11::arg("deltat"),
    pybind11::arg("q"),
    pybind11::arg("dq")=0.1,
    pybind11::arg("remove_cm")=true);

  m.def("getBondsBrokenBonds", &getBondsBrokenBonds,
    "Compute number of bonds and broken bonds.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "positions : (*, 2) float array-like\n"
    "    Positions.\n"
    "displacements : (*, 2) float array-like\n"
    "    Displacements.\n"
    "diameters : (*,) float array-like\n"
    "    Diameters.\n"
    "L : float\n"
    "    System size.\n"
    "A1 : float\n"
    "    Distance in pair average diameter unit under which particles are\n"
    "    considered bonded. (default: 1.15)\n"
    "A2 : float\n"
    "    Distance in pair average diameter unit above which particles are\n"
    "    considered unbonded. (default: 1.25)\n"
    "\n"
    "Returns\n"
    "-------\n"
    "nBonds : (*,) float numpy array\n"
    "    Array of initial number of bonds.\n"
    "nBrokenBonds : (*,) float numpy array\n"
    "    Array of number of broken bonds.",
    pybind11::arg("positions"),
    pybind11::arg("displacements"),
    pybind11::arg("diameters"),
    pybind11::arg("L"),
    pybind11::arg("A1")=1.15,
    pybind11::arg("A2")=1.25);

  m.def("getVelocityVorticity", &getVelocityVorticity,
    "Compute Gaussian-smoothed velocitiy field and vorticity field.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "positions : (*, 2) float array-like\n"
    "    Positions.\n"
    "velocities : (*, 2) float array-like\n"
    "    Velocities.\n"
    "L : float\n"
    "    System size.\n"
    "nBoxes : int\n"
    "    Number of boxes in each direction of the computed grid.\n"
    "sigma : float\n"
    "    Standard deviation of the Gaussian with which to convolute.\n"
    "centre : (2,) float array-like\n"
    "    Return positions with respect to `centre'.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "pos : (nBoxes, nBoxes, 2) float numpy array\n"
    "    Grid positions.\n"
    "vel : (nBoxes, nBoxes, 2) float numpy array\n"
    "    Gaussian-smoothed velocities.\n"
    "vor : (nBoxes, nBoxes) float numpy array\n"
    "    Vorticities from Gaussian-smoothed velocities.\n"
    "vel_p : (*, 2) float numpy array\n"
    "    Gaussian-smoothed velocities at particle positions.\n"
    "vor_p : (*,) float numpy array\n"
    "    Vorticities from Gaussian-smoothed velocities at particle positions.",
    pybind11::arg("positions"),
    pybind11::arg("velocities"),
    pybind11::arg("L"),
    pybind11::arg("nBoxes"),
    pybind11::arg("sigma"),
    pybind11::arg("centre")=pybind11::array_t<double>(0));

  m.def("getPolarCharge", &getPolarCharge,
    "Compute grid of polar charge.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "grid : (*, **, 2) float array-like\n"
    "    Regular square grid of vectors.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "charge : (*, **) int numpy array\n"
    "    Corresponding grid of charges.",
    pybind11::arg("grid"));

  m.def("getRadialCorrelations", &getRadialCorrelations,
    "Compute radial correlations between `values1' (and `values2') associated\n"
    "to each of the positions of a system of size `L'.\n"
    ".. math::"
    "    C(|r1 - r0|) = \\langle values(r0) values2(r1)^* \\rangle\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "positions : (*, 2) float array-like\n"
    "    Positions of the particles.\n"
    "L : float\n"
    "    Size of the system box.\n"
    "values1 : (*, **) float array-like\n"
    "    Values to compute the correlations of `values2' with.\n"
    "    NOTE: if these values are 2D arrays, the sum of the correlations on\n"
    "          each axis is returned.\n"
    "values2 : (*, **) float array-like\\n"
    "    Values to compute the correlations of `values1' with.\n"
    "nBins : int\n"
    "    Number of intervals of distances on which to compute the\n"
    "    correlations.\n"
    "rmin : float\n"
    "    Minimum distance (included) at which to compute the correlations.\n"
    "    (default: 0)\n"
    "rmax : float\n"
    "    Maximum distance (excluded) at which to compute the correlations.\n"
    "    (default: 0)\n"
    "    NOTE: if max == 0 then max = L/2.\n"
    "rescale_pair_distribution : bool\n"
    "    Rescale correlations by pair distribution function. (default: False)\n"
    "\n"
    "Returns\n"
    "-------\n"
    "correlations : (nBins, 2) complex Numpy array\n"
    "    Array of (r, C(r)) where r is the lower bound of the bind and C(r)\n"
    "    the radial correlation computed for this bin.",
    pybind11::arg("positions"),
    pybind11::arg("L"),
    pybind11::arg("values1"),
    pybind11::arg("values2"),
    pybind11::arg("nBins"),
    pybind11::arg("rmin")=0,
    pybind11::arg("rmax")=0,
    pybind11::arg("rescale_pair_distribution")=false);

  m.def("getRadialDirectonCorrelations", &getRadialDirectonCorrelations,
    "Compute longitudinal and transversal radial correlations between\n"
    "`values1' (and `values2') associated to each of the positions of a\n"
    "system of size `L'.\n"
    ".. math::"
    "    C(|r1 - r0|) = \\langle values(r0) values2(r1)^* \\rangle\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "positions : (*, 2) float array-like\n"
    "    Positions of the particles.\n"
    "L : float\n"
    "    Size of the system box.\n"
    "values1 : (*, 2) float array-like\n"
    "    Values to compute the correlations of `values2' with.\n"
    "values2 : (*, 2) float array-like\\n"
    "    Values to compute the correlations of `values1' with.\n"
    "nBins : int\n"
    "    Number of intervals of distances on which to compute the\n"
    "    correlations.\n"
    "rmin : float\n"
    "    Minimum distance (included) at which to compute the correlations.\n"
    "    (default: 0)\n"
    "rmax : float\n"
    "    Maximum distance (excluded) at which to compute the correlations.\n"
    "    (default: 0)\n"
    "    NOTE: if max == 0 then max = L/2.\n"
    "rescale_pair_distribution : bool\n"
    "    Rescale correlations by pair distribution function. (default: False)\n"
    "\n"
    "Returns\n"
    "-------\n"
    "correlationsLong : (nBins, 2) complex Numpy array\n"
    "    Array of (r, Cl(r)) where r is the lower bound of the bind and Cl(r)\n"
    "    the longitudinal radial correlation computed for this bin.\n"
    "correlationsTrans : (nBins, 2) complex Numpy array\n"
    "    Array of (r, Ct(r)) where r is the lower bound of the bind and Ct(r)\n"
    "    the transversal radial correlation computed for this bin.\n",
    pybind11::arg("positions"),
    pybind11::arg("L"),
    pybind11::arg("values1"),
    pybind11::arg("values2"),
    pybind11::arg("nBins"),
    pybind11::arg("rmin")=0,
    pybind11::arg("rmax")=0,
    pybind11::arg("rescale_pair_distribution")=false);

  m.def("getRadialDirectionAmplitudes", &getRadialDirectionAmplitudes,
    "Compute mean squared amplitudes of longitudinal and transversal `values'\n"
    "associated to each of the positions of a system of size `L'.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "positions : (*, 2) float array-like\n"
    "    Positions of the particles.\n"
    "L : float\n"
    "    Size of the system box.\n"
    "values : (*, 2) float array-like\n"
    "    Values to compute the amplitudes of.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "amplitudesLong : complex\n"
    "    Mean squared amplitude in the longitudinal direction.\n"
    "amplitudesTrans : complex\n"
    "    Mean squared amplitude in the transversal direction.\n",
    pybind11::arg("positions"),
    pybind11::arg("L"),
    pybind11::arg("values"));

  m.def("getWCA", &getWCA,
    "Compute WCA potential.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "positions : (*, 2) float array-like\n"
    "    Positions.\n"
    "diameters : (*,) float array-like\n"
    "    Diameters.\n"
    "L : float\n"
    "    System size.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "U : float\n"
    "    WCA potential.\n",
    pybind11::arg("positions"),
    pybind11::arg("diameters"),
    pybind11::arg("L"));

  m.def("getWCAForces", &getWCAForces,
    "Compute Weeks-Chandler-Andersen (WCA) forces.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "positions : (*, 2) float array-like\n"
    "    Positions.\n"
    "diameters : (*,) float array-like\n"
    "    Diameters.\n"
    "L : float\n"
    "    System size.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "forces : (*, 2) float numpy array\n"
    "    WCA forces.\n",
    pybind11::arg("positions"),
    pybind11::arg("diameters"),
    pybind11::arg("L"));

  m.def("getRAForces", &getRAForces,
    "Compute regularised 1/r^a forces.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "positions : (*, 2) float array-like\n"
    "    Positions.\n"
    "diameters : (*,) float array-like\n"
    "    Diameters.\n"
    "L : float\n"
    "    System size.\n"
    "a : float\n"
    "    Potential exponent. (default: 12)\n"
    "rcut : float\n"
    "    Potential cut-off radius. (default: 1.25)\n"
    "\n"
    "Returns\n"
    "-------\n"
    "forces : (*, 2) float numpy array\n"
    "    Regularised 1/r^a forces.\n",
    pybind11::arg("positions"),
    pybind11::arg("diameters"),
    pybind11::arg("L"),
    pybind11::arg("a")=12,
    pybind11::arg("rcut")=1.25);

  m.def("getRAHessian", &getRAHessian,
    "Compute regularised pairwise 1/r^a-potential Hessian matrix.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "positions : (*, 2) float array-like\n"
    "    Positions.\n"
    "diameters : (*,) float array-like\n"
    "    Diameters.\n"
    "L : float\n"
    "    System size.\n"
    "a : float\n"
    "    Potential exponent. (default: 12)\n"
    "rcut : float\n"
    "    Potential cut-off radius. (default: 1.25)\n"
    "\n"
    "Returns\n"
    "-------\n"
    "hessian : (2 *, 2 *) flaot numpy array\n"
    "    Regularised 1/r^a-potential Hessian matrix.\n",
    pybind11::arg("positions"),
    pybind11::arg("diameters"),
    pybind11::arg("L"),
    pybind11::arg("a")=12,
    pybind11::arg("rcut")=1.25);

  // m.def("phononOrderParameter", &phononOrderParameter);

  pybind11::class_<IntegratePropulsions>(m, "IntegratePropulsions")
    .def(pybind11::init<std::string const&>())
    .def("getPropulsions", &IntegratePropulsions::getPropulsions,
      "Compute propulsions at frame.\n"
      "\n"
      "Parameters\n"
      "----------\n"
      "frame : int\n"
      "    Index of frame.\n"
      "\n"
      "Returns\n"
      "-------\n"
      "propulsions : (*, 2) float numpy array\n"
      "    Propulsions.",
      pybind11::arg("frame"));

  m.def("invertMatrix", &invertMatrix,
    "Compute inverse matrix.\n"
    "\n"
    "Uses GNU Science Library (GSL).\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "matrix : (*, *) float array-like\n"
    "    Matrix to invert.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "invMatrix : (*, *) float numpy array\n"
    "    Inverse matrix.\n",
    pybind11::arg("matrix"));

  m.def("invertSparseMatrix", &invertSparseMatrix,
    "Compute inverse of sparse matrix.\n"
    "\n"
    "Uses Eigen.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "matrix : (*, *) float array-like\n"
    "    Matrix to invert.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "invMatrix : (*, *) float numpy array\n"
    "    Inverse matrix.\n",
    pybind11::arg("matrix"));

}

#ifndef PYCPP_HPP
#define PYCPP_HPP

#include "maths.hpp"
#include "dat.hpp"
#include "add.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

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
  // Put the `nValues' `values' in an `histogram' with `nBins' bins linearly
  // spaced between `vmin' and `vmax'.

// POSITIONS AND DISPLACEMENTS
// Compute vectors of positions and displacements for faster computation in C++
// from .datN files.

std::vector<std::vector<std::vector<double>>> getPositions(
  std::string filename,
  int const& nTime0, int* const& time0);
  // Compute positions at the `nTime0' initial times `time0' from the .datN file
  // `filename'.

std::vector<std::vector<std::vector<std::vector<double>>>> getDisplacements(
  std::string filename,
  int const& nTime0, int* const& time0, int const& nDt, int* const& dt,
  bool remove_cm);
  // Compute displacements from the `nTime0' initial times `time0' with the
  // `nDt' lag times `dt' from the .datN file `filename'.
  // Remove centre of mass displacement if `remove_cm'.

std::vector<std::vector<std::vector<std::vector<double>>>> getDisplacements(
  std::string filename,
  int const& nTime0, int* const& time0, int const& nDt, int* const& dt,
  bool remove_cm);
  // Compute displacements from the `nTime0' initial times `time0' with the
  // `nDt' lag times `dt' from the .datN file `filename'.
  // Remove centre of mass displacement if `remove_cm'.

pybind11::array_t<double> getDisplacementsPYB(
  std::string const& filename,
  pybind11::array_t<int> time0, pybind11::array_t<int> deltat,
  bool const& remove_cm);
  // Compute displacements from the initial times `time0' with the lag times
  // `dt' from the .datN file `filename'.
  // Remove centre of mass displacement if `remove_cm'.

std::vector<std::vector<pybind11::array_t<double>>> getDisplacementsPair(
  std::string const& filename,
  pybind11::array_t<int> time0, pybind11::array_t<int> deltat,
  double const& a1);
  // Compute differences of displacements of initially paired particles (i.e. at
  // distance lesser than `a1' mean diameter) from the initial times `time0'
  // with the lag times `dt' from the .datN file `filename'.

std::vector<std::vector<pybind11::array_t<double>>> getSeparationsPair(
  std::string const& filename,
  pybind11::array_t<int> time0, pybind11::array_t<int> deltat,
  double const& a1, bool const& scale);
  // Compute differences of positions of initially paired particles (i.e. at
  // distance lesser than `a1' mean diameter) from the initial times `time0'
  // with the lag times `dt' from the .datN file `filename'.
  // Scale separation by average diameter if `scale'.

std::vector<pybind11::array_t<double>> getVelocityCorrelationPair(
  std::string const& filename,
  pybind11::array_t<int> time0, pybind11::array_t<int> deltat,
  double const& a1);
  // Compute difference between velocity autocorrelation in time and correlation
  // of velocities of initially neighbouring particles.
  // Computes paired particles (i.e. at distance lesser than `a1' mean diameter)
  // from the initial times `time0' with the lag times `dt' from the .datN file
  // `filename'.

std::vector<pybind11::array_t<double>> getVelocityForcePropulsionCorrelation(
  std::string const& filename,
  pybind11::array_t<int> time0, pybind11::array_t<int> deltat,
  bool Heun);
  // Compute correlations between force and initial velocity, and between
  // propulsion and initial velocity, from the initial times `time0' with the
  // lag times `dt' from the .datN file `filename'.

std::tuple<pybind11::array_t<double>, std::vector<pybind11::array_t<double>>>
  getVelocityDifference(
    std::string const& filename, double const& frame,
    int const& nBins, double const& rmin, double rmax,
    bool const& remove_cm);
  // Compute velocity difference in the radial direction for particles whose
  // distance is in a certain range.

std::tuple<std::vector<pybind11::array_t<double>>, pybind11::array_t<double>>
  selfIntScattFunc(
    std::string const& filename,
    pybind11::array_t<int> time0, pybind11::array_t<int> deltat,
    pybind11::array_t<double> qn, double const& dq, bool remove_cm);
  // Compute self-intermediate scaterring functions for lag times `deltat' from
  // initial times `time0', and with wave vectors `qn' and width of the
  // wave-vector norm interval `dq'.

// DISTANCES

double getDistance(
  double const& x0, double const& y0, double const& x1, double const& y1,
  double const& L);
  // Compute distance between (`x0', `y0') and (`x1', `y1') in a periodic square
  // system of size `L'.

bool withinDistance(
  double const& A, double* const& diameters, int const& i, int const& j,
  double const& distance);
  // Compute if particles `i' and `j' separated by `distance' are within
  // distance `A' relative to their diameters.

extern "C" int pairIndex(int i, int j, int N);
  // For `N' particles, return a unique pair index for the couples (`i', `j')
  // and (`j', `i') in {0, ..., `N'(`N' + 1)/2 - 1}.
  // (adapted from Paul Mangold)

extern "C" void getDifferences(
  int N, double L, double* x, double* y, double *diameters,
  double* differences_x, double* differences_y,
  bool scale_diameter);
  // Compute position differences between the `N' particles of a system of size
  // `L', with x-axis positions given by `x' and y-axis positions given by `y'.
  // Differences are rescaled by the sum of the radii of the particles in the
  // pair if `scale_diameter'.
  // NOTE: `differences_x' and `differences_y' must have (at least)
  //       `N'(`N' - 1)/2 entries.

extern "C" void getDistances(
  int N, double L, double* x, double* y, double* diameters, double* distances,
  bool scale_diameter = false);
  // Compute distances between the `N' particles of a system of size `L', with
  // x-axis positions given by `x' and y-axis positions given by `y'.
  // Distances are rescaled by the sum of the radii of the particles in the pair
  // if `scale_diameter'.
  // NOTE: distances must have (at least) `N'(`N' - 1)/2 entries.

extern "C" void getOrientationNeighbours(
  int N, double A1, double* diameters, double* distances,
  double* dx, double* dy,
  int* oneighbours);
  // Compute for each of the `N' particles the number of other particles, at
  // distance lesser than `A1' relative to their average diameter in
  // `distances', with the same orientation of displacement (`dx', `dy').
  // NOTE: `distances' must have at least `N'(`N' - 1)/2 entries and follow
  //       the indexing of pairs given by pairIndex (such as returned by
  //       getDistances).
  // NOTE: `oneighbours' must have at least `N' entries.

extern "C" void getBrokenBonds(
  int N, double A1, double A2, double* diameters,
  double* distances0, double* distances1,
  int* brokenBonds, bool* brokenPairs);
  // Compute for each of `N' particles the number of other particles which are
  // at distance lesser than `A1' in `distances0' and distance greater than `A2'
  // in `distances1' relative to their average diameter.
  // Broken pair indices are flagged as true in `brokenPairs'.
  // NOTE: `distances0' and `distances1' must have at least `N'(`N' - 1)/2
  //       entries and follow the indexing of pairs given by pairIndex (such as
  //       returned by getDistances).
  // NOTE: `brokenBonds' must have at least `N' entries.
  // NOTE: `brokenPairs' must have at least `N(N - 1)/2' entries.

std::vector<pybind11::array_t<double>> getBondsBrokenBonds(
  pybind11::array_t<double> const& positions,
  pybind11::array_t<double> const& displacements,
  pybind11::array_t<double> const& diameters,
  double const& L, double const& A1, double const& A2);
  // Compute array of initial number of particles with `positions' in a box of
  // size `L' at a distance lesser than `A1' in pair average `diameters' unit
  // for each particle, and array of number of initially bonded particles which
  // are at a distance greater than `A2' after `displacements'.

extern "C" void getVanHoveDistances(
  int N, double L, double* x, double* y, double* dx, double* dy,
  double* distances);
  // Compute van Hove distances between the `N' particles of a system of size
  // `L', with x-axis positions given by `x' and y-axis positions given by `y',
  // and x-axis displacements given by `dx' and y-axis displacements given by
  // `dy'.

extern "C" void nonaffineSquaredDisplacement(
  int N, double L, double* x0, double* y0, double* x1, double* y1,
  double A1, double* diameters, double* D2min);
  // Compute nonaffine squared displacements for `N' particles of a system of
  // size `L', with x-axis initial positions `x0', y-axis initial positions
  // `y0', x-axis final positions `x1', and y-axis final positions `y1',
  // considering that neighbouring particles are within a distance `A1'
  // relative to their `diameters'.

extern "C" void pairDistribution(
  int nBins, double vmin, double vmax, double* histogram,
  int N, double L, double* x, double* y, double* diameters,
  bool scale_diameter = false);
  // Compute pair distribution function as `histogram' with `nBins' bins
  // linearly spaced between `vmin' and `vmax' from the distances between the
  // `N' particles of a system of size `L', with x-axis positions given by `x'
  // and y-axis positions given by `y'.
  // Distances are rescaled by the sum of the radii of the particles in the pair
  // if `scale_diameter'.

extern "C" void S4Fs(
  const char* filename,
  int nTime0, int* time0, int nDt, int* dt,
  int nq, double *qx, double *qy, int nk, double *kx, double *ky,
  double *S4, double *S4var);
  // Compute from the .datN file `filename' the four-point structure factor of
  // the real part of the self-intermediate scattering functions, computed at
  // the `nk' wave-vectors (`kx', `ky'), along the `nq' wave-vectors
  // (`qx', `qy'), from the positions and displacements from the `nTime0'
  // initial times `time0' with the `nDt' lag times `dt'.
  // Means are saved in `S4' and variances are saved in `S4var', for each value
  // of the lag time.

extern "C" void getLocalParticleDensity(
  int N, double L, double a, double* x, double* y, double* diameters,
  double* densities);
  // Compute for each of the `N' particles of a system of size `L', with x-axis
  // positions given by `x' and y-axis positions given by `y', the sum of the
  // areas of the particles in a box of size `a' centred on the particle divided
  // by the area of the box.
  // Particle areas are computed with a factor 2^(1/6) on diameters.

extern "C" void isNotInBubble(
  int N, double L, double philim, double dlim,
  double* x, double* y, double* densities,
  bool* notInBubble);
  // Determine which of the `N' particles of a system of size `L', with x-axis
  // positions given by `x' and y-axis positions given by `y', are not within
  // distance `dlim' of particles with `densities' below `philim'.

// POTENTIAL AND FORCES

double getWCA(
  pybind11::array_t<double> const& positions,
  pybind11::array_t<double> const& diameters,
  double const& L);
  // Compute WCA potential between particles at `positions', with `diameters',
  // in a periodic square box of linear size `L'.

pybind11::array_t<double> getWCAForces(
  pybind11::array_t<double> const& positions,
  pybind11::array_t<double> const& diameters,
  double const& L);
  // Compute forces from Weeks-Chandler-Andersen potential, between particles at
  // `positions', with `diameters', in a periodic square box of linear size `L'.

pybind11::array_t<double> getRAForces(
  pybind11::array_t<double> const& positions,
  pybind11::array_t<double> const& diameters,
  double const& L, double const& a, double const& rcut);
  // Compute regularised 1/r^`a' potential, with cut-off radius `rcut', between
  // particles at `positions', with `diameters', in a periodic square box of
  // linear size `L'.

pybind11::array_t<double> getRAHessian(
  pybind11::array_t<double> const& positions,
  pybind11::array_t<double> const& diameters,
  double const& L, double const& a, double const& rcut);
  // Compute regularised 1/r^`a'-potential Hessian matrix, with cut-off radius
  // `rcut', between particles at `positions', with `diameters', in a periodic
  // square box of linear size `L'.

// VELOCITIES

std::vector<pybind11::array_t<double>> getVelocityVorticity(
  pybind11::array_t<double> const& positions,
  pybind11::array_t<double> const& velocities,
  double const& L, int const& nBoxes, double const& sigma,
  pybind11::array_t<double> const& centre = pybind11::array_t<double>(0));
  // Compute Gaussian-smoothed velocitiy field and vorticity field, using
  // standard deviation `sigma', on a (`nBoxes', `nBoxes')-grid, from
  // `positions' and `velocities', in a system of size `L'.
  // Return positions with respect to `centre'.

// GRIDS

extern "C" void toGrid(
  int N, double L, double* x, double* y, double* values,
  int nBoxes, double* grid, bool average = false);
  // Maps square (sub-)system of `N' particles with positions (`x', `y') centred
  // around 0 and of (cropped) size `L' to a flattened square `grid' of size
  // `nBoxes'^2, and associates to each box the sum or averaged value of the
  // (`N',)-array `values'.

extern "C" void g2Dto1Dgrid(
  int nBoxes, double* g2D, double* grid,
  double* g1D, double* radii, int* nRadii);
  // Returns cylindrical average of flattened square grid `g2D' of size
  // `nBoxes'^2 with values of radii given by flatten square grid `grid' of same
  // size, as `nRadii'[0] values of `g1D' at corresponding `radii'.

extern "C" void g2Dto1Dgridhist(
  int nBoxes, double* g2D, double* grid,
  int nBins, double vmin, double vmax, double* g1D, double* g1Dstd);
  // Returns cylindrical average of flattened square grid `g2D' of size
  // `nBoxes'^2 with values of radii given by flatten square grid `grid' of same
  // size, as histogram `g1D' with `nBins' between `vmin' and `vmax' and
  // standard variation on this measure `g1Dstd'.

pybind11::array_t<int> getPolarCharge(pybind11::array_t<double> grid);
  // Return grid of polar charge.

// CORRELATIONS

pybind11::array_t<std::complex<double>> getRadialCorrelations(
  pybind11::array_t<double> positions, double L,
  pybind11::array_t<std::complex<double>> values1,
  pybind11::array_t<std::complex<double>> values2,
  int nBins, double rmin, double rmax,
  bool rescale_pair_distribution);

std::vector<pybind11::array_t<std::complex<double>>>
  getRadialDirectonCorrelations(
  pybind11::array_t<double> positions, double L,
  pybind11::array_t<std::complex<double>> values1,
  pybind11::array_t<std::complex<double>> values2,
  int nBins, double rmin, double rmax,
  bool rescale_pair_distribution);

pybind11::array_t<std::complex<double>> getRadialDirectionAmplitudes(
  pybind11::array_t<double> positions, double L,
  pybind11::array_t<std::complex<double>> values);

extern "C" void getVelocitiesOriCor(
  int N, double L, double* x, double* y, double* vx, double* vy,
  double* correlations,
  double sigma);
  // Compute radial correlations of orientation of velocities (`vx', `vy')
  // associated to each of the `N' particles of a system of size `L', with
  // x-axis positions given by `x' and y-axis positions given by `y', and mean
  // diameter `sigma'.

// READ

extern "C" void readDouble(
  const char* filename, int nTargets, long int* targets, double* out);
  // Read `nTargets' doubles in `filename' at `targets' and output in `out'.

class IntegratePropulsions;

// LINEAR ALGEBRA

pybind11::array_t<double>
  invertMatrix(pybind11::array_t<double> const& matrix);
  // Invert matrix.

pybind11::array_t<double>
  invertSparseMatrix(pybind11::array_t<double> const& matrix);
  // Invert matrix.

#endif

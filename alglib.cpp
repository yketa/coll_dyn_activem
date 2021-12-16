#include <alglib/alglibinternal.cpp>
#include <alglib/alglibmisc.cpp>
#include <alglib/ap.cpp>
#include <alglib/dataanalysis.cpp>
#include <alglib/integration.cpp>
#include <alglib/linalg.cpp>
#include <alglib/optimization.cpp>
#include <alglib/solvers.cpp>
#include <alglib/specialfunctions.cpp>
#include <alglib/statistics.cpp>

#include "alglib.hpp"

std::vector<int> clusters2D(
  int const& N, double const& L, double* positions, double const& r) {
  // Construct clusters from `N' particles in system of size `L' at `positions'
  // which are at a minimum distance `r' apart.
  // Particle `i' is at position (`positions[2*i]', `positions[2*i + 1]').
  // Returns array `clusters' of `N + 1' integers where `clusters[0]'' is the
  // number of different clusters, and `clusters[i + 1]' is the cluster index
  // of the `i'-th particles.

  alglib::clusterizerstate s;
  alglib::ahcreport rep;

  // distances between points
  alglib::real_2d_array d;
  d.setlength(N, N);
  for (int i=0; i < N; i++) {
    for (int j=i + 1; j < N; j++) {
      d[i][j] = sqrt(
        pow(algDistPeriod(positions[2*i], positions[2*j], L), 2)
        + pow(algDistPeriod(positions[2*i + 1], positions[2*j + 1], L), 2));
    }
  }

  // https://www.alglib.net/dataanalysis/clustering.php
  alglib::clusterizercreate(s);
  alglib::clusterizersetdistances(s, d, true);
  alglib::clusterizersetahcalgo(s, 1); // single linkage
  alglib::clusterizerrunahc(s, rep);

  alglib::ae_int_t k;
  alglib::integer_1d_array cidx;
  alglib::integer_1d_array cz;
  alglib::clusterizerseparatedbydist(rep, r, k, cidx, cz);

  std::vector<int> clusters(1, k);
  for (int i=0; i < N; i++) { clusters.push_back(cidx[i]); }

  return clusters;
}

#ifndef ALGLIB_HPP
#define ALGLIB_HPP

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

#include <functional>
#include <iostream>

#include "maths.hpp"


/////////////
// CLASSES //
/////////////


class CGMinimiser;


/*  CGMINIMISER
 *  -----------
 *  Wrapper of ALGLIB's conjugate gradient minimiser of function.
 *  (see https://www.alglib.net/translator/man/manual.cpp.html)
 */

class CGMinimiser {

  public:

    // CONSTRUCTORS

    CGMinimiser(
      //   function:   f[*x,      *f(x),   *grad_f(x)]
      std::function<void(double*, double*, double*)>
        function, // function to minimise
      // stopping conditions (see https://www.alglib.net/translator/man/manual.cpp.html#sub_mincgsetcond)
      int argsize,
      double stoppingGradient = 0, // `EpsG': condition on the value of the gradient
      double stoppingFunction = 0, // `EpsF': condition on consecutive values of the function
      double stoppingArgument = 0, // `EpsX': condition on the step size
      long int maximumIterations = 0 // `MaxIts': maximum number of iterations
      ) :
      inputFunction(function),
      dim(argsize),
      EpsG(stoppingGradient),
      EpsF(stoppingFunction),
      EpsX(stoppingArgument),
      MaxIts(maximumIterations)
      {}

    // DESTRUCTORS

    ~CGMinimiser() {;}

    // METHODS

    static void forwarder(
      const alglib::real_1d_array& x, double& func, alglib::real_1d_array& grad,
      void* ptr) {
      // Static member function to forward to function to minimise.

      static_cast<CGMinimiser*>(ptr)->inputFunction
        ((double*) x.getcontent(), &func, (double*) grad.getcontent());
    }

    static void rep(const alglib::real_1d_array &x, double func, void* ptr) {;}
      // Static member function for input of alglib::mincgoptimize.

    alglib::mincgreport minimise(double* minimum) {
      // Perform minimisation of inputFunction starting from minimum.

      // initialisation
      // std::cout << "starting minimisation (ALGLIB)" << std::endl;

      // optimization report
      // (see https://www.alglib.net/translator/man/manual.cpp.html#struct_mincgreport)
      alglib::mincgreport report;

      // state of the nonlinear CG optimizer
      // (see https://www.alglib.net/translator/man/manual.cpp.html#struct_mincgstate)
      alglib::mincgstate state;

      // minimum argument array
      alglib::real_1d_array x;
      x.setcontent(dim, minimum);

      // minimisation
      // (see https://www.alglib.net/translator/man/manual.cpp.html#sub_mincgoptimize)
      alglib::mincgcreate(x, state);
      alglib::mincgsetcond(state, EpsG, EpsF, EpsX, MaxIts);
      // alglib::mincgsetscale(state, s);
      alglib::mincgoptimize(state, &forwarder, &rep, (void*) this);
      alglib::mincgresults(state, x, report);
      // std::cout << "completion code: "
      //   << (int) report.terminationtype
      //   << std::endl;
      // std::cout << "iteration count: "
      //   << (int) report.iterationscount
      //   << std::endl;

      // result
      for (int i=0; i < dim; i++) { minimum[i] = x[i]; }

      return report;
    }

  private:

    // ATTRIBUTES

    std::function
      <void(double*, double*, double*)>
      inputFunction; // function to minimise

    int dim; // dimension of input variable

    double EpsG; // condition on the value of the gradient
    double EpsF; // condition on consecutive values of the function
    double EpsX; // condition on the step size
    long int MaxIts; // maximum number of iterations

};


///////////////
// FUNCTIONS //
///////////////


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

#endif

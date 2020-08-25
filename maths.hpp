#ifndef MATHS_HPP
#define MATHS_HPP

#include <random>
#include <cmath>
#include <ostream>

/////////////
// CLASSES //
/////////////

class Random {
  /*  This simple rnd class is a wrapper for the built-in c++ random number
   *  generator.
   *  (adapted from RLJ)
   */

  public:

    // CONTRUCTORS

    Random(int seed = 0, double g_co = 3) :
      g_cutoff(g_co) { generator.seed(seed); }
    Random(std::default_random_engine rndeng, double g_co = 3) :
      generator(rndeng), g_cutoff(g_co) {;}

    // DESTRUCTORS

    ~Random() { delete intmax; delete real01; delete normal; }

    // METHODS

    // random number engine class (e.g. for saving purposes)
    std::default_random_engine getGenerator() { return generator; }
    void setGenerator(std::default_random_engine rndeng) { generator = rndeng; }

    // member functions for generating random double in [0,1] and random integer in [0,max-1]
    double random01() { return (*real01)(generator); }
    int randomInt(int max) { return (*intmax)(generator) % max; }
    double gauss() { return (*normal)(generator); }
    double gauss_cutoff() {
      double g = this->gauss();
      while (fabs(g) > g_cutoff) {
        g = this->gauss();
      }
      return g;
    }

    // overload << operator
    friend std::ostream& operator << (std::ostream& os, const Random& rnd) {
      os << "**RANDOM GENERATOR**";
      return os;
    }

  private:

    // ATTRIBUTES

    std::default_random_engine generator;
    int max = 0x7fffffff;

    std::uniform_int_distribution<int>* intmax
      = new std::uniform_int_distribution<int>(0, max);
    std::uniform_real_distribution<double>* real01
      = new std::uniform_real_distribution<double>(0.0, 1.0);
    std::normal_distribution<double>* normal
      = new std::normal_distribution<double>(0.0, 1.0);

    const double g_cutoff; // cut-off for Gaussian white noise

};


////////////////
// PROTOTYPES //
////////////////

double getAngle(double cosinus, double signSinus);
  // Returns angle in radians from its cosinus and sign of its sinus.

double algDistPeriod(double const& x1, double const& x2, double const& length);
  // Returns algebraic distance from `x1' to `x2' on a line of length `length'
  // taking into account periodic boundary condition.

double dist2DPeriod(double* pos0, double* pos1, double const& length);
  // Returns distance between points on a plane, with positions `pos0' and
  // `pos1' taking into account period boundary condition in a square system
  // of size `length'.

#endif

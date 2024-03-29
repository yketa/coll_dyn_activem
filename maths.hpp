#ifndef MATHS_HPP
#define MATHS_HPP

#include <random>
#include <cmath>
#include <ostream>
#include <bits/stdc++.h>

/////////////
// CLASSES //
/////////////

#define RNDG std::mt19937 // Mersenne Twister
class Random {
  /*  This simple Random class is a wrapper for the built-in c++ random number
   *  generator.
   *  (adapted from RLJ)
   */

  public:

    // CONTRUCTORS

    Random(int const& seed = 0, double const& g_co = 3) :
      g_cutoff(g_co) { generator.seed(seed); }
    Random(RNDG rndeng, double const& g_co = 3) :
      generator(rndeng), g_cutoff(g_co) {;}

    // DESTRUCTORS

    ~Random() { delete intmax; delete real01; delete normal; }

    // METHODS

    // random number engine class (e.g. for saving purposes)
    RNDG getGenerator() { return generator; }
    void setGenerator(RNDG rndeng) { generator = rndeng; }
    std::normal_distribution<double>* getNormal() { return normal; }

    // member functions for generating random double in [0,1] and random integer in [0,max-1]
    double random01() { return (*real01)(generator); }
    int randomInt(int const& max) { return (*intmax)(generator) % max; }
    // member functions for generating normally distributed random doubles
    double gauss() { return (*normal)(generator); }
    double gauss_cutoff() {
      double g = this->gauss();
      while (fabs(g) > g_cutoff) {
        g = this->gauss();
      }
      return g;
    }
    double gauss(double const& mean, double const& std) {
      return std::normal_distribution<double>(mean, std)(generator);
    }

    // overload << operator
    friend std::ostream& operator << (std::ostream& os, const Random& rnd) {
      os << "**RANDOM GENERATOR**";
      return os;
    }

  private:

    // ATTRIBUTES

    RNDG generator;
    // std::default_random_engine generator; // (old) period = 1,686,629,820
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

double getAngleVector(double x, double y);
  // Returns angle in radians from the coordinates of a vector.

double algDistPeriod(double const& x1, double const& x2, double const& length);
  // Returns algebraic distance from `x1' to `x2' on a line of length `length'
  // taking into account periodic boundary condition.

double dist2DPeriod(double* pos0, double* pos1, double const& length,
  double* diff);
  // Returns distance from point `0' to `1' on a plane, with positions `pos0'
  // and `pos1' taking into account period boundary condition in a square system
  // of size `length', and saving in `diff' the difference vector.


///////////////
// FUNCTIONS //
///////////////

template<class type> std::vector<std::vector<type>> invert2x2(
  std::vector<std::vector<type>> const& matrix) {
  // Returns from [[a, b], [c, d]] the inverse matrix
  // [[d, -b], [-c, a]]/(a d - b c).

  type det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0];
  if ( det == 0 ) {
    throw std::invalid_argument("Cannot invert matrix with determinant 0.");
  }

  std::vector<std::vector<type>> inverse = {
    {matrix[1][1]/det, -matrix[0][1]/det},
    {-matrix[1][0]/det, matrix[0][0]/det}};
  return inverse;
}

template<class vecClass> std::vector<vecClass>*
	sortVec(std::vector<vecClass>* vec) {
	// Sort vector `vec', remove duplicate entries, and returns pointer to it.

	std::sort(vec->begin(), vec->end());
  vec->erase(std::unique(vec->begin(), vec->end()), vec->end());
	return vec;
}

template<class vecClass> std::vector<vecClass>*
  removeVec(std::vector<vecClass>* vec, vecClass element) {
  // Remove value `element' from vec.

  vec->erase(std::remove(vec->begin(), vec->end(), element), vec->end());
  return vec;
}

template<class vecClass> bool
  isInSortedVec(std::vector<vecClass> const* vec, vecClass element) {
  // Returns if `element' is in `vec' assuming that `vec' is sorted.

  typename std::vector<vecClass>::const_iterator lower =
    std::lower_bound(vec->begin(), vec->end(), element);
  return ( lower != vec->end() && *lower == element );
}

template<typename T> bool compareVec(std::vector<T>& v1, std::vector<T>& v2) {
  // Check if two vectors have the same elements.

  std::sort(v1.begin(), v1.end());
  std::sort(v2.begin(), v2.end());
  return v1 == v2;
}

#endif

#include <cmath>
#include <math.h>

#include "maths.hpp"

double getL(double phi, std::vector<double> const& diameters) {
  // Returns the length of a square system with packing fraction `phi'
  // containing particles with `diameters'.

  double totalArea = 0.0;
  for (auto i = diameters.begin(); i != diameters.end(); i++) {
    totalArea += M_PI*pow(*i, 2)/4.0;
  }
  return sqrt(totalArea/phi);
}

double getL(double phi, int N, double diameter) {
  // Returns the length of a square system with packing fraction `phi'
  // containing  `N' particles with same `diameter'.

  return getL(phi, std::vector<double>(N, diameter));
}

double getAngle(double cosinus, double signSinus) {
  // Returns angle in radians from its cosinus and sign of its sinus.

  double angle = acos(cosinus);
  angle *= signSinus > 0 ? 1 : -1;
  return angle;
}

double getAngleVector(double x, double y) {
  // Returns angle in radians from the coordinates of a vector.

  return getAngle(x/sqrt(pow(x, 2.0) + pow(y, 2.0)), y);
}

double algDistPeriod(double const& x1, double const& x2, double const& length) {
  // Returns algebraic distance from `x1' to `x2' on a line of length `L' taking
  // into account periodic boundary condition.

  double diff = x2 - x1;

  if ( fabs(diff) > length/2 ) {
    double diff1 = fabs(x1) + fabs(length - x2);
    double diff2 = fabs(length - x1) + fabs(x2);
    if ( diff1 < diff2 ) { diff = diff1; }
    else { diff = diff2; }
    diff *= (x2 > x1 ? -1 : 1);
  }

  return diff;
}

double dist2DPeriod(double* pos0, double* pos1, double const& length) {
  // Returns distance between points on a plane, with positions `pos0' and
  // `pos1' taking into account period boundary condition in a square system
  // of size `length'.

  return sqrt(
      pow(
        algDistPeriod( // separation in x position
          pos0[0],
          pos1[0],
          length),
        2)
      + pow(
        algDistPeriod( // separation in y position
          pos0[1],
          pos1[1],
          length),
        2));
}

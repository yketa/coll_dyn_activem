#include <cmath>
#include <math.h>

#include "maths.hpp"

double getAngle(double cosinus, double signSinus) {
  // Returns angle in radians from its cosinus and sign of its sinus.

  double angle = acos(cosinus);
  angle *= signSinus > 0 ? 1 : -1;
  return angle;
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

#include <cmath>
#include <math.h>

#include "maths.hpp"

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
  double absDiff = fabs(diff);
  if ( absDiff > length/2 ) return (x2 > x1 ? 1 : -1)*(absDiff - length);
  return diff;
}

double dist2DPeriod(double* pos0, double* pos1, double const& length,
  double* diff) {
  // Returns distance between points on a plane, with positions `pos0' and
  // `pos1' taking into account period boundary condition in a square system
  // of size `length', and saving in `diff' the difference vector.

  diff[0] = algDistPeriod(pos0[0], pos1[0], length); // separation in x position
  diff[1] = algDistPeriod(pos0[1], pos1[1], length); // separation in x position

  return sqrt(pow(diff[0], 2) + pow(diff[1], 2));
}

#ifndef ENV_HPP
#define ENV_HPP

#include <string>

////////////////
// PROTOTYPES //
////////////////

double getEnvDouble(std::string const& envName, double const& defaultValue = 0);
  // Returns environment variable as double.

int getEnvInt(std::string const& envName, int const& defaultValue = 0);
  // Returns environment variable as integer.

bool getEnvBool(std::string const& envName, bool const& defaultValue = 0);
  // Returns environment variable as boolean.

std::string getEnvString(std::string const& envName,
  std::string const& defaultValue = "");
  // Returns environment variable as string.

#endif

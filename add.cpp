#include <iostream>
#include <string>

#include "add.hpp"
#include "dat.hpp"
#include "env.hpp"
#include "particle.hpp"

int main() {

  // INPUT
  DatN dat(getEnvString("FILE", "in.datN"));
  const int frame = getEnvInt("FRAME", 0);
  std::vector<double> positions;
  std::vector<double> propulsions;
  for (int i=0; i < dat.getNumberParticles(); i++) {
    for (int dim=0; dim < 2; dim++) {
      positions.push_back(dat.getPosition(
        frame, i, dim, false));
      propulsions.push_back(dat.getPropulsion(
        frame
        #ifdef ADD_NEXT_PROPULSION
        + 1
        #endif
        , i, dim));
    }
  }

  // ADD OBJECT
  int init = getEnvInt("INIT", 0);
  int Niter = getEnvInt("NITER", 1);
  ADD add(
    dat.getNumberParticles(),
    dat.getSystemSize(),
    dat.getDiameters(),
    getEnvDouble("F", dat.getPropulsionVelocity()),
    getEnvDouble("DT", 1e-2),
    init,
    Niter,
    getEnvInt("LAGMIN", 0),
    getEnvInt("LAGMAX", 1),
    getEnvInt("NMAX", 0),
    getEnvInt("INTMAX", Niter),
    getEnvInt("SEED", 0),
    getEnvString("OUT", "out.datN"));
  for (int i=0; i < add.getNumberParticles(); i++) {
    for (int dim=0; dim < 2; dim++) {
      add.getPosition(i)[dim] = positions[2*i + dim];
      if ( ! getEnvBool("RANDOMISE_PROPULSIONS", true) ) {
        add.getPropulsion(i)[dim] = propulsions[2*i + dim];
      }
    }
  }
  // std::cout << "INITIAL FRAME: U/N = "
  //   << add.potential()/add.getNumberParticles() << std::endl;
  add.saveInitialState();

  // MINIMISATION
  int maxIter = getEnvInt("MAXITER", 0);
  for (int i=0; i < init + Niter; i++) {
    add.minimiseUeff(maxIter);
    // std::cout << "FRAME " << i << ": U/N = " <<
    //   add.potential()/add.getNumberParticles() <<  " sqrt(gradUeff2/N) = "
    //   << sqrt(add.gradientUeff2()/add.getNumberParticles()) << std::endl;
    add.saveNewState();
    add.iteratePropulsion();
  }

}

#include <iostream>
#include <string>

#include "add.hpp"
#include "dat.hpp"
#include "env.hpp"
#include "particle.hpp"

////////////////////////////////////////////////////////
// TIME FOR EACH MINIMISATION AND SQUARED VELOCITY IN MD
#include <chrono>
#include "readwrite.hpp"
////////////////////////////////////////////////////////

int main() {

  // INPUT
  DatN dat(getEnvString("FILE", "in.datN"));
  const int frame = getEnvInt("FRAME", 0);
  std::vector<double> positions;
  std::vector<double> propulsions;
  for (int i=0; i < dat.getNumberParticles(); i++) {
    for (int dim=0; dim < 2; dim++) {
      positions.push_back(dat.getPosition(
        frame
        #ifdef ADD_NEXT_PROPULSION
        + 1
        #endif
        , i, dim, false));
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
      add.getPropulsion(i)[dim] = propulsions[2*i + dim];
      /////////////////////
      // RANDOM PROPULSIONS
      // add.getPropulsion(i)[dim] =
      //   (add.getRandomGenerator())->gauss(0, add.getVelocity());
      /////////////////////
    }
  }
  std::cout << "INITIAL FRAME: U/N = "
    << add.potential()/add.getNumberParticles() << std::endl;
  add.saveInitialState();

  // MINIMISATION
  /////////////////////////////
  // TIME FOR EACH MINIMISATION
  Write clock((add.getOuput())->getOutputFile() + ".time");
  std::chrono::time_point<std::chrono::high_resolution_clock> time;
  /////////////////////////////
  int maxIter = getEnvInt("MAXITER", 0);
  for (int i=0; i < init + Niter; i++) {
    /////////////////////////////
    // TIME FOR EACH MINIMISATION
    time = std::chrono::high_resolution_clock::now();
    /////////////////////////////
    add.minimiseUeff(maxIter);
    /////////////////////////////
    // TIME FOR EACH MINIMISATION
    clock.write<double>(
        std::chrono::duration<double, std::milli>
          (std::chrono::high_resolution_clock::now() - time)
          .count());
    /////////////////////////////
    std::cout << "FRAME " << i << ": U/N = " <<
      add.potential()/add.getNumberParticles() <<  " sqrt(gradUeff2/N) = "
      << sqrt(add.gradientUeff2()/add.getNumberParticles()) << std::endl;
    add.saveNewState();
    add.iteratePropulsion();
  }

  /////////////////////////
  // SQUARED VELOCITY IN MD
  #ifdef ADD_MD
  assert (add.n_p_events == add.velo2MD.size());
  Write out_velo2MD((add.getOuput())->getOutputFile() + ".velo2MD");
  out_velo2MD.write<double>(add.dtMDstep);
  out_velo2MD.write<int>(add.n_p_events);
  out_velo2MD.write<int>(add._velo2MD.size());
  for (int i=0; i < add.velo2MD.size(); i++) {
    for (double v2MD : add.velo2MD[i]) {
      out_velo2MD.write<double>(v2MD);
    }
  }
  #endif
  /////////////////////////

}

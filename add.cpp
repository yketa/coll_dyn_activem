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
  int numberParticles = getEnvInt("N", dat.getNumberParticles());
  int ratioNumberParticles = numberParticles/dat.getNumberParticles(); // ratio of number of particles
  int nCopyCells = round(sqrt(ratioNumberParticles));
  if ( numberParticles != ratioNumberParticles*dat.getNumberParticles()
    || ratioNumberParticles != nCopyCells*nCopyCells ) {
    // ratio of number of particles has to be a perfect square to copy correctly
    throw std::invalid_argument(
      "Ratio of number of particles is not a perfect square.");
  }
  auto mapParticleIndex = // mapping from particle index to input particle index
    [&dat](int i){ return i%dat.getNumberParticles(); };
  auto copyCellIndex = // mapping from particle to index of copy cell
    [&dat](int i){ return i/dat.getNumberParticles(); };
  int index;
  std::vector<double> positions;
  std::vector<double> propulsions;
  const std::vector<double> datDiameters = dat.getDiameters();
  std::vector<double> diameters;
  for (int i=0; i < numberParticles; i++) {
    for (int dim=0; dim < 2; dim++) {
      index = mapParticleIndex(i);
      positions.push_back(
        dat.getPosition(frame, index, dim, false)
        + dat.getSystemSize()*(dim == 0 ?
          copyCellIndex(i) % nCopyCells : copyCellIndex(i) / nCopyCells));
      propulsions.push_back(dat.getPropulsion(
        frame
        #ifdef ADD_NEXT_PROPULSION
        + 1
        #endif
        , index, dim));
    }
    diameters.push_back(datDiameters[index]);
  }

  // ADD OBJECT
  int init = getEnvInt("INIT", 0);
  int Niter = getEnvInt("NITER", 1);
  ADD add(
    numberParticles,
    nCopyCells*dat.getSystemSize(),
    diameters,
    getEnvDouble("F", dat.getPropulsionVelocity()),
    getEnvDouble("DT", 1e-2),
    init,
    Niter,
    getEnvInt("LAGMIN", 0),
    getEnvInt("LAGMAX", 1),
    getEnvInt("NMAX", 0),
    getEnvInt("INTMAX", Niter),
    getEnvDouble("DTMD", 5e-4),
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

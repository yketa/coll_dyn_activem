#include <iostream>
#include <string>
#include <csignal>

#include "add.hpp"
#include "dat.hpp"
#include "env.hpp"
#include "particle.hpp"

ADD* addPTR;

void signalHandler(int signum) {
  // Signal handler which saves last computed configuration and flushes all the
  // output files.

  std::cerr
    << "Received signal " << signum
    << " (frame " << (addPTR->getSystem())->getDump()[0] << ")."
    << std::endl;

  // create system object with latest computed coordinates
  SystemN signal_system(0, 1, 0, new int(1), 0, 1,
    new std::vector<int>, new std::vector<int>, addPTR->getSystem(), -1,
    (addPTR->getSystem())->getOutputFile() + ".signal");
  signal_system.saveInitialState(); // save
  // current particules state
  std::vector<Particle> newParticles;
  for (int i=0; i < addPTR->getNumberParticles(); i++) {
    newParticles.push_back((addPTR->getSystem())->getParticle(i));
    for (int dim=0; dim < 2; dim++) {
      newParticles[i].position()[dim] += // WARNING: we assume that particles do not move more further than a half box length
        algDistPeriod( // equivalent position at distance lower than half box
          newParticles[i].position()[dim],
          addPTR->getPosition(i)[dim] // wrapped coordinate
            - (wrapCoordinate<SystemN>
                (addPTR->getSystem(), addPTR->getPosition(i)[dim])
              *addPTR->getSystemSize()),
          addPTR->getSystemSize());
      newParticles[i].propulsion()[dim] = addPTR->getPropulsion(i)[dim];
    }
    newParticles[i].orientation()[0] =
      getAngleVector(addPTR->getPropulsion(i)[0], addPTR->getPropulsion(i)[1]);
  }
  signal_system.saveNewState(newParticles); // save

  // flush everything
  signal_system.flushOutputFile();
  (addPTR->getSystem())->flushOutputFile();
  (addPTR->getOuput())->flush();

  // exit
  std::exit(signum);
}

int main() {

  // register signal SIGINT/SIGTERM and signal handler
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

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
    index = mapParticleIndex(i);
    for (int dim=0; dim < 2; dim++) {
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
  addPTR = &add;
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

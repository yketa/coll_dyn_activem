#include "env.hpp"
#include "iteration.hpp"
#include "particle.hpp"

int main() {

  // VARIABLE DEFINITION

  // random number generator
  int seed = getEnvInt("SEED", 1); // random seed

  // simulation
  double dt = getEnvDouble("DT", 1e-3); // time step
  int Niter = getEnvInt("NITER", 1000000); // number of iterations

  // active work computation
  int nWork = getEnvInt("NWORK", 0); // number of frames on which to compute active work

  // output
  std::string filename = getEnvString("FILE", "out.dat"); // output file name
  bool dump = getEnvBool("DUMP", 1); // dump positions and orientations to output file
  int period = getEnvInt("PERIOD", 1); // period of dumping of positions and orientations in number of frames

  // SYSTEM

  // simulation
  auto simulate = [&Niter] (System* system) { // use of lambda function to enable conditional definition of system
    // INITIALISATION
    system->saveInitialState(); // save first frame
    // ITERATION
    iterate_ABP_WCA(system, Niter); // run simulations
  };

  // definition
  std::string inputFilename = getEnvString("INPUT_FILENAME", ""); // input file from which to copy data
  if ( inputFilename == "" ) { // set parameters from environment variables

    // physical parameters
    int N = getEnvInt("N", 1); // number of particles in the system
    double lp = getEnvDouble("LP", 2); // dimensionless persistence length
    double phi = getEnvDouble("PHI", 0.02); // packing fraction
    double g = getEnvDouble("TORQUE_PARAMETER", 0); // torque parameter

    Parameters parameters(N, lp, phi, dt, g); // class of simulation parameters

    // system
    System system(
      &parameters, seed, filename, nWork, dump, period); // define system
    simulate(&system);
  }
  else { // set parameters from file

    // input file parameters
    int inputFrame = getEnvInt("INPUT_FRAME", 0); // frame to copy as initial frame

    // system
    System system(
      inputFilename, inputFrame, dt, seed, filename, nWork, dump, period); // define system
    simulate(&system);
  }

}

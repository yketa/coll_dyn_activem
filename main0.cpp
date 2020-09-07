#include <algorithm>
#include <math.h>

#include "env.hpp"
#include "fire.hpp"
#include "iteration.hpp"
#include "maths.hpp"
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
  std::string filename = getEnvString("FILE", "out.dat0"); // output file name
  bool dump = getEnvBool("DUMP", 1); // dump positions and orientations to output file
  int period = getEnvInt("PERIOD", 1); // period of dumping of positions and orientations in number of frames

  // SYSTEM

  // simulation
  auto simulate = [&Niter] (System0* system) { // use of lambda function to enable conditional definition of system
    // INITIALISATION
    system->saveInitialState(); // save first frame
    // ITERATION
    #if AOUP // simulation of AOUPs
    iterate_AOUP_WCA(system, Niter); // run simulations
    #else // simulation of ABPs (default)
    iterate_ABP_WCA(system, Niter); // run simulations
    #endif
  };

  // definition
  std::string inputFilename = getEnvString("INPUT_FILENAME", ""); // input file from which to copy data
  if ( inputFilename == "" ) { // set parameters from environment variables

    // physical parameters
    int N = getEnvInt("N", 1); // number of particles in the system
    double Dr = getEnvDouble("DR", 1.0/2.0); // rotational diffusivity
    double epsilon = getEnvDouble("EPSILON", Dr/3.0); // coefficient parameter of potential
    double D = getEnvDouble("D", epsilon); // translational diffusivity
    double v0 = getEnvDouble("V0",
      #if AOUP // simulation of AOUPs
      sqrt(M_PI*D*Dr/2.0)
      #else // simulation of ABPs (default)
      1
      #endif
    ); // self-propulsion velocity
    double phi = getEnvDouble("PHI", 0.02); // packing fraction

    // diameters
    double I = getEnvDouble("I", 0); // polydispersity index
    std::vector<double> diameters (N, 1.0); // array of diameters
    if ( N > 1 ) {
      for (int i=0; i < N; i++) {
        diameters[i] = 1 - I + 2*I*i/(N - 1);
      }
    }
    // randomisation of diameters order
    Random randomGenerator(seed);
    std::random_shuffle(diameters.begin(), diameters.end(),
      [&randomGenerator](int max) { return randomGenerator.randomInt(max); });
    // system size
    double totalArea = 0.0;
    for (int i=0; i < N; i++) {
      totalArea += M_PI*pow(diameters[i], 2)/4.0;
    }
    double L = sqrt(totalArea/phi);

    Parameters parameters(N, epsilon, v0, D, Dr, phi, L, dt); // class of simulation parameters

    // system
    System0 system(
        &parameters, diameters, seed, filename, nWork, dump, period); // define system
    FIRE_WCA(&system, // FIRE minimisation algorithm
      getEnvDouble("EMIN", 1),
      getEnvInt("ITERMAX", (int) 100.0/dt),
      getEnvDouble("DTMIN", dt*1e-3),
      getEnvDouble("DT0", dt*1e-1),
      getEnvDouble("DTMAX", dt));
    simulate(&system);
  }
  else { // set parameters from file

    // input file parameters
    int inputFrame = getEnvInt("INPUT_FRAME", 0); // frame to copy as initial frame

    // system
    System0 system(
      inputFilename, inputFrame, dt, seed, filename, nWork, dump, period); // define system
    simulate(&system);
  }

}

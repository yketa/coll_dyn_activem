#include <algorithm>
#include <math.h>

#include "dat.hpp"
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
  int init = getEnvInt("INIT", 10000); // initialisation number of iterations
  int Niter = getEnvInt("NITER", 10000); // number of production iterations
  int dtMin = getEnvInt("LAGMIN", 1); // minimum lag time
  int dtMax = getEnvInt("LAGMAX", 100); // maximum lag time
  int nMax = getEnvInt("NMAX", 10); // maxium number of lag times
  int intMax = getEnvInt("INTMAX", 100); // maximum number of initial times

  // output
  std::string filename = getEnvString("FILE", "out.datN"); // output file name

  // SYSTEM

  // simulation
  auto simulate = [&init, &Niter] (SystemN* system) { // use of lambda function to enable conditional definition of system
    // INITIALISATION
    system->saveInitialState(); // save first frame
    // ITERATION
    #if AOUP // simulation of AOUPs
    iterate_AOUP_WCA<SystemN>(system, init + Niter); // run simulations
    #else // simulation of ABPs (default)
    iterate_ABP_WCA<SystemN>(system, init + Niter); // run simulations
    #endif
  };

  // definition
  std::vector<int> time0;
  std::vector<int> deltat;
  std::string inputFilename = getEnvString("INPUT_FILENAME", ""); // input file from which to copy data
  if ( inputFilename == "" ) { // set parameters from environment variables

    // physical parameters
    int N = getEnvInt("N", 1); // number of particles in the system
    double Dr = getEnvDouble("DR", 1.0/2.0); // rotational diffusivity
    double epsilon = getEnvDouble("EPSILON", Dr/3.0); // coefficient parameter of potential
    double D = getEnvDouble("D", epsilon); // translational diffusivity
    double v0 =
      #if AOUP // simulation of AOUPs
      sqrt(D*Dr)
      #else // simulation of ABPs (default)
      getEnvDouble("V0", 1)
      #endif
    ; // self-propulsion velocity
    double phi = getEnvDouble("PHI", 0.02); // packing fraction

    // diameters
    double I = getEnvDouble("I", 0); // polydispersity index
    std::vector<double> diameters = getDiametersI(N, I, seed); // diameters

    Parameters parameters(N, epsilon, v0, D, Dr, phi, diameters, dt); // class of simulation parameters

    // system
    SystemN system(
      init, Niter, dtMin, &dtMax, nMax, intMax,
        &time0, &deltat,
      &parameters, diameters, seed, filename); // define system
    FIRE_WCA<SystemN>(&system, // FIRE minimisation algorithm
      getEnvDouble("EMIN", 1),
      getEnvInt("ITERMAX", (int) 100.0/dt),
      getEnvDouble("DTMIN", dt*1e-3),
      getEnvDouble("DT0", dt*1e-1),
      getEnvDouble("DTMAX", dt));
    # if AOUP // system of AOUPs
    initPropulsionAOUP<SystemN>(&system); // set initial self-propulsion vectors
    #endif
    simulate(&system);
  }
  else { // set parameters from file

    // input file parameters
    DatN inputDat(inputFilename, false); // input file data object
    int inputFrame = getEnvInt("INPUT_FRAME", 0); // frame to copy as initial frame

    // physical parameters
    int N = inputDat.getNumberParticles(); // number of particles in the system
    double Dr = getEnvDouble("DR", inputDat.getRotDiffusivity()); // rotational diffusivity
    double epsilon = getEnvDouble("EPSILON", inputDat.getPotentialParameter()); // coefficient parameter of potential
    double D = getEnvDouble("D", inputDat.getTransDiffusivity()); // translational diffusivity
    double v0 =
      #if AOUP // simulation of AOUPs
      sqrt(D*Dr)
      #else // simulation of ABPs (default)
      getEnvDouble("V0", inputDat.getPropulsionVelocity())
      #endif
    ; // self-propulsion velocity
    double phi = getEnvDouble("PHI", inputDat.getPackingFraction()); // packing fraction

    // diameters
    double I = getEnvDouble("I", -1); // polydispersity index

    if ( I >= 0 ) {

      // change diameters
      std::vector<double> diameters = getDiametersI(N, I, seed);
      for (auto i=diameters.begin(); i!=diameters.end(); i++) std::cout << *i << std::endl;
      Parameters parameters(
        N, epsilon, v0, D, Dr, phi, diameters, dt); // class of simulation parameters

      // system
      SystemN system(
        init, Niter, dtMin, &dtMax, nMax, intMax,
          &time0, &deltat,
        inputFilename, inputFrame, &parameters,
          diameters,
        seed, filename); // define system
      simulate(&system);
    }
    else {

      // keep diameters
      std::vector<double> diameters = inputDat.getDiameters();
      Parameters parameters(
        N, epsilon, v0, D, Dr, phi, diameters, dt); // class of simulation parameters

      // system
      SystemN system(
        init, Niter, dtMin, &dtMax, nMax, intMax,
          &time0, &deltat,
        inputFilename, inputFrame, &parameters,
        seed, filename); // define system
      simulate(&system);
    }
  }

}

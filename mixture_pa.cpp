#include <math.h>
#include <numeric>
#include <assert.h>
#include <algorithm>

#include "alglib.hpp"
#include "env.hpp"
#include "maths.hpp"
#include "particle.hpp"
#include "readwrite.hpp"

double force_norm(double const& dist, double const& sigmaij) {
  // Returns norm of the interparticle force between particles at `distance'
  // with mean diameter `sigmaij'.

  // WCA force
  if ( dist/sigmaij >= pow(2., 1./6.) ) {
    return 0;
  }
  else {
    return (48.0/pow(dist/sigmaij, 13) - 24.0/pow(dist/sigmaij, 7));
  }
}

class IterationMixture {

  public:

    IterationMixture(
      SystemN* sys, CellList* cl, double const& timeStep0,
      double const& eps, std::vector<int> const& gr,
      std::vector<double> const& rotDiff, std::vector<double> const& transDiff,
      std::vector<double> const& wid, double const& hei) :
      dt0(timeStep0), system(sys), cellList(cl), epsilon(eps), groups(gr),
      Dr(rotDiff), D(transDiff),
      L(wid), Lxy{std::accumulate(L.begin(), L.end(), 0.0), hei} {
      /*
       *  [HEADER (see SystemN::SystemN)]
       *  | (int) N | (double) epsilon | (double) v0 | (double) D | (double) Dr | (double) lp | (double) phi | (double) L | (int) seed | (double) dt |
       *  || (int) NinitialTimes | (int) initialTimes[0] | ... | initialTimes[NinitialTimes - 1] ||
       *  || (int) NlagTimes | (int) lagTimes[0]  | ... | (int) lagTimes[NlagTimes - 1] ||
       *  || (int) Nframes | (int) frameIndices[0] | ... | (int) frameIndices[Nframes - 1] ||
       *  ||    PARTICLE 1     | ... |     PARTICLE N    ||
       *  || (double) diameter | ... | (double) diameter ||
       *  [SUPPLEMENTAL HEADER (see IterationMixture::IterationMixture)]
       *  | (double) Lx | (double) Ly |
       */

      // output information
      (system->getOutput())->write<double>(Lxy[0]); // box size in x-direction
      (system->getOutput())->write<double>(Lxy[1]); // box size in y-direction

      // save initial state
      system->saveInitialState();
    }

    void compute_forces(bool const& wall) {
      // Adds force computed from force_norm to `forcei' and `forcej'.

      double dist;
      double diff[2];
      double sigmaij;
      double norm;
      double dL;

      SystemN* systemPTR = system;
      const double* systemSize = &(Lxy[0]);
      const std::vector<int>* groupsPTR = &groups;
      const double* eps = &epsilon;

      cellList->listConstructor<double*>(system->getPositions());

      // interactions
      cellList->pairs(
        [&systemPTR, &systemSize, &groupsPTR, &eps, &wall, &dist, &diff,
          &sigmaij, &norm]
        (int const& index1, int const& index2) {
          if ( (!wall) || groupsPTR->at(index1) == groupsPTR->at(index2) ) { // if `wall' do not interacte between groups
            diff[0] = algDistPeriod(
              (systemPTR->getParticle(index1))-> position()[0],
              (systemPTR->getParticle(index2))-> position()[0],
              systemSize[0]);
            diff[1] = algDistPeriod(
              (systemPTR->getParticle(index1))-> position()[1],
              (systemPTR->getParticle(index2))-> position()[1],
              systemSize[1]);
            dist = sqrt(pow(diff[0], 2) + pow(diff[1], 2)); // distance between particles
            sigmaij =
              ((systemPTR->getParticle(index1))->diameter()
              + (systemPTR->getParticle(index2))->diameter())/2.0; // mean diameter
            norm = (*eps)*force_norm(dist, sigmaij);
            if ( norm > 0 ) {
              if ( norm > 1000 ) {
                std::cerr << index1 << "/" << index2 << "[" << dist << ", " << sigmaij << "]: " << norm << std::endl;
                std::cerr << index1 << ": " << (systemPTR->getParticle(index1))-> position()[0] << " " << (systemPTR->getParticle(index1))-> position()[1] << std::endl;
                std::cerr << index2 << ": " << (systemPTR->getParticle(index2))-> position()[0] << " " << (systemPTR->getParticle(index2))-> position()[1] << std::endl << std::endl;
              }
              for (int dim=0; dim < 2; dim++) {
                (systemPTR->getParticle(index1))->force()[dim] +=
                  -(diff[dim]/sigmaij/dist)*norm;
                (systemPTR->getParticle(index2))->force()[dim] -=
                  -(diff[dim]/sigmaij/dist)*norm;
              }
            }
          }
        });

      // walls
      if ( wall ) {
        dL = 0;
        for (double l : L) {
          for (int i=0; i < system->getNumberParticles(); i++) {
            diff[0] = algDistPeriod(
              (system->getParticle(i))->position()[0],
              l + dL,
              Lxy[0]);
            dist = abs(diff[0]);
            // finite size wall
            // sigmaij =
            //   ((system.getParticle(i + dN))->diameter() + maxDiameters[groups[i]])/2;
            // infinitesimal wall
            sigmaij = (system->getParticle(i))->diameter()/2;
            norm = epsilon*force_norm(dist, sigmaij);
            if ( norm > 0 ) {
              (system->getParticle(i))->force()[0] +=
                -(diff[0]/sigmaij/dist)*norm;
            }
          }
          dL += l;
        }
      }
    }

    void iterate(int const& Niter, bool const& wall) {
      // Perform iterations.

      std::vector<Particle> newParticles(0);
      for (int i=0; i < system->getNumberParticles(); i++) {
        newParticles.push_back(Particle((system->getParticle(i))->diameter()));
      }

      #if HEUN // HEUN'S SCHEME
      double selfPropulsionCorrection; // correction to the self-propulsion force
      std::vector<double> positions (2*system->getNumberParticles(), 0.0); // positions backup
      std::vector<double> forces (2*system->getNumberParticles(), 0.0); // forces backup
      #endif

      double timeStep = system->getTimeStep();
      if ( wall ) { timeStep = dt0; }

      for (int iter=0; iter < Niter; iter++) {

        // COMPUTATION
        for (int i=0; i < system->getNumberParticles(); i++) {

          for (int dim=0; dim < 2; dim++) {

            // POSITIONS
            // initialise velocity
            (system->getParticle(i))->velocity()[dim] = 0.0;
            // initialise new positions with previous ones
            newParticles[i].position()[dim] =
              (system->getParticle(i))->position()[dim];
            // add self-propulsion
            (system->getParticle(i))->velocity()[dim] +=
              (system->getParticle(i))->propulsion()[dim];
            newParticles[i].position()[dim] +=
              timeStep*(system->getParticle(i))->propulsion()[dim];
            // initialise force
            (system->getParticle(i))->force()[dim] = 0.0;

            // SELF-PROPULSION VECTORS
            if ( wall ) {
              // // Brownian diffusion
              // newParticles[i].propulsion()[dim] =
              //   sqrt(2*(*std::min_element(D.begin(), D.end()))*10
              //     /timeStep)
              //   *(system->getRandomGenerator())->gauss_cutoff();
              // initialise new self-propulsion vectors with previous ones
              newParticles[i].propulsion()[dim] =
                (system->getParticle(i))->propulsion()[dim];
              // add drift
              newParticles[i].propulsion()[dim] +=
                -timeStep*Dr0
                *(system->getParticle(i))->propulsion()[dim];
              // add diffusion
              newParticles[i].propulsion()[dim] +=
                sqrt(2.0*timeStep
                  *pow(Dr0, 2.0)
                  *D0)
                *(system->getRandomGenerator())->gauss_cutoff();
            }
            else {
              // initialise new self-propulsion vectors with previous ones
              newParticles[i].propulsion()[dim] =
                (system->getParticle(i))->propulsion()[dim];
              // add drift
              newParticles[i].propulsion()[dim] +=
                -timeStep*Dr[groups[i]]
                *(system->getParticle(i))->propulsion()[dim];
              // add diffusion
              newParticles[i].propulsion()[dim] +=
                sqrt(2.0*timeStep
                  *pow(Dr[groups[i]], 2.0)
                  *D[groups[i]])
                *(system->getRandomGenerator())->gauss_cutoff();
            }
          }
        }

        // FORCES
        compute_forces(wall); // compute forces

        for (int i=0; i < system->getNumberParticles(); i++) {
          for (int dim=0; dim < 2; dim++) {
            (system->getParticle(i))->velocity()[dim] +=
              (system->getParticle(i))->force()[dim]; // add force
            newParticles[i].position()[dim] +=
              (system->getParticle(i))->force()[dim]
              *timeStep; // add force displacement
          }
        }

        // HEUN'S SCHEME
        #if HEUN
        for (int i=0; i < system->getNumberParticles(); i++) {

          for (int dim=0; dim < 2; dim++) {
            // POSITIONS
            positions[2*i + dim] = (system->getParticle(i))->position()[dim]; // save initial position
            (system->getParticle(i))->position()[dim] =
              newParticles[i].position()[dim]; // integrate position as if using Euler's scheme
            // FORCES
            forces[2*i + dim] = (system->getParticle(i))->force()[dim]; // save computed force at initial position
            (system->getParticle(i))->force()[dim] = 0.0; // re-initialise force
          }
        }

        // FORCES
        compute_forces(wall); // re-compute forces

        for (int i=0; i < system->getNumberParticles(); i++) {

          // CORRECTION TO INTERPARTICLE FORCE
          for (int dim=0; dim < 2; dim++) {
            (system->getParticle(i))->velocity()[dim] +=
              ((system->getParticle(i))->force()[dim] - forces[2*i + dim])/2; // velocity
            newParticles[i].position()[dim] +=
              ((system->getParticle(i))->force()[dim] - forces[2*i + dim])/2
              *timeStep; // position
            (system->getParticle(i))->force()[dim] =
              ((system->getParticle(i))->force()[dim] + forces[2*i + dim])/2; // force
          }

          // CORRECTION TO SELF-PROPULSION FORCE
          for (int dim=0; dim < 2; dim++) {
            selfPropulsionCorrection =
              (newParticles[i].propulsion()[dim]
              - (system->getParticle(i))->propulsion()[dim])
              /2;
            (system->getParticle(i))->velocity()[dim] +=
              selfPropulsionCorrection; // velocity
            newParticles[i].position()[dim] +=
              timeStep*selfPropulsionCorrection; // position
            newParticles[i].propulsion()[dim] +=
              -timeStep*Dr[groups[i]]
              *selfPropulsionCorrection; // self-propulsion vector
          }

          // RESET INITIAL POSITIONS
          for (int dim=0; dim < 2; dim++) {
            (system->getParticle(i))->position()[dim] = positions[2*i + dim]; // position
          }
        }
        #endif

        // ORIENTATION
        for (int i=0; i < system->getNumberParticles(); i++) {
          newParticles[i].orientation()[0] = getAngleVector(
              newParticles[i].propulsion()[0], newParticles[i].propulsion()[1]);
        }

        // SAVE AND COPY
        this->saveNewState(newParticles);
      }
    }

    void saveNewState(std::vector<Particle>& newParticles) {
      // Saves new state of particles to output file then copy it.

      double wrap;
      int cross;

      // DUMP FRAME
      system->getDump()[0]++;

      ////////////
      // SAVING //
      ////////////

      double kineticEnergy = 0;
      for (int i=0; i < system->getNumberParticles(); i++) {

        // COMPUTATION

        for (int dim=0; dim < 2; dim++) {
          // KINETIC ENERGY
          kineticEnergy += pow((system->getParticle(i))->velocity()[dim], 2.0);
          // COORDINATES
          // compute crossings
          wrap = std::remquo(newParticles[i].position()[dim], Lxy[dim],
            &cross);
          if (wrap < 0) cross -= 1;
          (system->getParticle(i))->cross()[dim] += cross;
          // keep particles in the box
          newParticles[i].position()[dim] -= cross*Lxy[dim];
        }

        // DUMP

        // VELOCITIES
        if ( isInSortedVec<int>(system->getFrames(), system->getDump()[0] - 1)
          || system->getDump()[0] == 1 ) {
          for (int dim=0; dim < 2; dim++) {
            (system->getOutput())->write<double>(
              (system->getParticle(i))->velocity()[dim],
              (system->getVelocitiesDumps())->at(i) + dim*sizeof(double));
          }
        }

        if ( isInSortedVec<int>(system->getFrames(), system->getDump()[0]) ) {
          // WRAPPED POSITION
          for (int dim=0; dim < 2; dim++) {
            (system->getOutput())->write<double>(
              newParticles[i].position()[dim]);
          }
          // ORIENTATION
          (system->getOutput())->write<double>(
            newParticles[i].orientation()[0]);
          // VELOCITIES
          (system->getVelocitiesDumps())->at(i) =
            (system->getOutput())->tellp(); // location to dump velocities at next time step
          for (int dim=0; dim < 2; dim++) {
            (system->getOutput())->write<double>(0.0); // zero by default until rewrite at next time step
          }
          // SELF-PROPULSION VECTORS
          for (int dim=0; dim < 2; dim++) {
            (system->getOutput())->write<double>(
              newParticles[i].propulsion()[dim]);
          }
          // UNWRAPPED POSITION
          for (int dim=0; dim < 2; dim++) {
            (system->getOutput())->write<double>(
              newParticles[i].position()[dim]
              + (system->getParticle(i))->cross()[dim]*Lxy[dim]);
          }
        }

      }

      //////////////
      // CHECKING //
      //////////////

      if (
        kineticEnergy > 1000
          *system->getNumberParticles()
          *pow(system->getPropulsionVelocity(), 2.0) ) {
        std::cerr << pow(system->getPropulsionVelocity(), 2.0) << std::endl;
        system->flushOutputFile();
        throw std::invalid_argument("Exceeded kinetic energy limit. <v^2> = "
          + std::to_string(kineticEnergy/system->getNumberParticles()));
      }

      /////////////
      // COPYING //
      /////////////

      system->copyState(newParticles);
    }

    Particle* getParticle(int const& i) { return system->getParticle(i); } // returns pointer to particle
    const int getNumberParticles() { return system->getNumberParticles(); } // returns number of particles
    Random* getRandomGenerator() { return system->getRandomGenerator(); } // returns pointer to random number generator

    const double dt0; // integration time step for initialisation
    const double Dr0 = 1e2; // rotational diffusivity for initialisation
    const double D0 = 1; // translational diffusivity for initialisation

  private:

    SystemN* system; // system object
    CellList* cellList; // cell list
    double const epsilon; // interaction potential coefficient
    std::vector<int> const groups; // indices of the groups of particles
    std::vector<double> const Dr; // rotational diffusivities of groups
    std::vector<double> const D; // translational diffusivities of groups
    std::vector<double> const L; // widths between walls
    double const Lxy[2]; // height and width of the system

};

int main() {

  /////////////////////////
  // VARIABLE DEFINITION //
  /////////////////////////

  // random number generator
  int seed = getEnvInt("SEED", 1); // random seed

  // simulation parameters
  double dt0 = getEnvDouble("DT0", 5e-5); // initialisation time step
  double dt = getEnvDouble("DT", 1e-3); // time step
  int init = getEnvInt("INIT", 10000); // initialisation number of iterations
  int Niter = getEnvInt("NITER", 10000); // number of production iterations
  int dtMin = getEnvInt("LAGMIN", 1); // minimum lag time
  int dtMax = getEnvInt("LAGMAX", 100); // maximum lag time
  int nMax = getEnvInt("NMAX", 10); // maxium number of lag times
  int intMax = getEnvInt("INTMAX", 100); // maximum number of initial times
  std::vector<int> time0;
  std::vector<int> deltat;

  // output
  std::string filename = getEnvString("FILE", "out.datM"); // output file name
  Write output(filename + ".outM"); // specific mixture output

  // physical parameters
  std::vector<int> N = {getEnvInt("N1", 1), getEnvInt("N2", 1)}; // number of particles in each group
  int subgroups = N.size(); // number of subgroups
  int numberParticles = std::accumulate(N.begin(), N.end(), 0); // total number of particles
  std::vector<int> groups (0); // group indices per particle
  for (int k=0; k < subgroups; k++) {
    for (int i=0; i < N[k]; i++) {
      groups.push_back(k);
    }
  }
  std::vector<double> Dr = {getEnvDouble("DR1", 1), getEnvDouble("DR2", 1)}; // rotational diffusivity in each group
  std::vector<double> D = {getEnvDouble("D1", 1), getEnvDouble("D2", 1)}; // translational diffusivity in each group
  double epsilon = getEnvDouble("EPSILON", 1); // coefficient parameter of potential
  double v02 = 0; // squared self-propulsion velocity
  for (int k=0; k < subgroups; k++) {
    v02 += N[k]*Dr[k]*D[k]/numberParticles;
  }
  ////////
  // TEST
  v02 = 100;
  ////////
  std::vector<double> phi =
    {getEnvDouble("PHI1", 0.02), getEnvDouble("PHI2", 0.02)}; // packing fraction in each group
  bool conf = false; // confinement (rerence to circular confinement, ignore here...)

  // EXPERIMENTAL (N+N particles)
  std::string inputFilename = getEnvString("INPUT_FILENAME", ""); // input file

  // diameters
  std::vector<std::vector<double>> diameters (0); // diameters in each group
  std::vector<double> maxDiameters (0); // maximum diameter in each group
  std::vector<double> sigma (0); // diameters
  if ( inputFilename == "" ) {
    std::vector<double> I = {getEnvDouble("I1", 0), getEnvDouble("I2", 0)}; // polydispersity index
    for (int k=0; k < subgroups; k++) {
      diameters.push_back(
        getDiametersI(N[k], I[k], seed + k));
      maxDiameters.push_back(
        *std::max_element(diameters[k].begin(), diameters[k].end()));
      for (double d : diameters[k]) {
        sigma.push_back(d);
      }
    }
  }
  else {
    DatM inputDat(inputFilename, false); // input file data object
    std::vector<double> inputDiameters = inputDat.getDiameters();
    for (int k=0; k < 2; k++) {
      diameters.push_back(std::vector<double>(0));
      for (int i=0; i < numberParticles/2; i++) {
        diameters[k].push_back(
          inputDiameters[
            (i + k*numberParticles/2)%inputDat.getNumberParticles()]);
      }
      maxDiameters.push_back(
        *std::max_element(diameters[k].begin(), diameters[k].end()));
      for (double d : diameters[k]) {
        sigma.push_back(d);
      }
    }
  }

  // system size
  double ratioL = getEnvDouble("RATIOL", 1); // ratio of system size
  std::vector<double> L (0); // horizontal sizes of subdomains (separated by walls in the y direction)
  for (int k=0; k < 2; k++) { L.push_back(getL_WCA(phi[k], diameters[k])); }
  double L2 = std::inner_product(L.begin(), L.end(), L.begin(), 0); // sum of squared length
  for (int k=0; k < 2; k++) {
    L[k] = pow(L[k], 2)*sqrt(ratioL/L2);
    std::cerr << "L[" << k << "]=" << L[k] << std::endl;
  }
  double Lx = std::accumulate(L.begin(), L.end(), 0.0); // linear size along x-axis
  double Ly = Lx/ratioL; // linear size along y-axis

  // parameter, cell list, and system object
  Parameters parameters(
    numberParticles, epsilon, sqrt(v02), D[0], Dr[0], 0,
    std::max(phi[0], phi[1]), sigma, std::max(Lx, Ly), dt);  // class of simulation parameters
  CellList cellList(
    numberParticles, Lx,
    pow(2., 1./6.)*(*std::max_element(sigma.begin(), sigma.end())),
    Ly);
  SystemN system(
    init, Niter, dtMin, &dtMax, nMax, intMax,
      &time0, &deltat,
    &parameters, sigma, seed, filename,
    conf); // define system
  for (int k=0; k < subgroups; k++) {
    if ( L[k] < pow(2., 1./6.)*maxDiameters[k] ) {
      bool flag = true;
      for (int l=0; l < subgroups; l++) {
        if ( k != l
          && L[l] > pow(2., 1./6.)*(maxDiameters[k] + maxDiameters[l]) ) {
          L[k] += pow(2., 1./6.)*maxDiameters[k];
          L[l] -= pow(2., 1./6.)*maxDiameters[k];
          flag = false;
          break;
        }
      }
      if ( flag ) {
        throw std::invalid_argument("System cannot be initialised.");
      }
    }
  }

  ////////////////////
  // INITIALISATION //
  ////////////////////

  if ( inputFilename != "" ) { // set positions from input file
    // /!\ EXPERIMENTAL for systems of N+N particles

    // input file parameters
    DatM inputDat(inputFilename, false); // input file data object
    int inputFrame = getEnvInt("INPUT_FRAME", 0); // frame to copy as initial frame

    int ratioSystemSize = round(Lx/inputDat.getSystemSizes()[0]); // ratio from old to new system size
    assert
      (ratioSystemSize == round(Ly/inputDat.getSystemSizes()[1]));
    int nCopyCells = ratioSystemSize;
    // int nCopyCells = round(sqrt(ratioSystemSize));
    // if ( ratioSystemSize != nCopyCells*nCopyCells ) {
    //   // ratio of system sizes has to be a perfect square to copy correctly
    //   throw std::invalid_argument(
    //     "Ratio of system sizes is not a perfect square.");
    // }
    auto mapParticleIndex = // mapping from particle index to input particle index
      [&inputDat](int i){ return i%inputDat.getNumberParticles(); };
    auto copyCellIndex = // mapping from particle to index of copy cell
      [&inputDat](int i){ return i/inputDat.getNumberParticles(); };

    // set positions
    double pos;
    for (int i=0; i < system.getNumberParticles(); i++) {
      for (int dim=0; dim < 2; dim++) {
        pos = inputDat.getPosition(inputFrame, mapParticleIndex(i), dim);
        while (pos < 0)
          pos += inputDat.getSystemSizes()[dim];
        while (pos > inputDat.getSystemSizes()[dim])
          pos -= inputDat.getSystemSizes()[dim];
        (system.getParticle(i))->position()[dim] =
            pos
            + inputDat.getSystemSizes()[dim]
              *(dim == 0 ?
                copyCellIndex(i) / nCopyCells : copyCellIndex(i) % nCopyCells);
        (system.getParticle(i))->propulsion()[dim] =
          inputDat.getPropulsion(inputFrame, mapParticleIndex(i), dim);
      }
      (system.getParticle(i))->orientation()[0] =
          inputDat.getOrientation(inputFrame, mapParticleIndex(i));
    }

  }
  else { // set positions from minimisation of overlap

    double l;
    // ordered initial positions
    int gridNumbers[2];
    int dN = 0;
    double dL = 0;
    for (int k=0; k < subgroups; k++) {
      // finite size wall
      // dL += maxDiameters[k];
      // l = L[k] - 2*maxDiameters[k];
      // infinitesimal wall
      dL += pow(2., 1./6.)*maxDiameters[k]/2;
      l = L[k] - pow(2., 1./6.)*maxDiameters[k];
      gridNumbers[0] =
        ceil(l/sqrt(Ly*l/N[k]));
      gridNumbers[1] =
        ceil(Ly/sqrt(Ly*l/N[k]));
      for (int i=0; i < N[k]; i++) {
        (system.getParticle(i + dN))->position()[0] =
          dL
          + fmod(
            ((i%gridNumbers[0]) + 0.5*(i/gridNumbers[0]))*(l/gridNumbers[0]),
            l);
        (system.getParticle(i + dN))->position()[1] =
          (i/gridNumbers[0])*(Ly/gridNumbers[1]);
      }
      dN += N[k];
      // finite size wall
      // dL += L[k] - maxDiameters[k];
      // infinitesimal wall
      dL += L[k] - pow(2., 1./6.)*maxDiameters[k]/2;
    }

    // minimise overlap
    auto potential_force =
      [&system, &cellList, &groups, &L, &Lx, &Ly, &dL, &l, &maxDiameters]
      (double *r, double* U, double *gradU) {
      // positions
      for (int i=0; i < system.getNumberParticles(); i++) {
        for (int dim=0; dim < 2; dim++) {
          (system.getParticle(i))->position()[dim] = r[2*i + dim];
        }
      }
      cellList.listConstructor<double*>(system.getPositions());
      // initialisation
      U[0] = 0;
      for (int i=0; i < system.getNumberParticles(); i++) {
        for (int dim=0; dim < 2; dim++) {
          gradU[2*i + dim] = 0;
        }
      }
      double diff[2];
      double dist;
      double sigmaij;
      // interactions
      cellList.pairs(
        [&system, &Lx, &Ly, &r, &U, &gradU, &diff, &dist, &sigmaij, &groups]
        (int const& index1, int const& index2) { // do for each individual pair
        if ( groups[index1] == groups[index2] ) {
          // rescaled diameter
          sigmaij =
            pow(2., 1./6.)*(
              (system.getParticle(index1))->diameter()
              + (system.getParticle(index2))->diameter()
            )/2;
          // distance
          diff[0] = algDistPeriod(r[2*index1 + 0], r[2*index2 + 0], Lx);
          diff[1] = algDistPeriod(r[2*index1 + 1], r[2*index2 + 1], Ly);
          dist = sqrt(pow(diff[0], 2) + pow(diff[1], 2));
          // potential (harmonic)
          if ( dist/sigmaij < 1 ) {
            U[0] += pow(1 - dist/sigmaij, 2)/2;
            for (int dim=0; dim < 2; dim++) {
              gradU[2*index1 + dim] +=
                (diff[dim]/sigmaij/dist)*(1 - dist/sigmaij);
              gradU[2*index2 + dim] -=
                (diff[dim]/sigmaij/dist)*(1 - dist/sigmaij);
            }
          }
        }
      });
      // walls
      dL = 0;
      for (double l : L) {
        for (int i=0; i < system.getNumberParticles(); i++) {
          diff[0] = algDistPeriod(
            (system.getParticle(i))->position()[0],
            l + dL,
            Lx);
          dist = abs(diff[0]);
          // finite size wall
          // sigmaij =
          //   ((system.getParticle(i))->diameter() + maxDiameters[groups[k]])/2;
          // infinitesimal wall
          sigmaij = pow(2., 1./6.)*(system.getParticle(i))->diameter()/2;
          if ( dist/sigmaij < 1 ) {
            U[0] += pow(1 - dist/sigmaij, 2)/2;
            gradU[2*i] += (diff[0]/sigmaij/dist)*(1 - dist/sigmaij);
          }
        }
        dL += l;
      }
      // std::cerr << "[CG] U = " << U[0] << std::endl;
    };
    alglib::mincgreport report;
    CGMinimiser Uminimiser(potential_force, 2*numberParticles, 0, 0, 0, 0);
    std::vector<double> positions (0);
    for (int i=0; i < numberParticles; i++) {
      for (int dim=0; dim < 2; dim++) {
        positions.push_back((system.getParticle(i))->position()[dim]);
      }
    }
    report = Uminimiser.minimise(&positions[0]);
    std::cerr << "termination: " << report.terminationtype << ", iterations: " << report.iterationscount << std::endl;
    // int termination = report.terminationtype;
    // int iterations = report.iterationscount;

  }

  ////////////////
  // SIMULATION //
  ////////////////

  IterationMixture iteration(
    &system, &cellList, dt0, epsilon, groups, Dr, D, L, Ly);

  if (init > 0) {

    // random initial propulsions
    double stdev = sqrt(iteration.D0*iteration.Dr0);
    for (int i=0; i < iteration.getNumberParticles(); i++) {
      for (int dim=0; dim < 2; dim++) {
        (iteration.getParticle(i))->propulsion()[dim] =
          (iteration.getRandomGenerator())->gauss(0, stdev);
      }
      (iteration.getParticle(i))->orientation()[0] = getAngleVector(
        (iteration.getParticle(i))->propulsion()[0],
        (iteration.getParticle(i))->propulsion()[1]);
    }
    // iterate (initialisation)
    iteration.iterate(init, true);

    // random initial propulsions
    for (int i=0; i < iteration.getNumberParticles(); i++) {
      stdev = sqrt(D[groups[i]]*Dr[groups[i]]);
      for (int dim=0; dim < 2; dim++) {
        (iteration.getParticle(i))->propulsion()[dim] =
          (iteration.getRandomGenerator())->gauss(0, stdev);
      }
      (iteration.getParticle(i))->orientation()[0] = getAngleVector(
        (iteration.getParticle(i))->propulsion()[0],
        (iteration.getParticle(i))->propulsion()[1]);
    }
  }
  // iterate
  iteration.iterate(Niter, false);

}

#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include <vector>
#include <string>
#include <random>

#include "dat.hpp"
#include "maths.hpp"
#include "readwrite.hpp"

/////////////
// CLASSES //
/////////////

class Particle;
class CellList;
class Parameters;
class System;
class System0;
class SystemN;
class Rotors;


/*  PARTICLE
 *  --------
 *  Store diameter, positions and orientation of a given particle.
 */

class Particle {

  public:

    // CONSTRUCTORS

    Particle(double const& d = 1);
    Particle(
      double const& x, double const& y,
      double const& ang, double const& px, double const& py,
      double const& d = 1);

    // METHODS

    double* position(); // returns pointer to position
    long int* cross(); // returns pointer to number of boundary crossings
    double* orientation(); // returns pointer to orientation
    double* propulsion(); // returns pointer to self-propulsion direction
    double* velocity(); // returns pointer to velocity

    double diameter() const; // returns pointer to diameter

    double* force(); // returns pointer to force

    double* forcep(); // returns pointer to force applied on self-propulsion (AOUP)
    double* torque(); // returns pointer to aligning torque (ABP)

  private:

    // ATTRIBUTES

    double r[2]; // position (2D)
    long int c[2]; // number of boundary crossings (2D)
    double theta; // orientation
    double p[2]; // self-propulsion vector (2D)
    double v[2]; // velocity (2D)

    double const sigma; // diameter

    double f[2]; // force exerted on particle (2D)
    double fp[2]; // force applied on self-propulsion (AOUP)

    double gamma; // aligning torque (ABP)

};


/*  CELL LIST
 *  ---------
 *  Speed up computation by storing closest neighbours.
 */

class CellList {

  public:

    // CONSTRUCTORS

    CellList();

    // DESTRUCTORS

    ~CellList();

    // METHODS

    int getNumberBoxes(); // return number of boxes in each dimension
    std::vector<int>* getCell(int const& index); // return pointer to vector of indexes in cell

    template<class SystemClass> void initialise(
      SystemClass* system, double const& rcut) {
      // Initialise cell list.

      // parameters of cell list
      cutOff = rcut;
      numberBoxes = std::max((int) (system->getSystemSize()/cutOff), 1);
      sizeBox = system->getSystemSize()/numberBoxes;

      // set size of cell list
      for (int i=0; i < pow(numberBoxes, 2); i++) {
        cellList.push_back(std::vector<int>());
      }

      // set number of neighbours to explore
      if ( numberBoxes == 1 ) { dmin = 1; }
      else if ( numberBoxes == 2 ) { dmin = 0; }
      else { dmin = -1; }

      // put particles in the boxes
      update(system);
    }

    template<class SystemClass> void update(SystemClass* system) {
      // Put particles in the cell list.

      #ifdef USE_CELL_LIST // this is not useful when not using cell lists

      // flush old lists
      for (int i=0; i < (int) cellList.size(); i++) {
        cellList[i].clear();
      }

      // create new lists
      for (int i=0; i < system->getNumberParticles(); i++) {
        cellList[index(system->getParticle(i))].push_back(i); // particles are in increasing order of indexes
      }

      #endif
    }

    int index(Particle *particle);
      // Index of the box corresponding to a given particle.

    std::vector<int> getNeighbours(Particle *particle);
      // Returns vector of indexes of neighbouring particles.

  private:

    // ATTRIBUTES

    double cutOff; // cut-off radius of the interactions

    int numberBoxes; // number of boxes in each dimension
    double sizeBox; // size of each box
    int dmin; // trick to avoid putting too much neighbours when rcut is large

    std::vector<std::vector<int>> cellList; // cells with indexes of particles

};


/*  PARAMETERS
 *  ----------
 *  Store parameters relative to a system of active Brownian particles.
 *  All parameters are constant throughout all the algorithm.
 */

class Parameters {

  public:

    // CONSTRUCTORS

    Parameters();
    Parameters( // using custom dimensionless parameters relations
      int N, double lp, double phi, double dt, double g = 0);
    Parameters( // defining all parameters independently
      int N, double epsilon, double v0, double D, double Dr,
        double phi, std::vector<double> const& diameters,
      double dt);
    Parameters( // copy other class
      Parameters const& parameters);
    Parameters( // copy other class
      Parameters* parameters);

    Parameters( // copy .dat file
      Dat const& inputDat, double dt = 0) :
      Parameters(
        inputDat.getNumberParticles(),
        inputDat.getPersistenceLength(),
        inputDat.getPackingFraction(),
        dt > 0 ? dt : inputDat.getTimeStep(),
        inputDat.getTorqueParameter()) {}
    Parameters( // copy .dat0 file
      Dat0 const& inputDat, double dt = 0) :
      Parameters(
        inputDat.getNumberParticles(),
        inputDat.getPotentialParameter(),
        inputDat.getPropulsionVelocity(),
        inputDat.getTransDiffusivity(),
        inputDat.getRotDiffusivity(),
        inputDat.getPackingFraction(),
        inputDat.getDiameters(),
        dt > 0 ? dt : inputDat.getTimeStep()) {}
    Parameters( // copy .datN files
      DatN const& inputDat, double dt = 0) :
      Parameters(
        inputDat.getNumberParticles(),
        inputDat.getPotentialParameter(),
        inputDat.getPropulsionVelocity(),
        inputDat.getTransDiffusivity(),
        inputDat.getRotDiffusivity(),
        inputDat.getPackingFraction(),
        inputDat.getDiameters(),
        dt > 0 ? dt : inputDat.getTimeStep()) {}

    // METHODS

    int getNumberParticles() const; // returns number of particles in the system
    double getPotentialParameter() const; // returns coefficient parameter of potential
    double getPropulsionVelocity() const; // returns self-propulsion velocity
    double getTransDiffusivity() const; // returns translational diffusivity
    double getRotDiffusivity() const; // returns rotational diffusivity
    double getPersistenceLength() const; // returns persistence length
    double getPackingFraction() const; // returns packing fraction
    double getSystemSize() const; // returns system size
    double getTorqueParameter() const; // returns torque parameter
    double getTimeStep() const; // returns time step

  private:

    // ATTRIBUTES

    int const numberParticles; // number of particles in the system
    double const potentialParameter; // coefficient parameter of potential
    double const propulsionVelocity; // self-propulsion velocity
    double const transDiffusivity; // translational diffusivity
    double const rotDiffusivity; // rotational diffusivity
    double const persistenceLength; // persistence length
    double const packingFraction; // packing fraction
    double const systemSize; // system size
    double const torqueParameter; // torque parameter
    double const timeStep; // time step

};


/*  SYSTEM
 *  ------
 *  Store physical and integration parameter.
 *  Access to distance and potentials.
 *  Save system state to output file.
 *  Using custom dimensionless parameters relations.
 */

class System {
  /*  Contains all the details to simulate a system of active Brownian
   *  particles, with dimensionless parameters taken from Nemoto et al., PRE 99
   *  022605 (2019).
   *  (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#Active%20Brownian%20particles)
   *
   *  Parameters are stored in a binary file with the following structure:
   *
   *  [HEADER (see System::System)]
   *  | (int) N | (double) lp | (double) phi | (double) L | (double) g | (int) seed | (double) dt | (int) framesWork | (bool) dump | (int) period |
   *
   *  [INITIAL FRAME (see System::saveInitialState)] (all double)
   *  ||                    FRAME 0                     ||
   *  ||          PARTICLE 1         | ... | PARTICLE N ||
   *  ||   R   | ORIENTATION |   V   | ... |     ...    ||
   *  || X | Y |    theta    | 0 | 0 | ... |     ...    ||
   *
   *  [BODY (see System::saveNewState)] (all double)
   *  ||                    FRAME 1 + i*period                  || ... || FRAME 1 + (i + framesWork - 1)*period |~
   *  ||              PARTICLE 1             | ... | PARTICLE N || ... ||                  ...                  |~
   *  ||   R   | ORIENTATION |       V       | ... |     ...    || ... ||                  ...                  |~
   *  || X | Y |    theta    |  V_X  |  V_Y  | ... |     ...    || ... ||                  ...                  |~
   *
   *  ~|                                                                                                                                 || ...
   *  ~|                                                                                                                                 || ...
   *  ~| ACTIVE WORK | ACTIVE WORK (FORCE) | ACTIVE WORK (ORIENTATION) |   ORDER PARAMETER   | 1st TORQUE INTEGRAL | 2nd TORQUE INTEGRAL || ...
   *  ~|      W      |          Wp         |             Wo            | norm nu | nuX | nuY |          I1         |          I2         || ...
   */

  public:

    // CONSTRUCTORS

    System();
    System(
      Parameters* parameters, int seed = 0, std::string filename = "",
      int nWork = 1, bool dump = true, int period = 1);
    System(
      System* system, int seed = 0, std::string filename = "",
      int nWork = 1, bool dump = true, int period = 1);
    System(
      std::string inputFilename, int inputFrame = 0, double dt = 0,
      int seed = 0, std::string filename = "",
      int nWork = 1, bool dump = true, int period = 1);

    System(
      int N, double lp, double phi, double g, double dt,
      int seed = 0, std::string filename = "",
      int nWork = 1, bool dump = true, int period = 1) :
      System(
        [&N, &lp, &phi, &g, &dt]{ return new Parameters(N, lp, phi, dt, g); }(),
        seed, filename, nWork, dump, period) {;}

    // cloning constructors
    System(System* dummy, int seed, int tau, std::string filename = "") :
      System(dummy, seed, filename, 1, filename != "", tau)
      { resetDump(); }
    System(int N, double lp, double phi, double g, double dt,
      int seed, int tau, std::string filename = "") :
      System(N, lp, phi, g, dt, seed, filename, 1, filename != "", tau)
      { resetDump(); }

    // DESTRUCTORS

    ~System();

    // METHODS

    Parameters* getParameters(); // returns pointer to class of parameters

    int getNumberParticles() const; // returns number of particles
    double getPersistenceLength() const; // returns dimensionless persistence length
    double getPackingFraction() const; // returns packing fraction
    double getSystemSize() const; // returns system size
    double getTimeStep() const; // returns time step

    int getRandomSeed() const; // returns random seed
    Random* getRandomGenerator(); // returns pointer to random generator
    void setGenerator(std::default_random_engine rndeng); // return random generator

    Particle* getParticle(int const& index); // returns pointer to given particle
    std::vector<Particle> getParticles(); // returns vector of particles

    CellList* getCellList(); // returns pointer to CellList object

    void flushOutputFile(); // flush output file
    std::string getOutputFile() const; // returns output file name

    void setTorqueParameter(double& g); // set new torque parameter
    double getTorqueParameter(); // returns torque parameter
    // NOTE: These functions modify and access the torque parameter which is
    //       effectively used in the computation and is an attribute of this
    //       class and not of `param'.

    double getBiasingParameter(); // returns biasing parameter [cloning algorithm]
    void setBiasingParameter(double s); // set biasing parameter [cloning algorithm]

    int* getDump(); // returns number of frames dumped since last reset
    void resetDump();
      // Reset time-extensive quantities over trajectory.
    void copyDump(System* system);
      // Copy dumps from other system.
      // WARNING: This also copies the index of last frame dumped. Consistency
      //          has to be checked.

    double getWork(); // returns last computed normalised rate of active work
    double getWorkForce(); // returns last computed force part of the normalised rate of active work
    double getWorkOrientation(); // returns last computed orientation part of the normalised rate of active work
    double getOrder(); // returns last computed averaged integrated order parameter
    double getOrder0(); // returns last computed averaged integrated order parameter along x-axis
    double getOrder1(); // returns last computed averaged integrated order parameter along y-axis
    double getTorqueIntegral1(); // returns last computed averaged first torque integral
    double getTorqueIntegral2(); // returns last computed averaged second torque integral
    // NOTE: All these quantities are computed every framesWork*dumpPeriod iterations.

    double* getTotalWork(); // returns computed active work since last reset
    double* getTotalWorkForce(); // returns computed force part of the active work since last rest
    double* getTotalWorkOrientation(); // returns computed orientation part of the active work since last reset
    double* getTotalOrder(); // returns computed integrated order parameter since last reset
    double* getTotalOrder0(); // returns computed integrated order parameter along x-axis since last reset
    double* getTotalOrder1(); // returns computed integrated order parameter along y-axis since last reset
    double* getTotalTorqueIntegral1(); // returns computed first torque integral since last reset
    double* getTotalTorqueIntegral2(); // returns computed second torque integral since last reset
    // NOTE: All these quantities are updated every framesWork*dumpPeriod iterations.
    //       All these quantities are extensive in time since last reset.

    void copyState(std::vector<Particle>& newParticles);
      // Copy positions and orientations.
    void copyState(System* system);
      // Copy positions and orientations.

    void saveInitialState();
      // Saves initial state of particles to output file.
    void saveNewState(std::vector<Particle>& newParticles);
      // Saves new state of particles to output file then copy it.

  private:

    // ATTRIBUTES

    Parameters param; // class of simulation parameters

    int const randomSeed; // random seed
    Random randomGenerator; // random number generator

    std::vector<Particle> particles; // vector of particles

    CellList cellList; // cell list

    Write output; // output class
    std::vector<long int> velocitiesDumps; // locations in output file to dump velocities

    int const framesWork; // number of frames on which to sum the active work before dumping
    bool const dumpParticles; // dump positions and orientations to output file
    int const dumpPeriod; // period of dumping of positions and orientations in number of frames

    double torqueParameter; // aligning torque parameter

    double biasingParameter; // biasing parameter [cloning algorithm]

    int dumpFrame; // number of frames dumped since last reset
    // Quantities
    // (0): sum of quantity since last dump
    // (1): normalised quantity over last dump period
    // (2): time-extensive quantity over trajectory since last reset
    double workSum[3]; // active work
    double workForceSum[3]; //force part of the active work
    double workOrientationSum[3]; // orientation part of the active work
    double orderSum[3]; // integrated order parameter norm (in units of the time step)
    double order0Sum[3]; // integrated order parameter along x-axis (in units of the time step)
    double order1Sum[3]; // integrated order parameter along y-axis (in units of the time step)
    double torqueIntegral1[3]; // first torque integral
    double torqueIntegral2[3]; // second torque integral

};


/*  SYSTEM0
 *  -------
 *  Store physical and integration parameter.
 *  Access to distance and potentials.
 *  Save system state to output file.
 *  Using all free parameters.
 */

class System0 {
  /*  Contains all the details to simulate a system of active particles.
   *  (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#Active%20Brownian%20particles)
   *  (see https://yketa.github.io/PhD_Wiki/#Active%20Ornstein-Uhlenbeck%20particles)
   *
   *  Parameters are stored in a binary file with the following structure:
   *
   *  [HEADER (see System::System)]
   *  | (int) N | (double) epsilon | (double) v0 | (double) D | (double) Dr | (double) lp | (double) phi | (double) L | (int) seed | (double) dt | (int) framesWork | (bool) dump | (int) period |
   *  ||    PARTICLE 1     | ... |     PARTICLE N    ||
   *  || (double) diameter | ... | (double) diameter ||
   *
   *  [INITIAL FRAME (see System0::saveInitialState)] (all double)
   *  ||                                 FRAME 0                                 ||
   *  ||                      PARTICLE 1                      | ... | PARTICLE N ||
   *  ||   R   | ORIENTATION |   V   |     P     | UNFOLDED R | ... |     ...    ||
   *  || X | Y |    theta    | 0 | 0 | P_X | P_Y |   X  |  Y  | ... |     ...    ||
   *
   *  [BODY (see System0::saveNewState)] (all double)
   *  ||                               FRAME 1 + i*period                                || ... || FRAME 1 + (i + framesWork - 1)*period |~
   *  ||                          PARTICLE 1                          | ... | PARTICLE N || ... ||                  ...                  |~
   *  ||   R   | ORIENTATION |       V       |     P     | UNFOLDED R | ... |     ...    || ... ||                  ...                  |~
   *  || X | Y |    theta    |  V_X  |  V_Y  | P_X | P_Y |   X  |  Y  | ... |     ...    || ... ||                  ...                  |~
   *
   *  ~|                                                                                 || ...
   *  ~|                                                                                 || ...
   *  ~| ACTIVE WORK | ACTIVE WORK (FORCE) | ACTIVE WORK (ORIENTATION) | ORDER PARAMETER || ...
   *  ~|      W      |          Wp         |             Wo            |        nu       || ...
   */

  public:

    // CONSTRUCTORS

    System0();
    System0(
      Parameters* parameters, int seed = 0, std::string filename = "",
      int nWork = 1, bool dump = true, int period = 1);
    System0(
      Parameters* parameters, std::vector<double>& diameters, int seed = 0,
      std::string filename = "", int nWork = 1, bool dump = true,
      int period = 1);
    System0(
      System0* system, int seed = 0, std::string filename = "",
      int nWork = 1, bool dump = true, int period = 1);
    System0(
      System0* system, std::vector<double>& diameters, int seed = 0,
      std::string filename = "", int nWork = 1, bool dump = true,
      int period = 1);
    System0(
      std::string inputFilename, int inputFrame = 0, double dt = 0,
      int seed = 0, std::string filename = "",
      int nWork = 1, bool dump = true, int period = 1);

    // DESTRUCTORS

    ~System0();

    // METHODS

    Parameters* getParameters(); // returns pointer to class of parameters
    std::vector<double> getDiameters() const; // returns vector of diameters

    int getNumberParticles() const; // returns number of particles
    double getPotentialParameter() const; // returns coefficient parameter of potential
    double getPropulsionVelocity() const; // returns self-propulsion velocity
    double getTransDiffusivity() const; // returns translational diffusivity
    double getRotDiffusivity() const; // returns rotational diffusivity
    double getPersistenceLength() const; // returns persistence length
    double getPackingFraction() const; // returns packing fraction
    double getSystemSize() const; // returns system size
    double getTimeStep() const; // returns time step

    int getRandomSeed() const; // returns random seed
    Random* getRandomGenerator(); // returns pointer to random generator

    Particle* getParticle(int const& index); // returns pointer to given particle
    std::vector<Particle> getParticles(); // returns vector of particles

    CellList* getCellList(); // returns pointer to CellList object

    std::string getOutputFile() const; // returns output file name

    int* getDump(); // returns number of frames dumped since last reset
    void resetDump();
      // Reset time-extensive quantities over trajectory.
    void copyDump(System0* system);
      // Copy dumps from other system.
      // WARNING: This also copies the index of last frame dumped. Consistency
      //          has to be checked.

    double getWork(); // returns last computed normalised rate of active work
    double getWorkForce(); // returns last computed force part of the normalised rate of active work
    double getWorkOrientation(); // returns last computed orientation part of the normalised rate of active work
    double getOrder(); // returns last computed averaged integrated order parameter
    // NOTE: All these quantities are computed every framesWork*dumpPeriod iterations.

    double* getTotalWork(); // returns computed active work since last reset
    double* getTotalWorkForce(); // returns computed force part of the active work since last rest
    double* getTotalWorkOrientation(); // returns computed orientation part of the active work since last reset
    double* getTotalOrder(); // returns computed integrated order parameter since last reset
    // NOTE: All these quantities are updated every framesWork*dumpPeriod iterations.
    //       All these quantities are extensive in time since last reset.

    void copyState(std::vector<Particle>& newParticles);
      // Copy positions and orientations.
    void copyState(System0* system);
      // Copy positions and orientations.

    void saveInitialState();
      // Saves initial state of particles to output file.
    void saveNewState(std::vector<Particle>& newParticles);
      // Saves new state of particles to output file then copy it.

  private:

    // ATTRIBUTES

    Parameters param; // class of simulation parameters

    int const randomSeed; // random seed
    Random randomGenerator; // random number generator

    std::vector<Particle> particles; // vector of particles

    CellList cellList; // cell list

    Write output; // output class
    std::vector<long int> velocitiesDumps; // locations in output file to dump velocities

    int const framesWork; // number of frames on which to sum the active work before dumping
    bool const dumpParticles; // dump positions and orientations to output file
    int const dumpPeriod; // period of dumping of positions and orientations in number of frames

    int dumpFrame; // number of frames dumped since last reset
    // Quantities
    // (0): sum of quantity since last dump
    // (1): normalised quantity over last dump period
    // (2): time-extensive quantity over trajectory since last reset
    double workSum[3]; // active work
    double workForceSum[3]; //force part of the active work
    double workOrientationSum[3]; // orientation part of the active work
    double orderSum[3]; // integrated order parameter norm (in units of the time step)

};


/*  SYSTEMN
 *  -------
 *  Store physical and integration parameter.
 *  Access to distance and potentials.
 *  Save system state to output file ONLY for pre-defined frames.
 *  Using all free parameters.
 */

class SystemN {
  /*  Contains all the details to simulate a system of active particles.
   *  (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#Active%20Brownian%20particles)
   *  (see https://yketa.github.io/PhD_Wiki/#Active%20Ornstein-Uhlenbeck%20particles)
   *
   *  Parameters are stored in a binary file with the following structure:
   *
   *  [HEADER (see System::System)]
   *  | (int) N | (double) epsilon | (double) v0 | (double) D | (double) Dr | (double) lp | (double) phi | (double) L | (int) seed | (double) dt | (int) init | (int) NLin | (int) NiterLin | (int) NLog | (int) nFrames |~
   *  ~|| (int) frames[0] | ... | (int) frames[nFrames - 1] ||~~
   *  ~~||    PARTICLE 1     | ... |     PARTICLE N    ||
   *  ~~|| (double) diameter | ... | (double) diameter ||
   *
   *  [FRAMES (see SystemN::saveInitialState & SystemN::saveNewState)] (all double)
   *  ||                                 FRAME I                                 ||
   *  ||                      PARTICLE 1                      | ... | PARTICLE N ||
   *  ||   R   | ORIENTATION |   V   |     P     | UNFOLDED R | ... |     ...    ||
   *  || X | Y |    theta    | 0 | 0 | P_X | P_Y |   X  |  Y  | ... |     ...    ||
   */

  public:

    // CONSTRUCTORS

    SystemN();
    SystemN(
      int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
        std::vector<int>* time0, std::vector<int>* deltat,
      Parameters* parameters, int seed = 0, std::string filename = "");
    SystemN(
      int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
        std::vector<int>* time0, std::vector<int>* deltat,
      Parameters* parameters, std::vector<double>& diameters, int seed = 0,
      std::string filename = "");
    SystemN(
      int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
        std::vector<int>* time0, std::vector<int>* deltat,
      SystemN* system, int seed = 0, std::string filename = "");
    SystemN(
      int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
        std::vector<int>* time0, std::vector<int>* deltat,
      SystemN* system, std::vector<double>& diameters, int seed = 0,
      std::string filename = "");
    SystemN(
      int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
        std::vector<int>* time0, std::vector<int>* deltat,
      std::string inputFilename, int inputFrame = 0, double dt = 0,
      int seed = 0, std::string filename = "");
    SystemN(
      int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
        std::vector<int>* time0, std::vector<int>* deltat,
      std::string inputFilename, int inputFrame = 0, Parameters* parameters = 0,
      int seed = 0, std::string filename = "");

    // DESTRUCTORS

    ~SystemN();

    // METHODS

    std::vector<int> const* getFrames(); // returns pointer to vector of indices of frames to save

    Parameters* getParameters(); // returns pointer to class of parameters
    std::vector<double> getDiameters() const; // returns vector of diameters

    int getNumberParticles() const; // returns number of particles
    double getPotentialParameter() const; // returns coefficient parameter of potential
    double getPropulsionVelocity() const; // returns self-propulsion velocity
    double getTransDiffusivity() const; // returns translational diffusivity
    double getRotDiffusivity() const; // returns rotational diffusivity
    double getPersistenceLength() const; // returns persistence length
    double getPackingFraction() const; // returns packing fraction
    double getSystemSize() const; // returns system size
    double getTimeStep() const; // returns time step

    int getRandomSeed() const; // returns random seed
    Random* getRandomGenerator(); // returns pointer to random generator

    Particle* getParticle(int const& index); // returns pointer to given particle
    std::vector<Particle> getParticles(); // returns vector of particles

    CellList* getCellList(); // returns pointer to CellList object

    std::string getOutputFile() const; // returns output file name

    int* getDump(); // returns number of frames dumped since last reset
    void resetDump();
      // Reset time-extensive quantities over trajectory.
    void copyDump(SystemN* system);
      // Copy dumps from other system.
      // WARNING: This also copies the index of last frame dumped. Consistency
      //          has to be checked.

    void copyState(std::vector<Particle>& newParticles);
      // Copy positions and orientations.
    void copyState(SystemN* system);
      // Copy positions and orientations.

    void saveInitialState();
      // Saves initial state of particles to output file.
    void saveNewState(std::vector<Particle>& newParticles);
      // Saves new state of particles to output file then copy it.

  private:

    // ATTRIBUTES

    std::vector<int> const frameIndices; // indices of frames to save
    // NOTE: This vector is sorted and the element 0 is removed.
    // NOTE: Frame 0 is ALWAYS saved first.

    Parameters param; // class of simulation parameters

    int const randomSeed; // random seed
    Random randomGenerator; // random number generator

    std::vector<Particle> particles; // vector of particles

    CellList cellList; // cell list

    Write output; // output class
    std::vector<long int> velocitiesDumps; // locations in output file to dump velocities

    int dumpFrame; // number of frames dumped since last reset

};


/*  ROTORS
 *  ------
 *  Store physical and integration parameter.
 *  Save system state to output file.
 */

class Rotors {
  /*  Contains all the details to simulate a system of interacting Brownian
   *  rotors.
   *  (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#N-interacting%20Brownian%20rotors)
   *
   *  Parameters are stored in a binary file with the following structure:
   *
   *  [HEADER (see Rotors::Rotors)]
   *  | (int) N | (double) Dr | (double) g | (double) dt | (int) framesOrder | (bool) dump | (int) period | (int) seed |
   *
   *  [INITIAL FRAME (see Rotors::saveInitialState)] (all double)
   *  ||         FRAME 0         ||
   *  || ROTOR 1 | ... | ROTOR N ||
   *  ||  theta  | ... |  theta  ||
   *
   *  [BODY (see Rotors::saveNewState)] (all double)
   *  ||      FRAME 1 + i*period     || ... || FRAME 1 + (i + framesOrder - 1)*period |~
   *  ||  ROTOR 1  | ... |  ROTOR N  || ... ||                  ...                   |~
   *  ||   theta   | ... |   theta   || ... ||                  ...                   |~
   *
   *  ~|                 |                         || ...
   *  ~| ORDER PARAMETER | SQUARED ORDER PARAMETER || ...
   *  ~|        nu       |            nu2          || ...
   */

  public:

    // CONSTRUCTORS

    Rotors();
    Rotors(
      int N, double Dr, double dt, int seed = 0, double g = 0,
      std::string filename = "", int nOrder = 1, bool dump = true,
      int period = 1);
    Rotors(
      Rotors* rotors, int seed = 0, std::string filename = "", int nOrder = 1,
      bool dump = true, int period = 1);

    // cloning constructor
    Rotors(Rotors* dummy, int seed, int tau, std::string filename = "") :
      Rotors(dummy, seed, "", tau, false, 1) { resetDump(); }

    // DESTRUCTORS

    ~Rotors();

    // METHODS

    int getNumberParticles() const; // returns number of rotors
    double getRotDiffusivity() const; // returns rotational diffusivity

    double getTorqueParameter(); // returns aligning torque parameter
    void setTorqueParameter(double g); // sets aligning torque parameter

    double getTimeStep() const; // returns simulation time step

    double* getOrientation(int const& index); // returns pointer to orientation of given rotor
    double* getTorque(int const& index); // returns pointer to torque applied on a given rotor

    Random* getRandomGenerator(); // returns pointer to random generator

    double getBiasingParameter(); // returns biasing parameter [cloning algorithm]
    void setBiasingParameter(double s); // set biasing parameter [cloning algorithm]

    int* getDump(); // returns number of frames dumped since last reset
    void resetDump();
      // Reset time-extensive quantities over trajectory.
    void copyDump(Rotors* rotors);
      // Copy dumps from other system.
      // WARNING: This also copies the index of last frame dumped. Consistency
      //          has to be checked.

    double getOrder(); // returns last computed averaged integrated order parameter
    double getOrderSq(); // returns last computed averaged integrated squared order parameter
    // NOTE: All these quantities are computed every framesOrder*dumpPeriod iterations.

    double* getTotalOrder(); // returns computed integrated order parameter since last reset
    double* getTotalOrderSq(); // returns computed integrated squared order parameter since last reset
    // NOTE: All these quantities are updated every framesOrder*dumpPeriod iterations.
    //       All these quantities are extensive in time since last reset.

    #if BIAS == 1
    #ifdef CONTROLLED_DYNAMICS
    double getBiasIntegral(); // returns computed bias integral over last dump period for controlled dynamics when biasing by squared polarisation
    // NOTE: This quantity is updated every framesOrder*dumpPeriod iterations.
    #endif
    #endif

    void copyState(std::vector<double>& newOrientations);
      // Copy orientations.
    void copyState(Rotors* rotors);
      // Copy orientations.

    void saveInitialState();
      // Saves initial state of rotors to output file.
    void saveNewState(std::vector<double>& newOrientations);
      // Saves new state of rotors to output file then copy it.

  private:

    // ATTRIBUTES

    int const numberParticles; // number of particles
    double const rotDiffusivity; // rotational diffusivity

    double torqueParameter; // aligning torque parameter

    double const timeStep; // simulation time step

    int const framesOrder; // number of frames on which to average the order parameter before dumping
    bool const dumpRotors; // dump orientations to output file
    int const dumpPeriod; // period of dumping of orientations in number of frames

    int const randomSeed; // random seed
    Random randomGenerator; // random number generator

    std::vector<double> orientations; // vector of orientations
    std::vector<double> torques; // vector of torques

    Write output; // output class

    double biasingParameter; // biasing parameter [cloning algorithm]

    int dumpFrame; // number of frames dumped since last reset
    // Quantities
    // (0): sum of quantity since last dump
    // (1): normalised quantity over last dump period
    // (2): time-extensive quantity over trajectory since last reset
    double orderSum[3]; // integrated order parameter norm (in units of the time step)
    double orderSumSq[3]; // integrated squared order parameter norm (in units of the time step)

    #if BIAS == 1
    #ifdef CONTROLLED_DYNAMICS
    double biasIntegral[2]; // bias for controlled dynamics when biasing by squared polarisation
    // (0): sum of quantity since last dump
    // (1): time-extensive quantity over last dump period
    #endif
    #endif

};


////////////////
// PROTOTYPES //
////////////////

double getL_WCA(double phi, std::vector<double> const& diameters);
  // Returns the length of a square system with packing fraction `phi'
  // containing particles with `diameters', considering the actual diameter
  // as the WCA diameter of interaction.

double getL_WCA(double phi, int N, double diameter = 1);
  // Returns the length of a square system with packing fraction `phi'
  // containing  `N' particles with same `diameter', considering the actual
  // diameter as the WCA diameter of interaction.

std::vector<double> getOrderParameter(std::vector<Particle>& particles);
  // Returns order parameter.

std::vector<double> getOrderParameter(std::vector<double>& orientations);
  // Returns order parameter.

double getOrderParameterNorm(std::vector<Particle>& particles);
  // Returns order parameter norm.

double getOrderParameterNorm(std::vector<double>& orientations);
  // Returns order parameter norm.

double getGlobalPhase(std::vector<Particle>& particles);
  // Returns global phase.

double getGlobalPhase(std::vector<double>& orientations);
  // Returns global phase.

std::vector<int> getLogFrames(
  int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
  std::vector<int>* time0, std::vector<int>* dt);
  // Returns vector of logarithmically spaced frames in overlapping linearly
  // spaced blocks.
  // Saves in `time0' the initial frames and in `dt' the lag times.
  // NOTE: `dtMax' may be modified according to other parameters.


///////////////
// FUNCTIONS //
///////////////

template<class SystemClass, typename F> void pairs_system(
  SystemClass* system, F function) {
  // Given a function `function` with parameters (int index1, int index2),
  // call this function with every unique pair of interacting particles, using
  // cell list if USE_CELL_LIST is defined or a double loop

  #ifdef USE_CELL_LIST // with cell list

  int index1, index2; // index of the couple of particles
  int i, j; // indexes of the cells
  int k, l; // indexes of the particles in the cell
  std::vector<int>* cell1;
  std::vector<int>* cell2;
  int numberBoxes = (system->getCellList())->getNumberBoxes();
  for (i=0; i < pow(numberBoxes, 2); i++) { // loop over cells

    cell1 = (system->getCellList())->getCell(i); // indexes of particles in the first cell
    for (k=0; k < (int) cell1->size(); k++) { // loop over particles in the first cell
      index1 = cell1->at(k);

      // interactions with particles in the same cell
      for (l=k+1; l < (int) cell1->size(); l++) { // loop over particles in the first cell
        index2 = cell1->at(l);
        function(index1, index2);
      }

      if ( numberBoxes == 1 ) { continue; } // only one cell

      // interactions with particles in other cells
      if ( numberBoxes == 2 ) { // 2 x 2 cells

        for (j=0; j < 4; j++) {
          if ( i == j ) { continue; } // same cell
          cell2 = (system->getCellList())->getCell(j); // indexes of particles in the second cell

          for (l=0; l < (int) cell2->size(); l++) { // loop over particles in the second cell
            index2 = cell2->at(l);
            if ( index1 < index2 ) { // only count once each couple
              function(index1, index2);
            }
          }
        }
      }
      else { // 3 x 3 cells or more

        int x = i%numberBoxes;
        int y = i/numberBoxes;
        for (int dx=0; dx <= 1; dx++) {
          for (int dy=-1; dy < 2*dx; dy++) { // these two loops correspond to (dx, dy) = {0, -1}, {1, -1}, {1, 0}, {1, 1}, so that half of the neighbouring cells are explored
            j = (numberBoxes + (x + dx))%numberBoxes
              + numberBoxes*((numberBoxes + (y + dy))%numberBoxes); // index of neighbouring cell
            cell2 = (system->getCellList())->getCell(j); // indexes of particles in the second cell

            for (l=0; l < (int) cell2->size(); l++) { // loop over particles in the second cell
              index2 = cell2->at(l);
              function(index1, index2);
            }
          }
        }
      }
    }
  }

  #else // with double loop

  for (int index1=0; index1 < system->getNumberParticles(); index1++) {
    for (int index2=index1+1; index2 < system->getNumberParticles(); index2++) {
      function(index1, index2);
    }
  }

  #endif
}

template<class SystemClass> int wrapCoordinate(
  SystemClass* system, double const& x) {
  // Returns the algebraic number of times the coordinate `x' is accross the
  // boundary of the system.

  int quot;
  double wrapX = std::remquo(x, system->getSystemSize(), &quot);
  if (wrapX < 0) quot -= 1;

  return quot;
}

template<class SystemClass> double diffPeriodic(
  SystemClass* system, double const& x1, double const& x2) {
  // Returns algebraic distance from `x1' to `x2' on a line taking into account
  // periodic boundary condition of the system.

    return algDistPeriod(x1, x2, system->getSystemSize());
}

template<class SystemClass> double getDistance(
  SystemClass* system, int const& index1, int const& index2) {
  // Returns distance between two particles in a given system.

  return dist2DPeriod(
    (system->getParticle(index1))->position(),
    (system->getParticle(index2))->position(),
    system->getSystemSize());
}

template<class SystemClass> double WCA_potential(SystemClass* system) {
  // Returns WCA potential of a given system.

  double potential = 0.0;
  auto addPotential = [&system, &potential](int index1, int index2) {

    double dist = getDistance<SystemClass>(system, index1, index2); // dimensionless distance between particles
    double sigma =
      ((system->getParticle(index1))->diameter()
      + (system->getParticle(index2))->diameter())/2.0; // equivalent diameter

    if (dist/sigma < pow(2., 1./6.)) { // distance lower than cut-off
      // compute potential
      potential += (system->getParameters())->getPotentialParameter()
        *(4.0*(1.0/pow(dist/sigma, 12.0) - 1.0/pow(dist/sigma, 6.0)) + 1.0);
    }
  };

  pairs_system<SystemClass>(system, addPotential);

  return potential;
}

template<class SystemClass> void WCA_force(
  SystemClass* system, int const& index1, int const& index2, double* force) {
  // Writes to `force' the force deriving from the WCA potential between
  // particles `index1' and `index2' in `system'.

  force[0] = 0.0;
  force[1] = 0.0;

  double dist = getDistance<SystemClass>(system, index1, index2); // distance between particles
  double sigma =
    ((system->getParticle(index1))->diameter()
    + (system->getParticle(index2))->diameter())/2.0; // equivalent diameter

  if (dist/sigma < pow(2., 1./6.)) { // distance lower than cut-off

    // compute force
    double coeff =
      (48.0/pow(dist/sigma, 14.0) - 24.0/pow(dist/sigma, 8.0))/pow(sigma, 2.0);
    for (int dim=0; dim < 2; dim++) {
      force[dim] = diffPeriodic<SystemClass>(system,
          (system->getParticle(index2))->position()[dim],
          (system->getParticle(index1))->position()[dim])
        *coeff;
    }
  }
}

template<class SystemClass> void add_WCA_force(
  SystemClass* system, int const& index1, int const& index2) {
  // Compute WCA forces between particles[index1] and particles[index2],
  // and add to the corresponding force lists in system.

  if ( index1 != index2 ) { // only consider different particles

    double force[2];
    WCA_force<SystemClass>(system, index1, index2, &force[0]);

    for (int dim=0; dim < 2; dim++) {
      if ( force[dim] != 0 ) {

        // update force arrays
        (system->getParticle(index1))->force()[dim] += force[dim];
        (system->getParticle(index2))->force()[dim] -= force[dim];
      }
    }
  }
}

template<class SystemClass> void initPropulsionAOUP(SystemClass* system) {
  // Draw self-propulsion vectors of each particle from their steady state
  // distribution.

  double std = sqrt(system->getTransDiffusivity()*system->getRotDiffusivity()); // standard deviation

  for (int i=0; i < system->getNumberParticles(); i++) {
    for (int dim=0; dim < 2; dim++) {
      (system->getParticle(i))->propulsion()[dim] =
        (system->getRandomGenerator())->gauss(0, std);
    }
    (system->getParticle(i))->orientation()[0] = getAngleVector(
      (system->getParticle(i))->propulsion()[0],
      (system->getParticle(i))->propulsion()[1]);
  }
}

#endif

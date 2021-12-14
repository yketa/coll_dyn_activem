#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include <vector>
#include <string>
#include <random>
#include <assert.h>
#include <iostream>

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
    Particle(Particle* particle);

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
  // https://aiichironakano.github.io/cs596/01-1LinkedListCell.pdf
  // Particles are assigned dynamically to cells tiling the system and
  // represented schematically below.
  //  _____ _____ _____
  // |     |     |     |
  // |L(6)L|L(5)L|R(4)R|
  // |_____|_____|_____|
  // |     |     |     |
  // |L(7)L| (*) |R(3)R|
  // |_____|_____|_____|
  // |     |     |     |
  // |L(8)L|R(1)R|R(2)R|
  // |_____|_____|_____|
  //
  // These cells are built such that all particles interacting with the ones in
  // cell (*) are either in cell (*) or in cells (1) to (8).
  // Using Newton's third law, it is only necessary to loop once over all cells
  // and computing interactions between particles in (*), and particles in {(*),
  // (1), (2), (3), (4)} to compute all interactions.
  // Cells (1) to (4) are denoted as the "right" (R) block, and cells (5) to (8)
  // as the "left" (L) block.

  public:

    // CONSTRUCTOR

    CellList() : N(0), L(0), n_cells(0), l_cell(0), linkedList(0), head(0) {}

    CellList(int const& N_, double const& L_, double const& r_cut_) :
      N(N_), L(L_), n_cells(floor(L/r_cut_)), l_cell(L/n_cells),
      linkedList(N, 0), head(n_cells*n_cells, 0) {
      // assert(n_cells >= 3); // only works for 9+ boxes grids
    }

    CellList(
      std::vector<std::vector<double>> const& positions_,
      double const& L_, double const& r_cut_) :
      N(positions_.size()), L(L_), n_cells(floor(L/r_cut_)), l_cell(L/n_cells),
      linkedList(N, 0), head(n_cells*n_cells, 0) {
      // assert(n_cells >= 3); // only works for 9+ boxes grids
      listConstructor<std::vector<double>>(positions_); // build linked list
    }

    // METHODS

    template<typename T> void listConstructor(std::vector<T> const& positions) {
      //  Construct linked list.

      int c; // cell index
      for (c=0; c < n_cells*n_cells; c++) { head[c] = -1; }
      for (int i=0; i < N; i++) {
        c =
          floor(positions[i][0]/l_cell) + n_cells*floor(positions[i][1]/l_cell);
        linkedList[i] = head[c]; // link to previous occupant
        head[c] = i; // last one goes to the header
      }
    }

    template<typename F> void pairs(F function) {
      // Compute function for all neighbouring particles.

      int  i, j; // particle indices
      int cx1, cy1, c1; // index of cell
      int cx2, cy2, c2; // index of neighbouring cell
      for (cx1=0; cx1 < n_cells; cx1++) {
        for (cy1=0; cy1 < n_cells; cy1++) {
          c1 = cx1 + n_cells*cy1;
          for (int inc=0; inc < 5; inc++) {
            cx2 = cx1 + increments[inc][0];
            if ( cx2 == -1 ) { cx2 = n_cells - 1; }
            if ( cx2 == n_cells ) { cx2 = 0; }
            cy2 = cy1 + increments[inc][1];
            if ( cy2 == -1 ) { cy2 = n_cells - 1; }
            if ( cy2 == n_cells ) { cy2 = 0; }
            c2 = cx2 + n_cells*cy2;

            // COMPUTE PAIRS
            i = head[c1];
            while ( i != -1 ) { // loop over particles of cell
              j = head[c2];
              while ( j != -1 ) { // loop over particles of neighbouring cell
                if ( c1 != c2 || i < j ) { // avoid double counting of pair (i, j) in same box
                  function(i, j);
                }
                j = linkedList[j];
              }
              i = linkedList[i];
            }
          }
        }
      }
    }

  private:

    double N; // number of particles
    double L; // system size
    double n_cells; // linear number of cells in the cell list
    double l_cell; // linear size of a cell

    double increments[5][2] = {{1, -1}, {0, 0}, {1, 0}, {0, 1}, {1, 1}}; // increments to RIGHT neighbouring cells (including cell itself)

    std::vector<int> linkedList; // linkedList[i] = particle indices to which the i-th particle points
    std::vector<int> head; // head[c] = index of the first particle in the c-th cell or -1 if there is no particle in the cell

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
    Parameters( // defining all parameters independently (L and lp from others)
      int N, double epsilon, double v0, double D, double Dr,
        double phi, std::vector<double> const& diameters,
      double dt);
    Parameters( // defining all parameters independently (all)
      int N, double epsilon, double v0, double D, double Dr, double lp,
        double phi, std::vector<double> const& diameters, double L,
      double dt, double Lc = 0);
    Parameters( // defining all parameters independently and confining length
      int N, double epsilon, double v0, double D, double Dr,
        double phi, double Lc, std::vector<double> const& diameters,
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
    // Parameters( // copy .datC files
    //   DatC const& inputDat, double dt = 0) :
    //   Parameters(
    //     inputDat.getNumberParticles(),
    //     inputDat.getPotentialParameter(),
    //     inputDat.getPropulsionVelocity(),
    //     inputDat.getTransDiffusivity(),
    //     inputDat.getRotDiffusivity(),
    //     inputDat.getPackingFraction(),
    //     inputDat.getConfiningLength(),
    //     inputDat.getDiameters(),
    //     dt > 0 ? dt : inputDat.getTimeStep()) {}

    // METHODS

    int getNumberParticles() const; // returns number of particles in the system
    double getPotentialParameter() const; // returns coefficient parameter of potential
    double getPropulsionVelocity() const; // returns self-propulsion velocity
    double getTransDiffusivity() const; // returns translational diffusivity
    double getRotDiffusivity() const; // returns rotational diffusivity
    double getPersistenceLength() const; // returns persistence length
    double getPackingFraction() const; // returns packing fraction
    double getSystemSize() const; // returns system size
    double getConfiningLength() const; // returns confining length
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
    double const confiningLength; // confining length
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
    std::vector<double> getDiameters() const; // returns vector of diameters

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
    void initialiseCellList(double const& cutOff = pow(2.0, 1./6.)); // initialise cell list
    void updateCellList(); // update cell list with current positions

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
    void initialiseCellList(double const& cutOff = pow(2.0, 1./6.)); // initialise cell list
    void updateCellList(); // update cell list with current positions

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
   *  | (int) N | (double) epsilon | (double) v0 | (double) D | (double) Dr | (double) lp | (double) phi | (double) L | (int) seed | (double) dt |
   *  || (int) NinitialTimes | (int) initialTimes[0] | ... | initialTimes[NinitialTimes - 1] ||
   *  || (int) NlagTimes | (int) lagTimes[0]  | ... | (int) lagTimes[NlagTimes - 1] ||
   *  || (int) Nframes | (int) frameIndices[0] | ... | (int) frameIndices[Nframes - 1] ||
   *  ||    PARTICLE 1     | ... |     PARTICLE N    ||
   *  || (double) diameter | ... | (double) diameter ||
   *
   *  [FRAMES (see SystemN::saveInitialState & SystemN::saveNewState)] (all double)
   *  ||                                   FRAME I                                   ||
   *  ||                        PARTICLE 1                        | ... | PARTICLE N ||
   *  ||   R   | ORIENTATION |     V     |     P     | UNFOLDED R | ... |     ...    ||
   *  || X | Y |    theta    | V_X | V_Y | P_X | P_Y |   X  |  Y  | ... |     ...    ||
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
    SystemN(
      int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
        std::vector<int>* time0, std::vector<int>* deltat,
      std::string inputFilename, int inputFrame = 0, Parameters* parameters = 0,
        std::vector<double> const diameters = {0},
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
    void initialiseCellList(double const& cutOff = pow(2.0, 1./6.)); // initialise cell list
    void updateCellList(); // update cell list with current positions

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

    bool isConfined() const { return false; }

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

    double kineticEnergy = 0; // sum of squared velocities
    double kineticEnergyFactor = 1e3; // threshold in units of squared self-propulsion velocity not to exceed for mean squared velocities

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

std::vector<double> getDiametersI(int N, double I, int seed = 0);
  // Returns vector of `N' uniformly distributed diameters with polydispersity
  // index `I'.

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
  // cell list if USE_CELL_LIST is defined or a double loop.

  #ifdef USE_CELL_LIST // with cell list

  (system->getCellList())->pairs(function);

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
  SystemClass* system, int const& index1, int const& index2, double* diff) {
  // Returns distance between two particles in a given system.

  return dist2DPeriod(
    (system->getParticle(index1))->position(),
    (system->getParticle(index2))->position(),
    system->getSystemSize(),
    diff);
}

template<class SystemClass> double WCA_potential(SystemClass* system) {
  // Returns WCA potential of a given system.

  double potential = 0.0;
  double diff[2];
  auto addPotential = [&system, &potential, &diff](int index1, int index2) {

    double dist = getDistance<SystemClass>(system, index1, index2, &diff[0]); // dimensionless distance between particles
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
  SystemClass* system, int const& index1, int const& index2,
  double* force, double* diff) {
  // Writes to `force' the force deriving from the WCA potential between
  // particles `index1' and `index2' in `system'.

  force[0] = 0.0;
  force[1] = 0.0;

  double dist = getDistance<SystemClass>(system, index1, index2, diff); // distance between particles
  double sigma =
    ((system->getParticle(index1))->diameter()
    + (system->getParticle(index2))->diameter())/2.0; // equivalent diameter

  if (dist/sigma < pow(2., 1./6.)) { // distance lower than cut-off

    // compute force
    double coeff =
      (48.0/pow(dist/sigma, 14) - 24.0/pow(dist/sigma, 8))/pow(sigma, 2);
    for (int dim=0; dim < 2; dim++) {
      force[dim] = -diff[dim]*coeff;
    }
  }
}

template<class SystemClass> void add_WCA_force(
  SystemClass* system, int const& index1, int const& index2,
  double* force, double* diff) {
  // Compute WCA forces between particles[index1] and particles[index2],
  // and add to the corresponding force lists in system.

  if ( index1 != index2 ) { // only consider different particles

    WCA_force<SystemClass>(system, index1, index2, force, diff);

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

template<class DatClass> void checkCellList(
  std::string const& filename, int const& frame,
  double const& cutOff = pow(2.0, 1./6.)) {
  // Check that cell list gives consistent neighbour list with brute force
  // double loop on the `frame'-th frame of `filename'.

  // data file
  DatClass dat(filename);
  std::cout << "FILE: " << (dat.getInput())->getInputFile() << std::endl;

  // parameters
  int N = dat.getNumberParticles();
  double L = dat.getSystemSize();
  std::vector<double> diameters = dat.getDiameters();

  // cell list
  std::vector<std::vector<double>> positions(N, std::vector<double> (2, 0));
  std::vector<double*> positionsPTR(N);
  for (int i=0; i < N; i++) {
    positionsPTR[i] = &(positions[i][0]);
    for (int dim=0; dim < 2; dim++) {
      positions[i][dim] = dat.getPosition(frame, i, dim, false);
    }
  }
  CellList cl(N, L,
    cutOff*(*std::max_element(diameters.begin(), diameters.end())));
  cl.listConstructor<double*>(positionsPTR);

  std::cout << "FRAME: " << frame << std::endl;
  std::cout << "CUT OFF: " << cutOff << std::endl << std::endl;
  double diff[2];
  double dist;
  double sigmaij;
  long int couples = 0;

  // brute force neighbours
  couples = 0;
  std::vector<std::vector<int>> neighboursBF(N, std::vector<int>(0));
  for (int i=0; i < N; i++) {
    for (int j=i + 1; j < N; j++) {
      dist = dist2DPeriod(positionsPTR[i], positionsPTR[j], L, &diff[0]);
      sigmaij = (diameters[i] + diameters[j])/2;
      if ( dist/sigmaij < cutOff ) {
        neighboursBF[i].push_back(j);
        neighboursBF[j].push_back(i);
        couples++;
      }
    }
  }
  std::cout << "[BRUTE FORCE] couples: " << couples << std::endl;

  // cell list neighbours
  couples = 0;
  std::vector<std::vector<int>> neighboursCL(N, std::vector<int>(0));
  for (int i=0; i < N; i++) { neighboursCL[i].clear(); }
  auto f =
    [&dist, &diff, &positionsPTR, &L, &sigmaij, &diameters, &couples, &cutOff,
      &neighboursCL]
    (int const& i, int const& j) {
      dist = dist2DPeriod(positionsPTR[i], positionsPTR[j], L, &diff[0]);
      sigmaij = (diameters[i] + diameters[j])/2;
      if ( dist/sigmaij < cutOff ) {
        neighboursCL[i].push_back(j);
        neighboursCL[j].push_back(i);
        couples++;
      }
    };
  cl.pairs(f);
  std::cout << "[CELL LIST] couples: " << couples << std::endl;

  // check
  std::cout << "CHECK: ";
  for (int i=0; i < N; i++) {
    if ( !compareVec(neighboursCL[i], neighboursBF[i]) ) {
      std::cout << std::endl << "[particle " << i << "]" << std::endl;
      std::cout << "[BRUTE FORCE] neighbours: ";
      for (int n : neighboursBF[i]) { std::cout << n << " "; }
      std::cout << std::endl << "[CELL LIST] neighbours: ";
      for (int n : neighboursCL[i]) { std::cout << n << " "; }
      std::cout << std::endl;
      throw std::invalid_argument("Different neighbours.");
    }
  }
  std::cout << "OK" << std::endl << std::endl;
}

#endif

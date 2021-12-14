#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include <vector>
#include <string>
#include <random>
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
  // Confinement, if present, is applied on the vertical (1) direction, and the
  // horizontal (0) direction remains periodic.
  // NOTE: This decomposition works ONLY when there at least 3 blocks in the
  //       PERIODIC direction.

  public:

    // CONSTRUCTORS

    CellList(double const& L, double const& Lc = 0) :
      systemSize(L),
      isConfined((Lc > 0)),
      confiningLength(Lc),
      cutOff(), dispMax(), numberParticles(), numberBoxes(), sizeBox(),
      cellList(), rightNeigbours(), leftNeigbours() {}

    // DESTRUCTORS

    ~CellList() {;}

    // METHODS

    int getNumberBoxes() { return numberBoxes[0]*numberBoxes[1]; } // return number of boxes
    int* getNumberBoxesFull() { return &(numberBoxes[0]); } // return pointer to number of boxes
    std::vector<int>* getCell(int const& index) { return &(cellList[index]); } // return pointer to vector of indexes in cell

    void initialise(std::vector<double> const& diameters,
      double const& potentialCutOff = pow(2.0, 1./6.)) {
      // Initialise cell list.

      numberParticles = diameters.size();

      #ifdef USE_CELL_LIST // this is not useful when not using cell lists

      // parameters of cell list
      cutOff = potentialCutOff
        *(*std::max_element(diameters.begin(), diameters.end()));
      numberBoxes[0] =
        std::max((int) (systemSize/cutOff), 1);
      numberBoxes[1] = isConfined ?
        std::max((int) (confiningLength/cutOff), 1) : numberBoxes[0];
      if ( numberBoxes[0] <= 2 && numberBoxes[1] <= 2 ) { // all boxes interact with each others
        numberBoxes[0] = 1;
        numberBoxes[1] = 1;
      }
      sizeBox[0] = systemSize/numberBoxes[0];
      sizeBox[1] = isConfined ? confiningLength/numberBoxes[1] : sizeBox[0];
      dispMax = (std::min(sizeBox[0], sizeBox[1]) - cutOff)/2;

      // set size of cell list
      cellList.clear();
      for (int i=0; i < numberBoxes[0]*numberBoxes[1]; i++) {
        cellList.push_back(std::vector<int>());
      }

      // create neighbouring cells lists
      rightNeigbours.clear();
      leftNeigbours.clear();
      std::vector<std::tuple<int,int>> rightIncrements(0);
      rightIncrements.push_back(std::make_tuple(0, -1));
      rightIncrements.push_back(std::make_tuple(1, -1));
      rightIncrements.push_back(std::make_tuple(1, 0));
      rightIncrements.push_back(std::make_tuple(1, 1));
      int i;
      int dx, dy;
      int neighbour;
      bool flag = true;
      for (int y=0; y < numberBoxes[1]; y++) {
        for (int x=0; x < numberBoxes[0]; x++) {
          i = x + numberBoxes[0]*y; // cell index
          rightNeigbours.push_back(std::vector<int>());
          leftNeigbours.push_back(std::vector<int>());
          for (std::tuple<int,int> rI : rightIncrements) {
            if ( getNumberBoxes() == 1 ) { break; }
            // *** right neighbours ***
            if ( numberBoxes[0] > 2 ) { // at least 3 blocks in the periodic direction
              dx = std::get<0>(rI);
              dy = std::get<1>(rI);
            }
            else { // only 2 blocks in the periodic direction (trick is to take increments {-dy, dx})
              dx = -std::get<1>(rI);
              dy = std::get<0>(rI);
              if ( x + dx >= numberBoxes[0] || x + dx < 0 ) { flag = false; }
            }
            if ( flag &&
              ( !isConfined || ( y + dy < numberBoxes[1] && y + dy >= 0 ) ) ) {
              neighbour =
                  (numberBoxes[0] + (x + dx))%numberBoxes[0]    // neighbour x
                + numberBoxes[0]*
                  ((numberBoxes[1] + (y + dy))%numberBoxes[1]); // neighbour y
              rightNeigbours[i].push_back(neighbour);
            }
            flag = true;
            // *** left neighbours ***
            if ( numberBoxes[0] > 2 ) { // at least 3 blocks in the periodic direction
              dx = -std::get<0>(rI);
              dy = -std::get<1>(rI);
            }
            else { // only 2 blocks in the periodic direction (trick is to take increments {-dy, dx})
              dx = std::get<1>(rI);
              dy = -std::get<0>(rI);
              if ( x + dx >= numberBoxes[0] || x + dx < 0 ) { flag = false; }
            }
            if ( flag &&
              ( !isConfined || ( y + dy < numberBoxes[1] && y + dy >= 0 ) ) ) {
              neighbour =
                  (numberBoxes[0] + (x + dx))%numberBoxes[0]    // neighbour x
                + numberBoxes[0]*
                  ((numberBoxes[1] + (y + dy))%numberBoxes[1]); // neighbour y
              leftNeigbours[i].push_back(neighbour);
            }
            flag = true;
          }
        }
      }

      #endif
    }

    void update(std::vector<double*> const& positions) {
      // Put particles in the cell list.

      #ifdef USE_CELL_LIST // this is not useful when not using cell lists

      // flush old lists
      for (int i=0; i < (int) cellList.size(); i++) {
        cellList[i].clear();
      }

      // create new lists
      for (int i=0; i < (int) positions.size(); i++) {
        cellList[index(positions[i])].push_back(i); // particles are in increasing order of indices
      }

      #endif
    }

    int index(double* const& position) {
      // Index of the box corresponding to a given position.

      int x = std::floor(position[0]/sizeBox[0]);
      int y = std::floor(position[1]/sizeBox[1]);
      // check values are in {0, ..., numberBoxes - 1}
      while ( x < 0 ) x += numberBoxes[0];
      while ( x >= numberBoxes[0] ) x -= numberBoxes[0];
      while ( y < 0 ) y += numberBoxes[1];
      while ( y >= numberBoxes[1] ) y -= numberBoxes[1];

      return x + numberBoxes[0]*y;
    }

    std::vector<int>* getRightNeighbourCells(int const& cellIndex) {
      // Returns pointer to vector of indexes of "right" (R) neighbouring cells.
      return &(rightNeigbours[cellIndex]);
    }
    std::vector<int>* getLeftNeighbourCells(int const& cellIndex) {
      // Returns pointer to vector of indexes of "left" (L) neighbouring cells.
      return &(leftNeigbours[cellIndex]);
    }

    std::vector<int> getNeighbours(double* const& position) {
      // Returns vector of indexes of particles neighbouring a given position.

      std::vector<int> neighbours(0); // vector of neighbouring particles

      int cellIndex = index(position);

      // particles in same cell
      for (int i : cellList[cellIndex]) {
        neighbours.push_back(i);
      }
      // right neighbours
      for (int c : rightNeigbours[cellIndex]) {
        for (int i : cellList[c]) {
          neighbours.push_back(i);
        }
      }
      // left neighbours
      for (int c : leftNeigbours[cellIndex]) {
        for (int i : cellList[c]) {
          neighbours.push_back(i);
        }
      }

      return neighbours;
    }

    template<typename F> void pairs(F function) {
      // Given a function `function` with parameters (int index1, int index2),
      // call this function with every unique pair of interacting particles.

      #ifdef USE_CELL_LIST

      int index1, index2; // index of the couple of particles
      // int i, j; // indexes of the cells
      int k, l; // indexes of the particles in the cell
      std::vector<int> cell1, cell2; // cells
      for (int i=0; i < getNumberBoxes(); i++) { // loop over cells

        cell1 = cellList[i]; // indexes of particles in the first cell
        for (k=0; k < (int) cell1.size(); k++) { // loop over particles in the first cell
          index1 = cell1[k];

          // interactions with particles in the same cell
          for (l=k+1; l < (int) cell1.size(); l++) { // loop over particles in the first cell
            index2 = cell1[l];
            function(index1, index2);
          }

          // interations with particles in other cells
          for (int j : rightNeigbours[i]) { // loop over "right" (R) neighbouring cells
            cell2 = cellList[j]; // indexes of particles in the second cell
            for (int index2 : cell2) { // loop over particles in the second cell
              function(index1, index2);
            }
          }
        }
      }

      #else

      for (int index1=0; index1 < numberParticles; index1++) {
        for (int index2=index1 + 1; index2 < numberParticles; index2++) {
          function(index1, index2);
        }
      }

      #endif
    }

  private:

    // ATTRIBUTES

    double const systemSize; // size of the system
    bool const isConfined; // confinement state of the system
    double const confiningLength; // confining length

    double cutOff; // cut-off radius of the interactions
    double dispMax; // maximum allowed displacement before recomputing the celllist

    int numberParticles; // number of particles
    int numberBoxes[2]; // number of boxes in each dimension
    double sizeBox[2]; // size of each box

    std::vector<std::vector<int>> cellList; // cells with indexes of particles
    std::vector<std::vector<int>> rightNeigbours; // "right" (R) neighbouring cells
    std::vector<std::vector<int>> leftNeigbours; // "left" (L) neighbouring cells

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


// /*  SYSTEMC
//  *  -------
//  *  Store physical and integration parameter.
//  *  Access to distance and potentials.
//  *  Save system state to output file ONLY for pre-defined frames.
//  *  Using all free parameters.
//  */
//
// class SystemC {
//   /*  Contains all the details to simulate a system of confinded active
//    *  particles.
//    *  (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#Active%20Brownian%20particles)
//    *  (see https://yketa.github.io/PhD_Wiki/#Active%20Ornstein-Uhlenbeck%20particles)
//    *  (see https://yketa.github.io/PhD_Wiki/#Confined%AOUP)
//    *
//    *  Parameters are stored in a binary file with the following structure:
//    *
//    *  [HEADER (see System::System)]
//    *  | (int) N | (double) epsilon | (double) v0 | (double) D | (double) Dr | (double) lp | (double) phi | (double) L | (double) Lc | (int) seed | (double) dt |
//    *  || (int) NinitialTimes | (int) initialTimes[0] | ... | initialTimes[NinitialTimes - 1] ||
//    *  || (int) NlagTimes | (int) lagTimes[0]  | ... | (int) lagTimes[NlagTimes - 1] ||
//    *  || (int) Nframes | (int) frameIndices[0] | ... | (int) frameIndices[Nframes - 1] ||
//    *  ||    PARTICLE 1     | ... |     PARTICLE N    ||
//    *  || (double) diameter | ... | (double) diameter ||
//    *
//    *  [FRAMES (see SystemC::saveInitialState & SystemC::saveNewState)] (all double)
//    *  ||                                 FRAME I                                 ||
//    *  ||                      PARTICLE 1                      | ... | PARTICLE N ||
//    *  ||   R   | ORIENTATION |   V   |     P     | UNFOLDED R | ... |     ...    ||
//    *  || X | Y |    theta    | 0 | 0 | P_X | P_Y |   X  |  Y  | ... |     ...    ||
//    */
//
//   public:
//
//     // CONSTRUCTORS
//
//     SystemC();
//     SystemC(
//       int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
//         std::vector<int>* time0, std::vector<int>* deltat,
//       Parameters* parameters, int seed = 0, std::string filename = "");
//     SystemC(
//       int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
//         std::vector<int>* time0, std::vector<int>* deltat,
//       Parameters* parameters, std::vector<double>& diameters, int seed = 0,
//       std::string filename = "");
//     SystemC(
//       int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
//         std::vector<int>* time0, std::vector<int>* deltat,
//       SystemC* system, int seed = 0, std::string filename = "");
//     SystemC(
//       int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
//         std::vector<int>* time0, std::vector<int>* deltat,
//       SystemC* system, std::vector<double>& diameters, int seed = 0,
//       std::string filename = "");
//     SystemC(
//       int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
//         std::vector<int>* time0, std::vector<int>* deltat,
//       std::string inputFilename, int inputFrame = 0, double dt = 0,
//       int seed = 0, std::string filename = "");
//     SystemC(
//       int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
//         std::vector<int>* time0, std::vector<int>* deltat,
//       std::string inputFilename, int inputFrame = 0, Parameters* parameters = 0,
//       int seed = 0, std::string filename = "");
//     SystemC(
//       int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
//         std::vector<int>* time0, std::vector<int>* deltat,
//       std::string inputFilename, int inputFrame = 0, Parameters* parameters = 0,
//         std::vector<double> const diameters = {0},
//       int seed = 0, std::string filename = "");
//
//     // DESTRUCTORS
//
//     ~SystemC();
//
//     // METHODS
//
//     std::vector<int> const* getFrames(); // returns pointer to vector of indices of frames to save
//
//     Parameters* getParameters(); // returns pointer to class of parameters
//     std::vector<double> getDiameters() const; // returns vector of diameters
//
//     int getNumberParticles() const; // returns number of particles
//     double getPotentialParameter() const; // returns coefficient parameter of potential
//     double getPropulsionVelocity() const; // returns self-propulsion velocity
//     double getTransDiffusivity() const; // returns translational diffusivity
//     double getRotDiffusivity() const; // returns rotational diffusivity
//     double getPersistenceLength() const; // returns persistence length
//     double getPackingFraction() const; // returns packing fraction
//     double getSystemSize() const; // returns system size
//     double getConfiningLength() const; // returns confining length
//     double getTimeStep() const; // returns time step
//
//     int getRandomSeed() const; // returns random seed
//     Random* getRandomGenerator(); // returns pointer to random generator
//
//     Particle* getParticle(int const& index); // returns pointer to given particle
//     std::vector<Particle> getParticles(); // returns vector of particles
//
//     CellList* getCellList(); // returns pointer to CellList object
//     void initialiseCellList(double const& cutOff = pow(2.0, 1./6.)); // initialise cell list
//     void updateCellList(); // update cell list with current positions
//
//     std::string getOutputFile() const; // returns output file name
//
//     int* getDump(); // returns number of frames dumped since last reset
//     void resetDump();
//       // Reset time-extensive quantities over trajectory.
//     void copyDump(SystemC* system);
//       // Copy dumps from other system.
//       // WARNING: This also copies the index of last frame dumped. Consistency
//       //          has to be checked.
//
//     void copyState(std::vector<Particle>& newParticles);
//       // Copy positions and orientations.
//     void copyState(SystemC* system);
//       // Copy positions and orientations.
//
//     void saveInitialState();
//       // Saves initial state of particles to output file.
//     void saveNewState(std::vector<Particle>& newParticles);
//       // Saves new state of particles to output file then copy it.
//
//     bool isConfined() const { return true; }
//
//   private:
//
//     // ATTRIBUTES
//
//     std::vector<int> const frameIndices; // indices of frames to save
//     // NOTE: This vector is sorted and the element 0 is removed.
//     // NOTE: Frame 0 is ALWAYS saved first.
//
//     Parameters param; // class of simulation parameters
//
//     int const randomSeed; // random seed
//     Random randomGenerator; // random number generator
//
//     std::vector<Particle> particles; // vector of particles
//
//     CellList cellList; // cell list
//
//     Write output; // output class
//     std::vector<long int> velocitiesDumps; // locations in output file to dump velocities
//
//     int dumpFrame; // number of frames dumped since last reset
//
//     double kineticEnergy = 0; // sum of squared velocities
//     double kineticEnergyFactor = 1e3; // threshold in units of squared self-propulsion velocity not to exceed for mean squared velocities
//
// };


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
  CellList cl(L);
  cl.initialise(diameters, cutOff);
  cl.update(positionsPTR);

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

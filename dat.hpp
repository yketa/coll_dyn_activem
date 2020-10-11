#ifndef DAT_HPP
#define DAT_HPP

#include <string>
#include <vector>

#include "readwrite.hpp"

/////////////
// CLASSES //
/////////////

class Dat;
class Dat0;
class DatN;
class DatR;


/*  DAT
 *  ---
 *  Read files as defined by the System class (see particle.hpp).
 */

class Dat {

  public:

    // CONSTRUCTORS

    Dat(std::string filename, bool loadWork = true, bool corruption = false);

    // DESTRUCTORS

    ~Dat();

    // METHODS

    int getNumberParticles() const; // returns number of particles
    double getPersistenceLength() const; // returns persistence length
    double getPackingFraction() const; // returns packing fraction
    double getSystemSize() const; // returns system size
    double getTorqueParameter() const; // returns torque parameter
    int getRandomSeed() const; // returns random seed
    double getTimeStep() const; // returns time step
    int getFramesWork() const; // returns number of frames on which to sum the active work before dumping

    long int getNumberWork() const; // returns number of computed work sums
    long int getFrames() const; // returns number of frames

    std::vector<double> getActiveWork(); // returns vector of computed active work sums
    std::vector<double> getActiveWorkForce(); // returns vector of computed active work (force) sums
    std::vector<double> getActiveWorkOri(); // returns vector of computed active work (orientation) sums
    std::vector<double> getOrderParameter(); // returns vector of computed order parameter sums
    std::vector<double> getOrderParameter0(); // returns vector of computed order parameter along x-axis sums
    std::vector<double> getOrderParameter1(); // returns vector of computed order parameter along y-axis sums
    std::vector<double> getTorqueIntegral1(); // returns vector of computed first torque integrals
    std::vector<double> getTorqueIntegral2(); // returns vector of computed second torque integrals

    double getPosition(
      int const& frame, int const& particle, int const& dimension);
      // Returns position of a given particle at a given frame.
    double getOrientation(int const& frame, int const& particle);
      // Returns orientation of a given particle at a given frame.
    double getVelocity(
      int const& frame, int const& particle, int const& dimension);
      // Returns velocity of a given particle at a given frame.

    void close() { input.close(); } // close file stream
    void open() { input.open(); } // open file stream

  private:

    // ATTRIBUTES

    int const numberParticles; // number of particles
    double const persistenceLength; // persistence length
    double const packingFraction; // packing fraction
    double const systemSize; // size of the system
    double const torqueParameter; // torque parameter
    int const randomSeed; // random seed
    double const timeStep; // time step
    int const framesWork; // number of frames on which to sum the active work before dumping
    bool const dumpParticles; // positions and orientations dumped in file
    int const dumpPeriod; // period of dumping of positions and orientations in number of frames

    Read input; // input class

    long int headerLength; // length of header in input file
    long int particleLength; // length the data of a single particle takes in a frame
    long int frameLength; // length the data of a single frame takes in a file
    long int workLength; // length the data of a single work dump takes in a file

    long int numberWork; // number of computed work sums
    long int frames; // number of frames

    std::vector<double> activeWork; // computed active work sums
    std::vector<double> activeWorkForce; // computed active work (force) sums
    std::vector<double> activeWorkOri; // computed active work (orientation) sums
    std::vector<double> orderParameter; // computer order parameter sums
    std::vector<double> orderParameter0; // computer order parameter along x-axis sums
    std::vector<double> orderParameter1; // computer order parameter along y-axis sums
    std::vector<double> torqueIntegral1; // computed first torque integrals
    std::vector<double> torqueIntegral2; // computed second torque integrals

};



/*  DAT0
 *  ----
 *  Read files as defined by the System0 class (see particle.hpp).
 */

class Dat0 {

  public:

    // CONSTRUCTORS

    Dat0(std::string filename, bool loadWork = true, bool corruption = false);

    // DESTRUCTORS

    ~Dat0();

    // METHODS

    int getNumberParticles() const; // returns number of particles
    double getPotentialParameter() const; // returns coefficient parameter of potential
    double getPropulsionVelocity() const; // returns self-propulsion velocity
    double getTransDiffusivity() const; // returns translational diffusivity
    double getRotDiffusivity() const; // returns rotational diffusivity
    double getPersistenceLength() const; // returns persistence length
    double getPackingFraction() const; // returns packing fraction
    double getSystemSize() const; // returns system size
    int getRandomSeed() const; // returns random seed
    double getTimeStep() const; // returns time step
    int getFramesWork() const; // returns number of frames on which to sum the active work before dumping

    long int getNumberWork() const; // returns number of computed work sums
    long int getFrames() const; // returns number of frames

    std::vector<double> getDiameters() const; // returns vector of diameters

    std::vector<double> getActiveWork(); // returns vector of computed active work sums
    std::vector<double> getActiveWorkForce(); // returns vector of computed active work (force) sums
    std::vector<double> getActiveWorkOri(); // returns vector of computed active work (orientation) sums
    std::vector<double> getOrderParameter(); // returns vector of computed order parameter sums

    double getPosition(
      int const& frame, int const& particle, int const& dimension,
      bool const& unfolded = false);
      // Returns position of a given particle at a given frame.
    double getOrientation(int const& frame, int const& particle);
      // Returns orientation of a given particle at a given frame.
    double getVelocity(
      int const& frame, int const& particle, int const& dimension);
      // Returns velocity of a given particle at a given frame.
    double getPropulsion(
      int const& frame, int const& particle, int const& dimension);
      // Returns self-propulsion vector of a given particle at a given frame.

  private:

    // ATTRIBUTES

    int const numberParticles; // number of particles
    double const potentialParameter; // coefficient parameter of potential
    double const propulsionVelocity; // self-propulsion velocity
    double const transDiffusivity; // translational diffusivity
    double const rotDiffusivity; // rotational diffusivity
    double const persistenceLength; // persistence length
    double const packingFraction; // packing fraction
    double const systemSize; // size of the system
    int const randomSeed; // random seed
    double const timeStep; // time step
    int const framesWork; // number of frames on which to sum the active work before dumping
    bool const dumpParticles; // positions and orientations dumped in file
    int const dumpPeriod; // period of dumping of positions and orientations in number of frames

    Read input; // input class

    long int headerLength; // length of header in input file
    long int particleLength; // length the data of a single particle takes in a frame
    long int frameLength; // length the data of a single frame takes in a file
    long int workLength; // length the data of a single work dump takes in a file

    long int numberWork; // number of computed work sums
    long int frames; // number of frames

    std::vector<double> diameters; // array of diameters

    std::vector<double> activeWork; // computed active work sums
    std::vector<double> activeWorkForce; // computed active work (force) sums
    std::vector<double> activeWorkOri; // computed active work (orientation) sums
    std::vector<double> orderParameter; // computer order parameter sums

};


/*  DATN
 *  ----
 *  Read files as defined by the SystemN class (see particle.hpp).
 */

class DatN {

  public:

    // CONSTRUCTORS

    DatN(std::string filename, bool loadWork = true, bool corruption = false);

    // DESTRUCTORS

    ~DatN();

    // METHODS

    int getNumberParticles() const; // returns number of particles
    double getPotentialParameter() const; // returns coefficient parameter of potential
    double getPropulsionVelocity() const; // returns self-propulsion velocity
    double getTransDiffusivity() const; // returns translational diffusivity
    double getRotDiffusivity() const; // returns rotational diffusivity
    double getPersistenceLength() const; // returns persistence length
    double getPackingFraction() const; // returns packing fraction
    double getSystemSize() const; // returns system size
    int getRandomSeed() const; // returns random seed
    double getTimeStep() const; // returns time step

    std::vector<int>* getTime0(); // returns pointer to vector of initial frames
    std::vector<int>* getDt(); // returns pointer to vector of lag times
    std::vector<int>* getFrames(); // returns pointer to vector of frames which were saved

    std::vector<double> getDiameters() const; // returns vector of diameters

    double getPosition(
      int const& frame, int const& particle, int const& dimension,
      bool const& unfolded = false);
      // Returns position of a given particle at a given frame.
    double getOrientation(int const& frame, int const& particle);
      // Returns orientation of a given particle at a given frame.
    double getVelocity(
      int const& frame, int const& particle, int const& dimension);
      // Returns velocity of a given particle at a given frame.
    double getPropulsion(
      int const& frame, int const& particle, int const& dimension);
      // Returns self-propulsion vector of a given particle at a given frame.

    int getFrameIndex(int const& frame);
      // Returns index of frame in file.

  private:

    // ATTRIBUTES

    int const numberParticles; // number of particles
    double const potentialParameter; // coefficient parameter of potential
    double const propulsionVelocity; // self-propulsion velocity
    double const transDiffusivity; // translational diffusivity
    double const rotDiffusivity; // rotational diffusivity
    double const persistenceLength; // persistence length
    double const packingFraction; // packing fraction
    double const systemSize; // size of the system
    int const randomSeed; // random seed
    double const timeStep; // time step

    int const initialTimes; // number of initial times
    std::vector<int> time0; // vector of initial frames
    int const lagTimes; // number of lag times
    std::vector<int> dt; // vector of lag times
    int const frames; // number of frames (minus 0)
    std::vector<int> frameIndices; // vector of frames which were saved (including frame 0)

    Read input; // input class

    long int headerLength; // length of header in input file
    long int particleLength; // length the data of a single particle takes in a frame
    long int frameLength; // length the data of a single frame takes in a file

    std::vector<double> diameters; // array of diameters

};


/*  DATR
 *  ----
 *  Read files as defined by the Rotors class (see particle.hpp).
 */

class DatR {

  public:

    // CONSTRUCTORS

    DatR(std::string filename, bool loadOrder = true);

    // DESTRUCTORS

    ~DatR();

    // METHODS

    int getNumberParticles() const; // returns number of rotors
    double getRotDiffusivity() const; // returns rotational diffusivity
    double getTorqueParameter() const; // returns aligning torque parameter
    double getTimeStep() const; // returns time step
    int getDumpPeriod() const; // returns period of dumping of orientations in number of frames
    int getRandomSeed() const; // returns random seed

    long int getFrames() const; // returns number of frames

    std::vector<double> getOrderParameter(); // returns vector of computed order parameter sums
    std::vector<double> getOrderParameterSq(); // returns vector of computed squared order parameter sums

    double getOrientation(int const& frame, int const& rotor);
      // Returns position of a given rotor at a given frame.

  private:

    // ATTRIBUTES

    int const numberParticles; // number of rotors
    double const rotDiffusivity; // rotational diffusivity
    double const torqueParameter; // aligning torque parameter
    double const timeStep; // time step
    int const framesOrder; // number of frames on which to average the order parameter before dumping
    bool const dumpRotors; // orientations dumped in file
    int const dumpPeriod; // period of dumping of orientations in number of frames
    int const randomSeed; // random seed

    Read input; // input class

    long int headerLength; // length of header in input file
    long int rotorLength; // length the data of a single rotor takes in a frame
    long int frameLength; // length the data of a single frame takes in a file
    long int orderLength; // length the data of a single order dump takes in a file

    long int numberOrder; // number of computed order parameter sums
    long int frames; // number of frames

    std::vector<double> orderParameter; // computed order parameter sums
    std::vector<double> orderParameterSq; // computed squared order parameter sums

};

#endif

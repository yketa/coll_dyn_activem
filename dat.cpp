#include <string>
#include <iostream>
#include <vector>
#include <algorithm>

#include "dat.hpp"

/*******
 * DAT *
 *******/

// CONSTRUCTORS

Dat::Dat(std::string filename, bool loadWork) :
  numberParticles(), persistenceLength(), packingFraction(), systemSize(),
    torqueParameter(), randomSeed(), timeStep(), framesWork(), dumpParticles(),
    dumpPeriod(),
  input(filename) {

  // HEADER INFORMATION
  input.read<const int>(&numberParticles);
  input.read<const double>(&persistenceLength);
  input.read<const double>(&packingFraction);
  input.read<const double>(&systemSize);
  input.read<const double>(&torqueParameter);
  input.read<const int>(&randomSeed);
  input.read<const double>(&timeStep);
  input.read<const int>(&framesWork);
  input.read<const bool>(&dumpParticles);
  input.read<const int>(&dumpPeriod);

  // FILE PARTS LENGTHS
  headerLength = input.tellg();
  particleLength = 5*sizeof(double)*dumpParticles;
  frameLength = numberParticles*particleLength;
  workLength = 8*sizeof(double);

  // ESTIMATION OF NUMBER OF COMPUTED WORK AND ORDER PARAMETER SUMS AND FRAMES
  numberWork = (input.getFileSize() - headerLength - frameLength)/(
    framesWork*frameLength + workLength);
  frames = !dumpParticles ? 0 :
    (input.getFileSize() - headerLength - numberWork*workLength)/frameLength;

  // FILE CORRUPTION CHECK
  if ( input.getFileSize() !=
    headerLength + frames*frameLength + numberWork*workLength ) {
    throw std::invalid_argument("Invalid file size.");
  }

  // ACTIVE WORK AND ORDER PARAMETER
  if ( loadWork ) {
    double work;
    for (int i=0; i < numberWork; i++) {
      input.read<double>(&work,
        headerLength                     // header
        + frameLength                    // frame with index 0
        + (1 + i)*framesWork*frameLength // all following packs of framesWork frames
        + i*workLength);                 // previous values of the active work
      activeWork.push_back(work);
      input.read<double>(&work);
      activeWorkForce.push_back(work);
      input.read<double>(&work);
      activeWorkOri.push_back(work);
      input.read<double>(&work);
      orderParameter.push_back(work);
      input.read<double>(&work);
      orderParameter0.push_back(work);
      input.read<double>(&work);
      orderParameter1.push_back(work);
      input.read<double>(&work);
      torqueIntegral1.push_back(work);
      input.read<double>(&work);
      torqueIntegral2.push_back(work);
    }
  }
}

// DESTRUCTORS

Dat::~Dat() {}

// METHODS

int Dat::getNumberParticles() const { return numberParticles; }
double Dat::getPersistenceLength() const { return persistenceLength; }
double Dat::getPackingFraction() const { return packingFraction; }
double Dat::getSystemSize() const { return systemSize; }
double Dat::getTorqueParameter() const { return torqueParameter; }
int Dat::getRandomSeed() const { return randomSeed; }
double Dat::getTimeStep() const { return timeStep; }
int Dat::getFramesWork() const { return framesWork; }

long int Dat::getNumberWork() const { return numberWork; }
long int Dat::getFrames() const { return frames; }

std::vector<double> Dat::getActiveWork() { return activeWork; }
std::vector<double> Dat::getActiveWorkForce() { return activeWorkForce; }
std::vector<double> Dat::getActiveWorkOri() { return activeWorkOri; }
std::vector<double> Dat::getOrderParameter() { return orderParameter; }
std::vector<double> Dat::getOrderParameter0() { return orderParameter0; }
std::vector<double> Dat::getOrderParameter1() { return orderParameter1; }
std::vector<double> Dat::getTorqueIntegral1() { return torqueIntegral1; }
std::vector<double> Dat::getTorqueIntegral2() { return torqueIntegral2; }

double Dat::getPosition(
  int const& frame, int const& particle, int const& dimension) {
  // Returns position of a given particle at a given frame.

  return input.read<double>(
    headerLength                                     // header
    + frame*frameLength                              // other frames
    + particle*particleLength                        // other particles
    + (std::max(frame - 1, 0)/framesWork)*workLength // active work sums (taking into account the frame with index 0)
    + dimension*sizeof(double));                     // dimension
}

double Dat::getOrientation(int const& frame, int const& particle){
  // Returns orientation of a given particle at a given frame.

  return input.read<double>(
    headerLength                                     // header
    + frame*frameLength                              // other frames
    + particle*particleLength                        // other particles
    + (std::max(frame - 1, 0)/framesWork)*workLength // active work sums (taking into account the frame with index 0)
    + 2*sizeof(double));                             // positions
}

double Dat::getVelocity(
  int const& frame, int const& particle, int const& dimension) {
  // Returns velocity of a given particle at a given frame.

  return input.read<double>(
    headerLength                                     // header
    + frame*frameLength                              // other frames
    + particle*particleLength                        // other particles
    + (std::max(frame - 1, 0)/framesWork)*workLength // active work sums (taking into account the frame with index 0)
    + 3*sizeof(double)                               // positions and orientation
    + dimension*sizeof(double));                     // dimension
}


/********
 * DAT0 *
 ********/

// CONSTRUCTORS

Dat0::Dat0(std::string filename, bool loadWork) :
  numberParticles(), potentialParameter(), propulsionVelocity(),
    transDiffusivity(), rotDiffusivity(), persistenceLength(),
    packingFraction(), systemSize(), randomSeed(), timeStep(), framesWork(),
    dumpParticles(), dumpPeriod(),
  input(filename) {

  // HEADER INFORMATION
  input.read<const int>(&numberParticles);
  input.read<const double>(&potentialParameter);
  input.read<const double>(&propulsionVelocity);
  input.read<const double>(&transDiffusivity);
  input.read<const double>(&rotDiffusivity);
  input.read<const double>(&persistenceLength);
  input.read<const double>(&packingFraction);
  input.read<const double>(&systemSize);
  input.read<const int>(&randomSeed);
  input.read<const double>(&timeStep);
  input.read<const int>(&framesWork);
  input.read<const bool>(&dumpParticles);
  input.read<const int>(&dumpPeriod);

  // DIAMETERS
  double diameter;
  for (int i=0; i < numberParticles; i++) {
    input.read<double>(&diameter);
    diameters.push_back(diameter);
  }

  // FILE PARTS LENGTHS
  headerLength = input.tellg();
  particleLength = 9*sizeof(double)*dumpParticles;
  frameLength = numberParticles*particleLength;
  workLength = 4*sizeof(double);

  // ESTIMATION OF NUMBER OF COMPUTED WORK AND ORDER PARAMETER SUMS AND FRAMES
  numberWork = (input.getFileSize() - headerLength - frameLength)/(
    framesWork*frameLength + workLength);
  frames = !dumpParticles ? 0 :
    (input.getFileSize() - headerLength - numberWork*workLength)/frameLength;

  // FILE CORRUPTION CHECK
  if ( input.getFileSize() !=
    headerLength + frames*frameLength + numberWork*workLength ) {
    throw std::invalid_argument("Invalid file size.");
  }

  // ACTIVE WORK AND ORDER PARAMETER
  if ( loadWork ) {
    double work;
    for (int i=0; i < numberWork; i++) {
      input.read<double>(&work,
        headerLength                     // header
        + frameLength                    // frame with index 0
        + (1 + i)*framesWork*frameLength // all following packs of framesWork frames
        + i*workLength);                 // previous values of the active work
      activeWork.push_back(work);
      input.read<double>(&work);
      activeWorkForce.push_back(work);
      input.read<double>(&work);
      activeWorkOri.push_back(work);
      input.read<double>(&work);
      orderParameter.push_back(work);
    }
  }
}

// DESTRUCTORS

Dat0::~Dat0() {}

// METHODS

int Dat0::getNumberParticles() const { return numberParticles; }
double Dat0::getPotentialParameter() const { return potentialParameter; }
double Dat0::getPropulsionVelocity() const { return propulsionVelocity; }
double Dat0::getTransDiffusivity() const { return transDiffusivity; }
double Dat0::getRotDiffusivity() const { return rotDiffusivity; }
double Dat0::getPersistenceLength() const { return persistenceLength; }
double Dat0::getPackingFraction() const { return packingFraction; }
double Dat0::getSystemSize() const { return systemSize; }
int Dat0::getRandomSeed() const { return randomSeed; }
double Dat0::getTimeStep() const { return timeStep; }
int Dat0::getFramesWork() const { return framesWork; }

long int Dat0::getNumberWork() const { return numberWork; }
long int Dat0::getFrames() const { return frames; }

std::vector<double> Dat0::getDiameters() { return diameters; }

std::vector<double> Dat0::getActiveWork() { return activeWork; }
std::vector<double> Dat0::getActiveWorkForce() { return activeWorkForce; }
std::vector<double> Dat0::getActiveWorkOri() { return activeWorkOri; }
std::vector<double> Dat0::getOrderParameter() { return orderParameter; }

double Dat0::getPosition(
  int const& frame, int const& particle, int const& dimension,
  bool const& unfolded) {
  // Returns position of a given particle at a given frame.

  return input.read<double>(
    headerLength                                     // header
    + frame*frameLength                              // other frames
    + particle*particleLength                        // other particles
    + (std::max(frame - 1, 0)/framesWork)*workLength // active work sums (taking into account the frame with index 0)
    + 7*unfolded*sizeof(double)                      // unfolded positions
    + dimension*sizeof(double));                     // dimension
}

double Dat0::getOrientation(int const& frame, int const& particle){
  // Returns orientation of a given particle at a given frame.

  return input.read<double>(
    headerLength                                     // header
    + frame*frameLength                              // other frames
    + particle*particleLength                        // other particles
    + (std::max(frame - 1, 0)/framesWork)*workLength // active work sums (taking into account the frame with index 0)
    + 2*sizeof(double));                             // positions
}

double Dat0::getVelocity(
  int const& frame, int const& particle, int const& dimension) {
  // Returns velocity of a given particle at a given frame.

  return input.read<double>(
    headerLength                                     // header
    + frame*frameLength                              // other frames
    + particle*particleLength                        // other particles
    + (std::max(frame - 1, 0)/framesWork)*workLength // active work sums (taking into account the frame with index 0)
    + 3*sizeof(double)                               // positions and orientation
    + dimension*sizeof(double));                     // dimension
}

double Dat0::getPropulsion(
  int const& frame, int const& particle, int const& dimension) {
  // Returns self-propulsion vector of a given particle at a given frame.

  return input.read<double>(
    headerLength                                     // header
    + frame*frameLength                              // other frames
    + particle*particleLength                        // other particles
    + (std::max(frame - 1, 0)/framesWork)*workLength // active work sums (taking into account the frame with index 0)
    + 5*sizeof(double)                               // positions, orientation, and velocities
    + dimension*sizeof(double));                     // dimension
}


/********
 * DATN *
 ********/

// CONSTRUCTORS

DatN::DatN(std::string filename, bool loadWork) :
  numberParticles(), potentialParameter(), propulsionVelocity(),
    transDiffusivity(), rotDiffusivity(), persistenceLength(),
    packingFraction(), systemSize(), randomSeed(), timeStep(),
    init(), NLin(), NiterLin(), NLog(), frames(),
  input(filename) {

  // HEADER INFORMATION
  input.read<const int>(&numberParticles);
  input.read<const double>(&potentialParameter);
  input.read<const double>(&propulsionVelocity);
  input.read<const double>(&transDiffusivity);
  input.read<const double>(&rotDiffusivity);
  input.read<const double>(&persistenceLength);
  input.read<const double>(&packingFraction);
  input.read<const double>(&systemSize);
  input.read<const int>(&randomSeed);
  input.read<const double>(&timeStep);
  input.read<const int>(&init);
  input.read<const int>(&NLin);
  input.read<const int>(&NiterLin);
  input.read<const int>(&NLog);
  input.read<const int>(&frames);

  // FRAMES
  int frame;
  for (int i=0; i < frames; i++) {
    input.read<int>(&frame);
    frameIndices.push_back(frame);
  }

  // DIAMETERS
  double diameter;
  for (int i=0; i < numberParticles; i++) {
    input.read<double>(&diameter);
    diameters.push_back(diameter);
  }

  // FILE PARTS LENGTHS
  headerLength = input.tellg();
  particleLength = 9*sizeof(double);
  frameLength = numberParticles*particleLength;

  // FILE CORRUPTION CHECK
  if ( input.getFileSize() !=
    headerLength + frames*frameLength ) {
    throw std::invalid_argument("Invalid file size.");
  }
}

// DESTRUCTORS

DatN::~DatN() {}

// METHODS

int DatN::getNumberParticles() const { return numberParticles; }
double DatN::getPotentialParameter() const { return potentialParameter; }
double DatN::getPropulsionVelocity() const { return propulsionVelocity; }
double DatN::getTransDiffusivity() const { return transDiffusivity; }
double DatN::getRotDiffusivity() const { return rotDiffusivity; }
double DatN::getPersistenceLength() const { return persistenceLength; }
double DatN::getPackingFraction() const { return packingFraction; }
double DatN::getSystemSize() const { return systemSize; }
int DatN::getRandomSeed() const { return randomSeed; }
double DatN::getTimeStep() const { return timeStep; }

int DatN::getInit() const { return init; }
int DatN::getNLin() const { return NLin; }
int DatN::getNiterLin() const { return NiterLin; }
int DatN::getNLog() const {return NLog; }
std::vector<int>* DatN::getFrames() { return &frameIndices; }

std::vector<double> DatN::getDiameters() { return diameters; }

double DatN::getPosition(
  int const& frame, int const& particle, int const& dimension,
  bool const& unfolded) {
  // Returns position of a given particle at a given frame.

  return input.read<double>(
    headerLength                                     // header
    + frame*frameLength                              // other frames
    + particle*particleLength                        // other particles
    + 7*unfolded*sizeof(double)                      // unfolded positions
    + dimension*sizeof(double));                     // dimension
}

double DatN::getOrientation(int const& frame, int const& particle){
  // Returns orientation of a given particle at a given frame.

  return input.read<double>(
    headerLength                                     // header
    + frame*frameLength                              // other frames
    + particle*particleLength                        // other particles
    + 2*sizeof(double));                             // positions
}

double DatN::getVelocity(
  int const& frame, int const& particle, int const& dimension) {
  // Returns velocity of a given particle at a given frame.

  return input.read<double>(
    headerLength                                     // header
    + frame*frameLength                              // other frames
    + particle*particleLength                        // other particles
    + 3*sizeof(double)                               // positions and orientation
    + dimension*sizeof(double));                     // dimension
}

double DatN::getPropulsion(
  int const& frame, int const& particle, int const& dimension) {
  // Returns self-propulsion vector of a given particle at a given frame.

  return input.read<double>(
    headerLength                                     // header
    + frame*frameLength                              // other frames
    + particle*particleLength                        // other particles
    + 5*sizeof(double)                               // positions, orientation, and velocities
    + dimension*sizeof(double));                     // dimension
}


/********
 * DATR *
 ********/

// CONSTRUCTORS

DatR::DatR(std::string filename, bool loadOrder) :
  numberParticles(), rotDiffusivity(), torqueParameter(), timeStep(),
    framesOrder(), dumpRotors(), dumpPeriod(), randomSeed(),
  input(filename) {

  // HEADER INFORMATION
  input.read<const int>(&numberParticles);
  input.read<const double>(&rotDiffusivity);
  input.read<const double>(&torqueParameter);
  input.read<const double>(&timeStep);
  input.read<const int>(&framesOrder);
  input.read<const bool>(&dumpRotors);
  input.read<const int>(&dumpPeriod);
  input.read<const int>(&randomSeed);

  // FILE PARTS LENGTHS
  headerLength = input.tellg();
  rotorLength = 1*sizeof(double);
  frameLength = numberParticles*rotorLength;
  orderLength = 2*sizeof(double);

  // ESTIMATION OF NUMBER OF COMPUTED ORDER PARAMETER SUMS AND FRAMES
  numberOrder = (input.getFileSize() - headerLength - frameLength)/(
    framesOrder*frameLength + orderLength);
  frames = !dumpRotors ? 0 :
    (input.getFileSize() - headerLength - numberOrder*orderLength)/frameLength;

  // FILE CORRUTION CHECK
  if ( input.getFileSize() !=
    headerLength + frames*frameLength + numberOrder*orderLength ) {
    throw std::invalid_argument("Invalid file size.");
  }

  // ORDER PARAMETER
  if ( loadOrder ) {
    double order;
    for (int i=0; i < numberOrder; i++) {
      input.read<double>(&order,
        headerLength                      // header
        + frameLength                     // frame with index 0
        + (1 + i)*framesOrder*frameLength // all following packs of framesOrder frames
        + i*orderLength);                 // previous values of the order parameter
      orderParameter.push_back(order);
      input.read<double>(&order);
      orderParameterSq.push_back(order);
    }
  }
}

// DESTRUCTORS

DatR::~DatR() {}

// METHODS

int DatR::getNumberParticles() const { return numberParticles; }
double DatR::getRotDiffusivity() const { return rotDiffusivity; }
double DatR::getTorqueParameter() const { return torqueParameter; }
double DatR::getTimeStep() const { return timeStep; }
int DatR::getDumpPeriod() const { return dumpPeriod; }
int DatR::getRandomSeed() const { return randomSeed; }

long int DatR::getFrames() const { return frames; }

std::vector<double> DatR::getOrderParameter() { return orderParameter; }
std::vector<double> DatR::getOrderParameterSq() { return orderParameterSq; }

double DatR::getOrientation(int const& frame, int const& rotor) {
  // Returns position of a given rotor at a given frame.

  return input.read<double>(
    headerLength                                         // header
    + frame*frameLength                                  // other frames
    + rotor*rotorLength                                  // other rotors
    + (std::max(frame - 1, 0)/framesOrder)*orderLength); // order parameter sums (taking into account the frame with index 0)
}

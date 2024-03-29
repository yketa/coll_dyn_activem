#include <cmath>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>
#include <assert.h>

#include "dat.hpp"
#include "particle.hpp"
#include "maths.hpp"

/////////////
// CLASSES //
/////////////

/************
 * PARTICLE *
 ************/

// CONSTRUCTORS

Particle::Particle(const double& d) :
  r{0, 0}, c{0, 0}, theta(0), p{0, 0}, v{0, 0}, sigma(d),
  f{0, 0}, fp{0, 0}, gamma(0) {}
Particle::Particle(
  const double& x, const double& y,
  const double& ang, const double& px, const double& py,
  const double& d) :
  r{x, y}, c{0, 0}, theta(ang), p{px, py}, v{0, 0}, sigma(d),
  f{0, 0}, fp{0, 0}, gamma(0) {}
Particle::Particle(Particle* particle) :
  r{particle->position()[0], particle->position()[1]},
  c{particle->cross()[0], particle->cross()[1]},
  theta(particle->orientation()[0]),
  p{particle->propulsion()[0], particle->propulsion()[1]},
  v{particle->velocity()[0] , particle->velocity()[1]},
  sigma(particle->diameter()),
  f{particle->force()[0], particle->force()[1]},
  fp{particle->forcep()[0], particle->forcep()[1]},
  gamma(particle->torque()[0]) {}

// METHODS

double* Particle::position() { return &r[0]; } // returns pointer to position
long int* Particle::cross() { return &c[0]; } // returns pointer to number of boundary crossings
double* Particle::orientation() { return &theta; } // returns pointer to orientation
double* Particle::propulsion() { return &p[0]; } // returns pointer to self-propulsion direction
double* Particle::velocity() { return &v[0]; } // returns pointer to velocity

double Particle::diameter() const { return sigma; } // returns pointer to diameter

double* Particle::force() { return &f[0]; }; // returns pointer to force

double* Particle::forcep() { return &fp[0]; }; // returns pointer to force applied on self-propulsion (AOUP)
double* Particle::torque() { return &gamma; } // returns pointer to aligning torque (ABP)


/************
 * CELLLIST *
 ************/

#ifdef _OPENMP
int CellList::tid = 0;
int CellList::numth = 0;
#endif


/**************
 * PARAMETERS *
 **************/

// CONSTRUCTORS

Parameters::Parameters() :
  numberParticles(0), potentialParameter(0), propulsionVelocity(0),
    transDiffusivity(0), rotDiffusivity(0),
    persistenceLength(0),
    packingFraction(0),
    confiningLength(0),
    systemSize(0),
    torqueParameter(0),
    timeStep(0) {}

Parameters::Parameters(int N, double lp, double phi, double dt, double g) :
  numberParticles(N), potentialParameter(1.0), propulsionVelocity(1.0),
    transDiffusivity(1.0/(3.0*lp)), rotDiffusivity(1.0/lp),
    persistenceLength(lp),
    packingFraction(phi),
    confiningLength(0),
    systemSize(getL_WCA(phi, N, 1.0)),
    torqueParameter(g),
    timeStep(dt) {}

Parameters::Parameters(
  int N, double epsilon, double v0, double D, double Dr,
    double phi, std::vector<double> const& diameters,
  double dt, bool conf) :
  numberParticles(N), potentialParameter(epsilon), propulsionVelocity(v0),
    transDiffusivity(D), rotDiffusivity(Dr),
    persistenceLength(v0/Dr),
    packingFraction(phi),
    confiningLength(
      conf
        ? getLc_WCA(phi, diameters)
        : 0),
    systemSize(
      conf
        ? confiningLength +
          2*pow(2, 1/6)*(*std::max_element(diameters.begin(), diameters.end()))
        : getL_WCA(phi, diameters)),
    torqueParameter(0),
    timeStep(dt) {}

Parameters::Parameters(
  int N, double epsilon, double v0, double D, double Dr, double lp,
    double phi, std::vector<double> const& diameters, double L,
  double dt, double Lc) :
  numberParticles(N), potentialParameter(epsilon), propulsionVelocity(v0),
    transDiffusivity(D), rotDiffusivity(Dr),
    persistenceLength(lp),
    packingFraction(phi),
    confiningLength(Lc),
    systemSize(L),
    torqueParameter(0),
    timeStep(dt) {}

Parameters::Parameters(Parameters const& parameters) :
  numberParticles(parameters.getNumberParticles()),
  potentialParameter(parameters.getPotentialParameter()),
  propulsionVelocity(parameters.getPropulsionVelocity()),
  transDiffusivity(parameters.getTransDiffusivity()),
  rotDiffusivity(parameters.getRotDiffusivity()),
  persistenceLength(parameters.getPersistenceLength()),
  packingFraction(parameters.getPackingFraction()),
  confiningLength(parameters.getConfiningLength()),
  systemSize(parameters.getSystemSize()),
  torqueParameter(parameters.getTorqueParameter()),
  timeStep(parameters.getTimeStep()) {}

Parameters::Parameters(Parameters* parameters) :
  numberParticles(parameters->getNumberParticles()),
  potentialParameter(parameters->getPotentialParameter()),
  propulsionVelocity(parameters->getPropulsionVelocity()),
  transDiffusivity(parameters->getTransDiffusivity()),
  rotDiffusivity(parameters->getRotDiffusivity()),
  persistenceLength(parameters->getPersistenceLength()),
  packingFraction(parameters->getPackingFraction()),
  confiningLength(parameters->getConfiningLength()),
  systemSize(parameters->getSystemSize()),
  torqueParameter(parameters->getTorqueParameter()),
  timeStep(parameters->getTimeStep()) {}

// METHODS

int Parameters::getNumberParticles() const { return numberParticles; }
double Parameters::getPotentialParameter() const { return potentialParameter; }
double Parameters::getPropulsionVelocity() const { return propulsionVelocity; }
double Parameters::getTransDiffusivity() const { return transDiffusivity; }
double Parameters::getRotDiffusivity() const { return rotDiffusivity; }
double Parameters::getPersistenceLength() const { return persistenceLength; }
double Parameters::getPackingFraction() const {return packingFraction; }
double Parameters::getConfiningLength() const { return confiningLength; }
double Parameters::getSystemSize() const { return systemSize; }
double Parameters::getTorqueParameter() const { return torqueParameter; }
double Parameters::getTimeStep() const { return timeStep; }


/**********
 * SYSTEM *
 **********/

// CONSTRUCTORS

System::System() :
  param(new Parameters()),
  randomSeed(0), randomGenerator(),
  particles(0),
  cellList(),
  output(""), velocitiesDumps(),
  framesWork(0), dumpParticles(0), dumpPeriod(0),
  torqueParameter(0),
  biasingParameter(0),
  dumpFrame(-1),
  workSum {0, 0, 0}, workForceSum {0, 0, 0}, workOrientationSum {0, 0, 0},
    orderSum {0, 0, 0}, order0Sum {0, 0, 0}, order1Sum {0, 0, 0},
    torqueIntegral1 {0, 0, 0}, torqueIntegral2 {0, 0, 0}
  {}

System::System(
  Parameters* parameters, int seed, std::string filename,
  int nWork, bool dump, int period) :
  param(parameters),
  randomSeed(seed), randomGenerator(randomSeed),
  particles(parameters->getNumberParticles()),
  cellList(),
  output(filename), velocitiesDumps(parameters->getNumberParticles()),
  framesWork(nWork > 0 ? nWork : (int)
    parameters->getPersistenceLength()/(parameters->getTimeStep()*period)),
    dumpParticles(dump), dumpPeriod(period),
  torqueParameter(parameters->getTorqueParameter()),
  biasingParameter(0),
  dumpFrame(-1),
  workSum {0, 0, 0}, workForceSum {0, 0, 0}, workOrientationSum {0, 0, 0},
    orderSum {0, 0, 0}, order0Sum {0, 0, 0}, order1Sum {0, 0, 0},
    torqueIntegral1 {0, 0, 0}, torqueIntegral2 {0, 0, 0}
  {

  // write header with system parameters to output file
  output.write<int>(getNumberParticles());
  output.write<double>(getPersistenceLength());
  output.write<double>(getPackingFraction());
  output.write<double>(getSystemSize());
  output.write<double>(getTorqueParameter());
  output.write<int>(randomSeed);
  output.write<double>(getTimeStep());
  output.write<int>(framesWork);
  output.write<bool>(dumpParticles);
  output.write<int>(dumpPeriod);
  output.close();

  // put particles on a grid with random orientation
  int gridSize = ceil(sqrt(getNumberParticles())); // size of the grid on which to put the particles
  double gridSpacing = getSystemSize()/gridSize;
  for (int i=0; i < getNumberParticles(); i++) { // loop over particles
    // position on the grid
    particles[i].position()[0] = ((i%gridSize) + 0.5*(i/gridSize))*gridSpacing;
    particles[i].position()[1] = (i/gridSize)*gridSpacing;
    // random orientation
    particles[i].orientation()[0] = 2*M_PI*randomGenerator.random01();
  }

  // initialise cell list
  initialiseCellList();
}

System::System(
  System* system, int seed, std::string filename,
  int nWork, bool dump, int period) :
  param(system->getParameters()),
  randomSeed(seed), randomGenerator(randomSeed),
  particles(system->getNumberParticles()),
  cellList(),
  output(filename), velocitiesDumps(system->getNumberParticles()),
  framesWork(nWork > 0 ? nWork : (int)
    system->getPersistenceLength()/(system->getTimeStep()*period)),
    dumpParticles(dump), dumpPeriod(period),
  torqueParameter((system->getParameters())->getTorqueParameter()),
  biasingParameter(0),
  dumpFrame(-1),
  workSum {0, 0, 0}, workForceSum {0, 0, 0}, workOrientationSum {0, 0, 0},
    orderSum {0, 0, 0}, order0Sum {0, 0, 0}, order1Sum {0, 0, 0},
    torqueIntegral1 {0, 0, 0}, torqueIntegral2 {0, 0, 0}
  {

  // write header with system parameters to output file
  output.write<int>(getNumberParticles());
  output.write<double>(getPersistenceLength());
  output.write<double>(getPackingFraction());
  output.write<double>(getSystemSize());
  output.write<double>(getTorqueParameter());
  output.write<int>(randomSeed);
  output.write<double>(getTimeStep());
  output.write<int>(framesWork);
  output.write<bool>(dumpParticles);
  output.write<int>(dumpPeriod);
  output.close();

  // initialise cell list
  initialiseCellList();
  // copy positions and orientations and update cell list
  copyState(system);
  // copy dumps
  copyDump(system);
}

System::System(
  std::string inputFilename, int inputFrame, double dt,
  int seed, std::string filename,
  int nWork, bool dump, int period) :
  param(Dat(inputFilename, false), dt),
  randomSeed(seed), randomGenerator(randomSeed),
  particles(0),
  cellList(),
  output(filename), velocitiesDumps(0),
  framesWork(nWork > 0 ? nWork : (int)
    getPersistenceLength()/(getTimeStep()*period)),
    dumpParticles(dump), dumpPeriod(period),
  torqueParameter(0),
  biasingParameter(0),
  dumpFrame(-1),
  workSum {0, 0, 0}, workForceSum {0, 0, 0}, workOrientationSum {0, 0, 0},
    orderSum {0, 0, 0}, order0Sum {0, 0, 0}, order1Sum {0, 0, 0},
    torqueIntegral1 {0, 0, 0}, torqueIntegral2 {0, 0, 0}
  {

  // load data
  Dat inputDat(inputFilename, false); // data object

  // set torque parameter
  double g = inputDat.getTorqueParameter();
  setTorqueParameter(g);

  // resize vector of particles
  particles.resize(getNumberParticles());

  // resize velocity dumps
  velocitiesDumps.resize(getNumberParticles()); // resize vector of locations of velocity dumps

  // set positions and orientations
  for (int i=0; i < getNumberParticles(); i++) {
    // positions
    for (int dim=0; dim < 2; dim++) {
      particles[i].position()[dim] = inputDat.getPosition(inputFrame, i, dim);
    }
    // orientations
    particles[i].orientation()[0] = inputDat.getOrientation(inputFrame, i);
  }

  // write header with system parameters to output file
  output.write<int>(getNumberParticles());
  output.write<double>(getPersistenceLength());
  output.write<double>(getPackingFraction());
  output.write<double>(getSystemSize());
  output.write<double>(getTorqueParameter());
  output.write<int>(randomSeed);
  output.write<double>(getTimeStep());
  output.write<int>(framesWork);
  output.write<bool>(dumpParticles);
  output.write<int>(dumpPeriod);
  output.close();

  // initialise cell list
  initialiseCellList();
}

// DESTRUCTORS

System::~System() {}

// METHODS

Parameters* System::getParameters() { return &param; }
std::vector<double> System::getDiameters() const {
  return std::vector<double> (getNumberParticles(), 1.0);
}

int System::getNumberParticles() const {
  return param.getNumberParticles(); }
double System::getPersistenceLength() const {
  return param.getPersistenceLength(); }
double System::getPackingFraction() const {
  return param.getPackingFraction(); }
double System::getSystemSize() const {
  return param.getSystemSize(); }
double System::getTimeStep() const {
  return param.getTimeStep(); }

int System::getRandomSeed() const { return randomSeed; }
Random* System::getRandomGenerator() { return &randomGenerator; }
void System::setGenerator(RNDG rndeng) {
  randomGenerator.setGenerator(rndeng);
}

Particle* System::getParticle(int const& index) { return &(particles[index]); }
std::vector<Particle> System::getParticles() { return particles; }

CellList* System::getCellList() { return &cellList; }
void System::initialiseCellList(double const& cutOff) {
  const std::vector<double> diameters = getDiameters();
  cellList = CellList(
    getNumberParticles(), getSystemSize(),
    cutOff*(*std::max_element(diameters.begin(), diameters.end())));
  updateCellList();
}
void System::updateCellList() {
  std::vector<double*> positions(0);
  for (int i=0; i < getNumberParticles(); i++) {
    positions.push_back(particles[i].position());
  }
  cellList.listConstructor<double*>(positions);
}

void System::flushOutputFile() { output.flush(); }
std::string System::getOutputFile() const { return output.getOutputFile(); }

void System::setTorqueParameter(double& g) { torqueParameter = g; }
double System::getTorqueParameter() { return torqueParameter; }

double System::getBiasingParameter() { return biasingParameter; }
void System::setBiasingParameter(double s) { biasingParameter = s; }

int* System::getDump() { return &dumpFrame; }

void System::resetDump() {
  // Reset time-extensive quantities over trajectory.

  dumpFrame = 0;

  workSum[0] = 0;
  workForceSum[0] = 0;
  workOrientationSum[0] = 0;
  orderSum[0] = 0;
  order0Sum[0] = 0;
  order1Sum[0] = 0;
  torqueIntegral1[0] = 0;
  torqueIntegral2[0] = 0;

  workSum[2] = 0;
  workForceSum[2] = 0;
  workOrientationSum[2] = 0;
  orderSum[2] = 0;
  order0Sum[2] = 0;
  order1Sum[2] = 0;
  torqueIntegral1[2] = 0;
  torqueIntegral2[2] = 0;
}

void System::copyDump(System* system) {
  // Copy dumps from other system.
  // WARNING: This also copies the index of last frame dumped. Consistency
  //          has to be checked.

  dumpFrame = system->getDump()[0];

  workSum[2] = system->getTotalWork()[0];
  workForceSum[2] = system->getTotalWorkForce()[0];
  workOrientationSum[2] = system->getTotalWorkOrientation()[0];
  orderSum[2] = system->getTotalOrder()[0];
  order0Sum[2] = system->getTotalOrder0()[0];
  order1Sum[2] = system->getTotalOrder1()[0];
  torqueIntegral1[2] = system->getTotalTorqueIntegral1()[0];
  torqueIntegral2[2] = system->getTotalTorqueIntegral2()[0];
}

double System::getWork() { return workSum[1]; }
double System::getWorkForce() { return workForceSum[1]; }
double System::getWorkOrientation() { return workOrientationSum[1]; }
double System::getOrder() { return orderSum[1]; }
double System::getOrder0() { return order0Sum[1]; }
double System::getOrder1() { return order1Sum[1]; }
double System::getTorqueIntegral1() { return torqueIntegral1[1]; }
double System::getTorqueIntegral2() { return torqueIntegral2[1]; }

double* System::getTotalWork() { return &(workSum[2]); }
double* System::getTotalWorkForce() { return &(workForceSum[2]); }
double* System::getTotalWorkOrientation() { return &(workOrientationSum[2]); }
double* System::getTotalOrder() { return &(orderSum[2]); }
double* System::getTotalOrder0() { return &(order0Sum[2]); }
double* System::getTotalOrder1() { return &(order1Sum[2]); }
double* System::getTotalTorqueIntegral1() { return &(torqueIntegral1[2]); }
double* System::getTotalTorqueIntegral2() { return &(torqueIntegral2[2]); }

void System::copyState(std::vector<Particle>& newParticles) {
  // Copy positions and propulsions.

  for (int i=0; i < getNumberParticles(); i++) {
    for (int dim=0; dim < 2; dim++) {
      // POSITIONS
      particles[i].position()[dim] = newParticles[i].position()[dim];
    }
    // ORIENTATIONS
    particles[i].orientation()[0] = newParticles[i].orientation()[0];
    for (int dim=0; dim < 2; dim++) {
      // SELF-PROPULSION VECTORS
      particles[i].propulsion()[dim] = newParticles[i].propulsion()[dim];
    }
  }

  // UPDATING CELL LIST
  updateCellList();
}

void System::copyState(System* system) {
  // Copy positions and propulsions.

  for (int i=0; i < getNumberParticles(); i++) {
    for (int dim=0; dim < 2; dim++) {
      // POSITIONS
      particles[i].position()[dim] = (system->getParticle(i))->position()[dim];
    }
    // ORIENTATIONS
    particles[i].orientation()[0] = (system->getParticle(i))->orientation()[0];
    for (int dim=0; dim < 2; dim++) {
      // SELF-PROPULSION VECTORS
      particles[i].propulsion()[dim] =
        (system->getParticle(i))->propulsion()[dim];
    }
  }

  // UPDATING CELL LIST
  updateCellList();
}

void System::saveInitialState() {
  // Saves initial state of particles to output file.

  // output
  if ( dumpParticles ) {
    output.open();

    for (int i=0; i < getNumberParticles(); i++) { // output all particles
      // POSITIONS
      for (int dim=0; dim < 2; dim++) { // output position in each dimension
        output.write<double>(particles[i].position()[dim]);
      }
      // ORIENTATIONS
      output.write<double>(particles[i].orientation()[0]); // output orientation
      // VELOCITIES
      velocitiesDumps[i] = output.tellp(); // location to dump velocities at next time step
      for (int dim=0; dim < 2; dim++) { // output velocity in each dimension
        output.write<double>(0.0); // zero by default for initial frame
      }
    }

    output.close();
  }

  // reset dump
  resetDump();
}

void System::saveNewState(std::vector<Particle>& newParticles) {
  // Saves new state of particles to output file then copy it.

  int cross;

  // DUMP FRAME
  dumpFrame++;

  ////////////
  // SAVING //
  ////////////

  // OPEN FILE
  if ( ( dumpParticles && dumpFrame % dumpPeriod == 0 ) ||
    ( dumpParticles && (dumpFrame - 1) % dumpPeriod == 0 ) ||
    ( dumpFrame % (framesWork*dumpPeriod) == 0 ) ) {
      output.open();
  }

  for (int i=0; i < getNumberParticles(); i++) { // output all particles

    // ACTIVE WORK and ORDER PARAMETER (computation)
    for (int dim=0; dim < 2; dim++) {
      // active work
      workSum[0] +=
        (cos(newParticles[i].orientation()[0] - dim*M_PI/2)
          + cos(particles[i].orientation()[0] - dim*M_PI/2))
        *(newParticles[i].position()[dim] - particles[i].position()[dim]) // NOTE: at this stage, newParticles[i].position() are not rewrapped, so this difference is the actual displacement
        /2;
      // force part of the active work
      workForceSum[0] +=
        (cos(newParticles[i].orientation()[0] - dim*M_PI/2)
          + cos(particles[i].orientation()[0] - dim*M_PI/2))
        *getTimeStep()*particles[i].force()[dim]/3/getPersistenceLength()
        /2;
      // orientation part of the active work
      workOrientationSum[0] +=
        (cos(newParticles[i].orientation()[0] - dim*M_PI/2)
          + cos(particles[i].orientation()[0] - dim*M_PI/2))
        *getTimeStep()*cos(particles[i].orientation()[0] - dim*M_PI/2)
        /2;
    }

    // COORDINATES
    for (int dim=0; dim < 2; dim++) {
      // compute crossings
      cross = wrapCoordinate<System>(this, newParticles[i].position()[dim]);
      particles[i].cross()[dim] += cross;
      // keep particles in the box
      newParticles[i].position()[dim] -= cross*getSystemSize();
    }
    if ( dumpParticles && dumpFrame % dumpPeriod == 0 ) {
      // output wrapped position in each dimension
      for (int dim=0; dim < 2; dim++) {
        output.write<double>(newParticles[i].position()[dim]);
      }
      // output unwrapped position in each dimension
      // for (int dim=0; dim < 2; dim++) {
      //   output.write<double>(newParticles[i].position()[dim]
      //     + particles[i].cross()[dim]*getSystemSize());
      // }
    }

    if ( dumpParticles && (dumpFrame - 1) % dumpPeriod == 0 ) {

      // VELOCITIES
      for (int dim=0; dim < 2; dim++) {
        output.write<double>(
          particles[i].velocity()[dim],
          velocitiesDumps[i] + dim*sizeof(double));
      }
    }

    if ( dumpParticles && dumpFrame % dumpPeriod == 0 ) {

      // ORIENTATION
      output.write<double>(newParticles[i].orientation()[0]);

      // VELOCITIES
      velocitiesDumps[i] = output.tellp(); // location to dump velocities at next time step
      for (int dim=0; dim < 2; dim++) {
        output.write<double>(0.0); // zero by default until rewrite at next time step
      }
    }


  }

  // ORDER PARAMETER
  std::vector<double> orderOld = getOrderParameter(particles);
  double orderNormSqOld = pow(orderOld[0], 2) + pow(orderOld[1], 2);
  std::vector<double> orderNew = getOrderParameter(newParticles);
  double orderNormSqNew = pow(orderNew[0], 2) + pow(orderNew[1], 2);
  orderSum[0] += (sqrt(orderNormSqOld) + sqrt(orderNormSqNew))/2;
  order0Sum[0] += (orderOld[0] + orderNew[0])/2;
  order1Sum[0] += (orderOld[1] + orderNew[1])/2;
  // GLOBAL PHASE
  double globalPhaseOld = getGlobalPhase(particles);
  double globalPhaseNew = getGlobalPhase(newParticles);

  // FIRST TORQUE INTEGRAL
  torqueIntegral1[0] += (orderNormSqOld + orderNormSqNew)/2;
  // SECOND TORQUE INTEGRAL
  for (int i=0; i < getNumberParticles(); i++) {
    #if 1 // method with global phase
    torqueIntegral2[0] += orderNormSqOld
      *pow(sin(particles[i].orientation()[0] - globalPhaseOld), 2)/2.0;
    torqueIntegral2[0] += orderNormSqNew
      *pow(sin(newParticles[i].orientation()[0] - globalPhaseNew), 2)/2.0;
    #else // method with double sum
    double sumSinOld (0.0), sumSinNew (0.0);
    for (int j=0; j < getNumberParticles(); j++) {
      sumSinOld += sin(
        particles[i].orientation()[0] - particles[j].orientation()[0]);
      sumSinNew += sin(
        newParticles[i].orientation()[0] - newParticles[j].orientation()[0]);
    }
    torqueIntegral2[0] += (pow(sumSinOld, 2) + pow(sumSinNew, 2))
      /2.0/pow(getNumberParticles(), 2);
    #endif
  }

  // ACTIVE WORK and ORDER PARAMETER (output)
  if ( dumpFrame % (framesWork*dumpPeriod) == 0 ) {
    // compute normalised rates since last dump
    workSum[1] = workSum[0]/(
      getNumberParticles()*getTimeStep()*framesWork*dumpPeriod);
    workForceSum[1] = workForceSum[0]/(
      getNumberParticles()*getTimeStep()*framesWork*dumpPeriod);
    workOrientationSum[1] = workOrientationSum[0]/(
      getNumberParticles()*getTimeStep()*framesWork*dumpPeriod);
    orderSum[1] = orderSum[0]/(
      framesWork*dumpPeriod);
    order0Sum[1] = order0Sum[0]/(
      framesWork*dumpPeriod);
    order1Sum[1] = order1Sum[0]/(
      framesWork*dumpPeriod);
    torqueIntegral1[1] = torqueIntegral1[0]/(
      framesWork*dumpPeriod);
    torqueIntegral2[1] = torqueIntegral2[0]/(
      getNumberParticles()*framesWork*dumpPeriod);
    // output normalised rates
    output.write<double>(workSum[1]);
    output.write<double>(workForceSum[1]);
    output.write<double>(workOrientationSum[1]);
    output.write<double>(orderSum[1]);
    output.write<double>(order0Sum[1]);
    output.write<double>(order1Sum[1]);
    output.write<double>(torqueIntegral1[1]);
    output.write<double>(torqueIntegral2[1]);
    // update time extensive quantities over trajectory since last reset
    workSum[2] += workSum[0]/getNumberParticles();
    workForceSum[2] += workForceSum[0]/getNumberParticles();
    workOrientationSum[2] += workOrientationSum[0]/getNumberParticles();
    orderSum[2] += orderSum[0];
    order0Sum[2] += order0Sum[0];
    order1Sum[2] += order1Sum[0];
    torqueIntegral1[2] += torqueIntegral1[0];
    torqueIntegral2[2] += torqueIntegral2[0]/getNumberParticles();
    // reset sums
    workSum[0] = 0;
    workForceSum[0] = 0;
    workOrientationSum[0] = 0;
    orderSum[0] = 0;
    order0Sum[0] = 0;
    order1Sum[0] = 0;
    torqueIntegral1[0] = 0;
    torqueIntegral2[0] = 0;
  }

  // CLOSE FILE
  if ( output.is_open() ) { output.close(); }

  /////////////
  // COPYING //
  /////////////

  copyState(newParticles);
}


/***********
 * SYSTEM0 *
 ***********/

// CONSTRUCTORS

System0::System0() :
  param(new Parameters()),
  randomSeed(0), randomGenerator(),
  particles(0),
  cellList(),
  output(""), velocitiesDumps(),
  framesWork(0), dumpParticles(0), dumpPeriod(0),
  dumpFrame(-1),
  workSum {0, 0, 0}, workForceSum {0, 0, 0}, workOrientationSum {0, 0, 0},
    orderSum {0, 0, 0} {}

System0::System0(
  Parameters* parameters, int seed, std::string filename,
  int nWork, bool dump, int period) :
  param(parameters),
  randomSeed(seed), randomGenerator(randomSeed),
  particles(0),
  cellList(),
  output(filename), velocitiesDumps(parameters->getNumberParticles()),
  framesWork(nWork > 0 ? nWork : (int)
    1/(parameters->getRotDiffusivity()*parameters->getTimeStep()*period)),
    dumpParticles(dump), dumpPeriod(period),
  dumpFrame(-1),
  workSum {0, 0, 0}, workForceSum {0, 0, 0}, workOrientationSum {0, 0, 0},
    orderSum {0, 0, 0} {

  // set diameters
  for (int i=0; i < getNumberParticles(); i++) {
    particles.push_back(Particle(1.0));
  }

  // write header with system parameters to output file
  output.write<int>(getNumberParticles());
  output.write<double>(getPotentialParameter());
  output.write<double>(getPropulsionVelocity());
  output.write<double>(getTransDiffusivity());
  output.write<double>(getRotDiffusivity());
  output.write<double>(getPersistenceLength());
  output.write<double>(getPackingFraction());
  output.write<double>(getSystemSize());
  output.write<int>(randomSeed);
  output.write<double>(getTimeStep());
  output.write<int>(framesWork);
  output.write<bool>(dumpParticles);
  output.write<int>(dumpPeriod);

  // write particles' diameters
  for (int i=0; i < getNumberParticles(); i++) {
    output.write<double>(getParticle(i)->diameter());
  }

  // put particles on a grid with random orientation
  int gridSize = ceil(sqrt(getNumberParticles())); // size of the grid on which to put the particles
  double gridSpacing = getSystemSize()/gridSize;
  for (int i=0; i < getNumberParticles(); i++) { // loop over particles
    // position on the grid
    particles[i].position()[0] = ((i%gridSize) + 0.5*(i/gridSize))*gridSpacing;
    particles[i].position()[1] = (i/gridSize)*gridSpacing;
    // random orientation
    particles[i].orientation()[0] = 2*M_PI*randomGenerator.random01();
    // self-propulsion vector
    particles[i].propulsion()[0] =
      getPropulsionVelocity()*cos(particles[i].orientation()[0]);
    particles[i].propulsion()[1] =
      getPropulsionVelocity()*sin(particles[i].orientation()[0]);
  }

  // initialise cell list
  initialiseCellList();
}

System0::System0(
  Parameters* parameters, std::vector<double>& diameters, int seed,
  std::string filename, int nWork, bool dump, int period) :
  param(parameters),
  randomSeed(seed), randomGenerator(randomSeed),
  particles(0),
  cellList(),
  output(filename), velocitiesDumps(parameters->getNumberParticles()),
  framesWork(nWork > 0 ? nWork : (int)
    1/(parameters->getRotDiffusivity()*parameters->getTimeStep()*period)),
    dumpParticles(dump), dumpPeriod(period),
  dumpFrame(-1),
  workSum {0, 0, 0}, workForceSum {0, 0, 0}, workOrientationSum {0, 0, 0},
    orderSum {0, 0, 0} {

  // set diameters
  for (int i=0; i < getNumberParticles(); i++) {
    particles.push_back(Particle(diameters[i]));
  }

  // write header with system parameters to output file
  output.write<int>(getNumberParticles());
  output.write<double>(getPotentialParameter());
  output.write<double>(getPropulsionVelocity());
  output.write<double>(getTransDiffusivity());
  output.write<double>(getRotDiffusivity());
  output.write<double>(getPersistenceLength());
  output.write<double>(getPackingFraction());
  output.write<double>(getSystemSize());
  output.write<int>(randomSeed);
  output.write<double>(getTimeStep());
  output.write<int>(framesWork);
  output.write<bool>(dumpParticles);
  output.write<int>(dumpPeriod);

  // write particles' diameters
  for (int i=0; i < parameters->getNumberParticles(); i++) {
    output.write<double>(getParticle(i)->diameter());
  }

  // put particles on a grid with random orientation
  int gridSize = ceil(sqrt(getNumberParticles())); // size of the grid on which to put the particles
  double gridSpacing = getSystemSize()/gridSize;
  for (int i=0; i < getNumberParticles(); i++) { // loop over particles
    // position on the grid
    particles[i].position()[0] = ((i%gridSize) + 0.5*(i/gridSize))*gridSpacing;
    particles[i].position()[1] = (i/gridSize)*gridSpacing;
    // random orientation
    particles[i].orientation()[0] = 2*M_PI*randomGenerator.random01();
    // self-propulsion vector
    particles[i].propulsion()[0] =
      getPropulsionVelocity()*cos(particles[i].orientation()[0]);
    particles[i].propulsion()[1] =
      getPropulsionVelocity()*sin(particles[i].orientation()[0]);
  }

  // initialise cell list
  initialiseCellList();
}

System0::System0(
  System0* system, int seed, std::string filename,
  int nWork, bool dump, int period) :
  param(system->getParameters()),
  randomSeed(seed), randomGenerator(randomSeed),
  particles(0),
  cellList(),
  output(filename), velocitiesDumps(system->getNumberParticles()),
  framesWork(nWork > 0 ? nWork : (int)
    1/(system->getRotDiffusivity()*system->getTimeStep()*period)),
    dumpParticles(dump), dumpPeriod(period),
  dumpFrame(-1),
  workSum {0, 0, 0}, workForceSum {0, 0, 0}, workOrientationSum {0, 0, 0},
    orderSum {0, 0, 0} {

  // set diameters
  std::vector<double> diameters = system->getDiameters();
  for (int i=0; i < getNumberParticles(); i++) {
    particles.push_back(Particle(diameters[i]));
  }

  // write header with system parameters to output file
  output.write<int>(getNumberParticles());
  output.write<double>(getPotentialParameter());
  output.write<double>(getPropulsionVelocity());
  output.write<double>(getTransDiffusivity());
  output.write<double>(getRotDiffusivity());
  output.write<double>(getPersistenceLength());
  output.write<double>(getPackingFraction());
  output.write<double>(getSystemSize());
  output.write<int>(randomSeed);
  output.write<double>(getTimeStep());
  output.write<int>(framesWork);
  output.write<bool>(dumpParticles);
  output.write<int>(dumpPeriod);

  // write particles' diameters
  for (int i=0; i < getNumberParticles(); i++) {
    output.write<double>(getParticle(i)->diameter());
  }

  // initialise cell list
  initialiseCellList();
  // copy positions, orientations, and self-propulsion vectors, and update cell list
  copyState(system);
  // copy dumps
  copyDump(system);
}

System0::System0(
  System0* system, std::vector<double>& diameters, int seed,
  std::string filename, int nWork, bool dump, int period) :
  param(system->getParameters()),
  randomSeed(seed), randomGenerator(randomSeed),
  particles(0),
  cellList(),
  output(filename), velocitiesDumps(system->getNumberParticles()),
  framesWork(nWork > 0 ? nWork : (int)
    1/(system->getRotDiffusivity()*system->getTimeStep()*period)),
    dumpParticles(dump), dumpPeriod(period),
  dumpFrame(-1),
  workSum {0, 0, 0}, workForceSum {0, 0, 0}, workOrientationSum {0, 0, 0},
    orderSum {0, 0, 0} {

  // set diameters
  for (int i=0; i < getNumberParticles(); i++) {
    particles.push_back(Particle(diameters[i]));
  }

  // write header with system parameters to output file
  output.write<int>(getNumberParticles());
  output.write<double>(getPotentialParameter());
  output.write<double>(getPropulsionVelocity());
  output.write<double>(getTransDiffusivity());
  output.write<double>(getRotDiffusivity());
  output.write<double>(getPersistenceLength());
  output.write<double>(getPackingFraction());
  output.write<double>(getSystemSize());
  output.write<int>(randomSeed);
  output.write<double>(getTimeStep());
  output.write<int>(framesWork);
  output.write<bool>(dumpParticles);
  output.write<int>(dumpPeriod);

  // write particles' diameters
  for (int i=0; i < getNumberParticles(); i++) {
    output.write<double>(getParticle(i)->diameter());
  }

  // initialise cell list
  initialiseCellList();
  // copy positions, orientations, and self-propulsion vectors, and update cell list
  copyState(system);
  // copy dumps
  copyDump(system);
}

System0::System0(
  std::string inputFilename, int inputFrame, double dt,
  int seed, std::string filename,
  int nWork, bool dump, int period) :
  param(Dat0(inputFilename, false), dt),
  randomSeed(seed), randomGenerator(randomSeed),
  particles(0),
  cellList(),
  output(filename), velocitiesDumps(0),
  framesWork(nWork > 0 ? nWork : (int)
    1/(getRotDiffusivity()*getTimeStep()*period)),
    dumpParticles(dump), dumpPeriod(period),
  dumpFrame(-1),
  workSum {0, 0, 0}, workForceSum {0, 0, 0}, workOrientationSum {0, 0, 0},
    orderSum {0, 0, 0} {

  // load data
  Dat0 inputDat(inputFilename, false); // data object

  // set diameters
  std::vector<double> diameters = inputDat.getDiameters();
  for (int i=0; i < getNumberParticles(); i++) {
    particles.push_back(Particle(diameters[i]));
  }

  // resize velocity dumps
  velocitiesDumps.resize(2*getNumberParticles()); // resize vector of locations of velocity dumps

  // set positions and orientations
  for (int i=0; i < getNumberParticles(); i++) {
    // positions
    for (int dim=0; dim < 2; dim++) {
      particles[i].position()[dim] = inputDat.getPosition(inputFrame, i, dim);
    }
    // orientations
    particles[i].orientation()[0] = inputDat.getOrientation(inputFrame, i);
    // self-propulsion vector
    for (int dim=0; dim < 2; dim++) {
      particles[i].propulsion()[dim] =
        inputDat.getPropulsion(inputFrame, i, dim);
    }
  }

  // write header with system parameters to output file
  output.write<int>(getNumberParticles());
  output.write<double>(getPotentialParameter());
  output.write<double>(getPropulsionVelocity());
  output.write<double>(getTransDiffusivity());
  output.write<double>(getRotDiffusivity());
  output.write<double>(getPersistenceLength());
  output.write<double>(getPackingFraction());
  output.write<double>(getSystemSize());
  output.write<int>(randomSeed);
  output.write<double>(getTimeStep());
  output.write<int>(framesWork);
  output.write<bool>(dumpParticles);
  output.write<int>(dumpPeriod);

  // write particles' diameters
  for (int i=0; i < getNumberParticles(); i++) {
    output.write<double>(getParticle(i)->diameter());
  }

  // initialise cell list
  initialiseCellList();
}

// DESTRUCTORS

System0::~System0() {}

// METHODS

Parameters* System0::getParameters() { return &param; }
std::vector<double> System0::getDiameters() const {
  std::vector<double> diameters(0);
  for (int i=0; i < getNumberParticles(); i++) {
    diameters.push_back(particles[i].diameter());
  }
  return diameters;
}

int System0::getNumberParticles() const {
  return param.getNumberParticles(); }
double System0::getPotentialParameter() const {
  return param.getPotentialParameter(); }
double System0::getPropulsionVelocity() const {
  return param.getPropulsionVelocity(); }
double System0::getTransDiffusivity() const {
  return param.getTransDiffusivity(); }
double System0::getRotDiffusivity() const {
  return param.getRotDiffusivity(); }
double System0::getPersistenceLength() const {
  return param.getPersistenceLength(); }
double System0::getPackingFraction() const {
  return param.getPackingFraction(); }
double System0::getSystemSize() const {
  return param.getSystemSize(); }
double System0::getTimeStep() const {
  return param.getTimeStep(); }

int System0::getRandomSeed() const { return randomSeed; }
Random* System0::getRandomGenerator() { return &randomGenerator; }

Particle* System0::getParticle(int const& index) { return &(particles[index]); }
std::vector<Particle> System0::getParticles() { return particles; }

CellList* System0::getCellList() { return &cellList; }
void System0::initialiseCellList(double const& cutOff) {
  const std::vector<double> diameters = getDiameters();
  cellList = CellList(
    getNumberParticles(), getSystemSize(),
    cutOff*(*std::max_element(diameters.begin(), diameters.end())));
  updateCellList();
}
void System0::updateCellList() {
  std::vector<double*> positions(0);
  for (int i=0; i < getNumberParticles(); i++) {
    positions.push_back(particles[i].position());
  }
  cellList.listConstructor<double*>(positions);
}

void System0::flushOutputFile() { output.flush(); }
std::string System0::getOutputFile() const { return output.getOutputFile(); }

int* System0::getDump() { return &dumpFrame; }

void System0::resetDump() {
  // Reset time-extensive quantities over trajectory.

  dumpFrame = 0;

  workSum[0] = 0;
  workForceSum[0] = 0;
  workOrientationSum[0] = 0;
  orderSum[0] = 0;

  workSum[2] = 0;
  workForceSum[2] = 0;
  workOrientationSum[2] = 0;
  orderSum[2] = 0;
}

void System0::copyDump(System0* system) {
  // Copy dumps from other system.
  // WARNING: This also copies the index of last frame dumped. Consistency
  //          has to be checked.

  dumpFrame = system->getDump()[0];

  workSum[2] = system->getTotalWork()[0];
  workForceSum[2] = system->getTotalWorkForce()[0];
  workOrientationSum[2] = system->getTotalWorkOrientation()[0];
  orderSum[2] = system->getTotalOrder()[0];
}

double System0::getWork() { return workSum[1]; }
double System0::getWorkForce() { return workForceSum[1]; }
double System0::getWorkOrientation() { return workOrientationSum[1]; }
double System0::getOrder() { return orderSum[1]; }

double* System0::getTotalWork() { return &(workSum[2]); }
double* System0::getTotalWorkForce() { return &(workForceSum[2]); }
double* System0::getTotalWorkOrientation() { return &(workOrientationSum[2]); }
double* System0::getTotalOrder() { return &(orderSum[2]); }

void System0::copyState(std::vector<Particle>& newParticles) {
  // Copy positions and propulsions.

  for (int i=0; i < getNumberParticles(); i++) {
    for (int dim=0; dim < 2; dim++) {
      // POSITIONS
      particles[i].position()[dim] = newParticles[i].position()[dim];
    }
    // ORIENTATIONS
    particles[i].orientation()[0] = newParticles[i].orientation()[0];
    for (int dim=0; dim < 2; dim++) {
      // SELF-PROPULSION VECTORS
      particles[i].propulsion()[dim] = newParticles[i].propulsion()[dim];
    }
  }

  // UPDATING CELL LIST
  updateCellList();
}

void System0::copyState(System0* system) {
  // Copy positions and propulsions.

  for (int i=0; i < getNumberParticles(); i++) {
    for (int dim=0; dim < 2; dim++) {
      // POSITIONS
      particles[i].position()[dim] = (system->getParticle(i))->position()[dim];
    }
    // ORIENTATIONS
    particles[i].orientation()[0] = (system->getParticle(i))->orientation()[0];
    for (int dim=0; dim < 2; dim++) {
      // SELF-PROPULSION VECTORS
      particles[i].propulsion()[dim] =
        (system->getParticle(i))->propulsion()[dim];
    }
  }

  // UPDATING CELL LIST
  updateCellList();
}

void System0::saveInitialState() {
  // Saves initial state of particles to output file.

  // output
  if ( dumpParticles ) {

    for (int i=0; i < getNumberParticles(); i++) { // output all particles
      // WRAPPED POSITIONS
      for (int dim=0; dim < 2; dim++) { // output position in each dimension
        output.write<double>(particles[i].position()[dim]);
      }
      // ORIENTATIONS
      output.write<double>(particles[i].orientation()[0]); // output orientation
      // VELOCITIES
      velocitiesDumps[i] = output.tellp(); // location to dump velocities at next time step
      for (int dim=0; dim < 2; dim++) { // output velocity in each dimension
        output.write<double>(0.0); // zero by default for initial frame
      }
      // SELF-PROPULSION VECTORS
      for (int dim=0; dim < 2; dim++) { // output position in each dimension
        output.write<double>(particles[i].propulsion()[dim]);
      }
      // UNWRAPPED POSITIONS
      for (int dim=0; dim < 2; dim++) { // output position in each dimension
        output.write<double>(particles[i].position()[dim]);
      }
    }
  }

  // reset dump
  resetDump();
}

void System0::saveNewState(std::vector<Particle>& newParticles) {
  // Saves new state of particles to output file then copy it.

  int cross;

  // DUMP FRAME
  dumpFrame++;

  ////////////
  // SAVING //
  ////////////

  for (int i=0; i < getNumberParticles(); i++) {

    // COMPUTATION

    for (int dim=0; dim < 2; dim++) {

      // ACTIVE WORK
      // active work
      workSum[0] +=
        (newParticles[i].propulsion()[dim] + particles[i].propulsion()[dim])
        *(newParticles[i].position()[dim] - particles[i].position()[dim]) // NOTE: at this stage, newParticles[i].position() are not rewrapped, so this difference is the actual displacement
        /2;
      // force part of the active work
      workForceSum[0] +=
        (newParticles[i].propulsion()[dim] + particles[i].propulsion()[dim])
        *getTimeStep()*getPotentialParameter()*particles[i].force()[dim]
        /2;
      // orientation part of the active work
      workOrientationSum[0] +=
        (newParticles[i].propulsion()[dim] + particles[i].propulsion()[dim])
        *getTimeStep()*particles[i].propulsion()[dim]
        /2;

      // COORDINATES
      // compute crossings
      cross = wrapCoordinate<System0>(this, newParticles[i].position()[dim]);
      particles[i].cross()[dim] += cross;
      // keep particles in the box
      newParticles[i].position()[dim] -= cross*getSystemSize();
    }

    // DUMP

    // VELOCITIES
    if ( dumpParticles && (dumpFrame - 1) % dumpPeriod == 0 ) {
      for (int dim=0; dim < 2; dim++) {
        output.write<double>(
          particles[i].velocity()[dim],
          velocitiesDumps[i] + dim*sizeof(double));
      }
    }

    if ( dumpParticles && dumpFrame % dumpPeriod == 0 ) {
      // WRAPPED POSITION
      for (int dim=0; dim < 2; dim++) {
        output.write<double>(newParticles[i].position()[dim]);
      }
      // ORIENTATION
      output.write<double>(newParticles[i].orientation()[0]);
      // VELOCITIES
      velocitiesDumps[i] = output.tellp(); // location to dump velocities at next time step
      for (int dim=0; dim < 2; dim++) {
        output.write<double>(0.0); // zero by default until rewrite at next time step
      }
      // SELF-PROPULSION VECTORS
      for (int dim=0; dim < 2; dim++) {
        output.write<double>(newParticles[i].propulsion()[dim]);
      }
      // UNWRAPPED POSITION
      for (int dim=0; dim < 2; dim++) {
        output.write<double>(newParticles[i].position()[dim]
          + particles[i].cross()[dim]*getSystemSize());
      }
    }
  }

  // COMPUTATION

  // ORDER PARAMETER
  orderSum[0] +=
    (getOrderParameterNorm(particles) + getOrderParameterNorm(newParticles))
      /2;

  // DUMP

  // ACTIVE WORK and ORDER PARAMETER
  if ( dumpFrame % (framesWork*dumpPeriod) == 0 ) {
    // compute normalised rates since last dump
    workSum[1] = workSum[0]/(
      getNumberParticles()*getTimeStep()*framesWork*dumpPeriod);
    workForceSum[1] = workForceSum[0]/(
      getNumberParticles()*getTimeStep()*framesWork*dumpPeriod);
    workOrientationSum[1] = workOrientationSum[0]/(
      getNumberParticles()*getTimeStep()*framesWork*dumpPeriod);
    orderSum[1] = orderSum[0]/(
      framesWork*dumpPeriod);
    // output normalised rates
    output.write<double>(workSum[1]);
    output.write<double>(workForceSum[1]);
    output.write<double>(workOrientationSum[1]);
    output.write<double>(orderSum[1]);
    // update time extensive quantities over trajectory since last reset
    workSum[2] += workSum[0]/getNumberParticles();
    workForceSum[2] += workForceSum[0]/getNumberParticles();
    workOrientationSum[2] += workOrientationSum[0]/getNumberParticles();
    orderSum[2] += orderSum[0];
    // reset sums
    workSum[0] = 0;
    workForceSum[0] = 0;
    workOrientationSum[0] = 0;
    orderSum[0] = 0;
  }

  /////////////
  // COPYING //
  /////////////

  copyState(newParticles);
}


/***********
 * SYSTEMN *
 ***********/

// CONSTRUCTORS

SystemN::SystemN() :
  frameIndices(),
  param(new Parameters()),
  confined(false),
  randomSeed(0), randomGenerator(),
  particles(0),
  cellList(),
  output(""), velocitiesDumps(),
  dumpFrame(-1) {}

SystemN::SystemN(
  int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
    std::vector<int>* time0, std::vector<int>* deltat,
  Parameters* parameters, int seed, std::string filename,
  bool conf) :
  frameIndices(
    getLogFrames(init, Niter, dtMin, dtMax, nMax, intMax, time0, deltat)),
  param(parameters),
  confined(conf),
  randomSeed(seed), randomGenerator(randomSeed),
  particles(0),
  cellList(),
  output(filename), velocitiesDumps(parameters->getNumberParticles()),
  dumpFrame(-1) {

  // set diameters
  for (int i=0; i < getNumberParticles(); i++) {
    particles.push_back(Particle(1.0));
  }

  // write header with system parameters to output file
  output.write<int>(getNumberParticles());
  output.write<double>(getPotentialParameter());
  output.write<double>(getPropulsionVelocity());
  output.write<double>(getTransDiffusivity());
  output.write<double>(getRotDiffusivity());
  output.write<double>(getPersistenceLength());
  output.write<double>(getPackingFraction());
  output.write<double>(getSystemSize());
  output.write<int>(randomSeed);
  output.write<double>(getTimeStep());

  // write frames
  output.write<int>(time0->size()); // number of initial frames
  for (auto t0 = time0->begin(); t0 != time0->end(); t0++) {
    output.write<int>(*t0); // frame index
  }
  output.write<int>(deltat->size()); // number of lag times
  for (auto t = deltat->begin(); t != deltat->end(); t++) {
    output.write<int>(*t); // lag time
  }
  output.write<int>(frameIndices.size()); // number of frames
  for (std::vector<int>::const_iterator frame = frameIndices.begin();
    frame != frameIndices.end(); frame++) {
    output.write<int>(*frame); // frame index
  }

  // write particles' diameters
  for (int i=0; i < getNumberParticles(); i++) {
    output.write<double>(getParticle(i)->diameter());
  }

  // put particles on a grid with random orientation
  int gridSize = ceil(sqrt(getNumberParticles())); // size of the grid on which to put the particles
  double gridSpacing = getSystemSize()/gridSize;
  for (int i=0; i < getNumberParticles(); i++) { // loop over particles
    // position on the grid
    particles[i].position()[0] = ((i%gridSize) + 0.5*(i/gridSize))*gridSpacing;
    particles[i].position()[1] = (i/gridSize)*gridSpacing;
    for (int dim=0; dim < 2; dim++) {
      particles[i].position()[dim] -= wrapCoordinate<SystemN>(this,
        particles[i].position()[dim])*getSystemSize();
    }
    if ( isConfined() ) {
      std::vector<double> diameters = getDiameters();
      for (int dim=0; dim < 2; dim++) {
        particles[i].position()[dim] /=
          getSystemSize()
            /((1./sqrt(2))*(
              param.getConfiningLength()
              - 2*pow(2., 1./6.)*
                (*std::max_element(diameters.begin(), diameters.end()))));
        particles[i].position()[dim] +=
          (1./sqrt(2))*(
            getSystemSize()/2 - param.getConfiningLength()/2
            + 3*pow(2., 1./6.)*
              (*std::max_element(diameters.begin(), diameters.end())));
      }
    }
    // random orientation
    particles[i].orientation()[0] = 2*M_PI*randomGenerator.random01();
    // self-propulsion vector
    particles[i].propulsion()[0] =
      getPropulsionVelocity()*cos(particles[i].orientation()[0]);
    particles[i].propulsion()[1] =
      getPropulsionVelocity()*sin(particles[i].orientation()[0]);
  }

  // initialise cell list
  initialiseCellList();
}

SystemN::SystemN(
  int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
    std::vector<int>* time0, std::vector<int>* deltat,
  Parameters* parameters, std::vector<double>& diameters, int seed,
  std::string filename,
  bool conf) :
  frameIndices(
    getLogFrames(init, Niter, dtMin, dtMax, nMax, intMax, time0, deltat)),
  param(parameters),
  confined(conf),
  randomSeed(seed), randomGenerator(randomSeed),
  particles(0),
  cellList(),
  output(filename), velocitiesDumps(parameters->getNumberParticles()),
  dumpFrame(-1) {

  // set diameters
  for (int i=0; i < getNumberParticles(); i++) {
    particles.push_back(Particle(diameters[i]));
  }

  // write header with system parameters to output file
  output.write<int>(getNumberParticles());
  output.write<double>(getPotentialParameter());
  output.write<double>(getPropulsionVelocity());
  output.write<double>(getTransDiffusivity());
  output.write<double>(getRotDiffusivity());
  output.write<double>(getPersistenceLength());
  output.write<double>(getPackingFraction());
  output.write<double>(getSystemSize());
  output.write<int>(randomSeed);
  output.write<double>(getTimeStep());

  // write frames
  output.write<int>(time0->size()); // number of initial frames
  for (auto t0 = time0->begin(); t0 != time0->end(); t0++) {
    output.write<int>(*t0); // frame index
  }
  output.write<int>(deltat->size()); // number of lag times
  for (auto t = deltat->begin(); t != deltat->end(); t++) {
    output.write<int>(*t); // lag time
  }
  output.write<int>(frameIndices.size()); // number of frames
  for (std::vector<int>::const_iterator frame = frameIndices.begin();
    frame != frameIndices.end(); frame++) {
    output.write<int>(*frame); // frame index
  }

  // write particles' diameters
  for (int i=0; i < parameters->getNumberParticles(); i++) {
    output.write<double>(getParticle(i)->diameter());
  }

  // put particles on a grid with random orientation
  int gridSize = ceil(sqrt(getNumberParticles())); // size of the grid on which to put the particles
  double gridSpacing = getSystemSize()/gridSize;
  for (int i=0; i < getNumberParticles(); i++) { // loop over particles
    // position on the grid
    particles[i].position()[0] = ((i%gridSize) + 0.5*(i/gridSize))*gridSpacing;
    particles[i].position()[1] = (i/gridSize)*gridSpacing;
    for (int dim=0; dim < 2; dim++) {
      particles[i].position()[dim] -= wrapCoordinate<SystemN>(this,
        particles[i].position()[dim])*getSystemSize();
    }
    if ( isConfined() ) {
      std::vector<double> diameters = getDiameters();
      for (int dim=0; dim < 2; dim++) {
        particles[i].position()[dim] /=
          getSystemSize()
            /((1./sqrt(2))*(
              param.getConfiningLength()
              - 2*pow(2., 1./6.)*
                (*std::max_element(diameters.begin(), diameters.end()))));
        particles[i].position()[dim] +=
          (1./sqrt(2))*(
            getSystemSize()/2 - param.getConfiningLength()/2
            + 3*pow(2., 1./6.)*
              (*std::max_element(diameters.begin(), diameters.end())));
      }
    }
    // random orientation
    particles[i].orientation()[0] = 2*M_PI*randomGenerator.random01();
    // self-propulsion vector
    particles[i].propulsion()[0] =
      getPropulsionVelocity()*cos(particles[i].orientation()[0]);
    particles[i].propulsion()[1] =
      getPropulsionVelocity()*sin(particles[i].orientation()[0]);
  }

  // initialise cell list
  initialiseCellList();
}

SystemN::SystemN(
  int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
    std::vector<int>* time0, std::vector<int>* deltat,
  SystemN* system, int seed, std::string filename,
  bool conf) :
  frameIndices(
    getLogFrames(init, Niter, dtMin, dtMax, nMax, intMax, time0, deltat)),
  param(system->getParameters()),
  confined(conf),
  randomSeed(seed), randomGenerator(randomSeed),
  particles(0),
  cellList(),
  output(filename), velocitiesDumps(system->getNumberParticles()),
  dumpFrame(-1) {

  // set diameters
  std::vector<double> diameters = system->getDiameters();
  for (int i=0; i < getNumberParticles(); i++) {
    particles.push_back(Particle(diameters[i]));
  }

  // write header with system parameters to output file
  output.write<int>(getNumberParticles());
  output.write<double>(getPotentialParameter());
  output.write<double>(getPropulsionVelocity());
  output.write<double>(getTransDiffusivity());
  output.write<double>(getRotDiffusivity());
  output.write<double>(getPersistenceLength());
  output.write<double>(getPackingFraction());
  output.write<double>(getSystemSize());
  output.write<int>(randomSeed);
  output.write<double>(getTimeStep());

  // write frames
  output.write<int>(time0->size()); // number of initial frames
  for (auto t0 = time0->begin(); t0 != time0->end(); t0++) {
    output.write<int>(*t0); // frame index
  }
  output.write<int>(deltat->size()); // number of lag times
  for (auto t = deltat->begin(); t != deltat->end(); t++) {
    output.write<int>(*t); // lag time
  }
  output.write<int>(frameIndices.size()); // number of frames
  for (std::vector<int>::const_iterator frame = frameIndices.begin();
    frame != frameIndices.end(); frame++) {
    output.write<int>(*frame); // frame index
  }

  // write particles' diameters
  for (int i=0; i < getNumberParticles(); i++) {
    output.write<double>(getParticle(i)->diameter());
  }

  // initialise cell list
  initialiseCellList();
  // copy positions, orientations, and self-propulsion vectors, and update cell list
  copyState(system);
  // copy dumps
  copyDump(system);
}

SystemN::SystemN(
  int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
    std::vector<int>* time0, std::vector<int>* deltat,
  SystemN* system, std::vector<double>& diameters, int seed,
  std::string filename,
  bool conf) :
  frameIndices(
    getLogFrames(init, Niter, dtMin, dtMax, nMax, intMax, time0, deltat)),
  param(system->getParameters()),
  confined(conf),
  randomSeed(seed), randomGenerator(randomSeed),
  particles(0),
  cellList(),
  output(filename), velocitiesDumps(system->getNumberParticles()),
  dumpFrame(-1) {

  // set diameters
  for (int i=0; i < getNumberParticles(); i++) {
    particles.push_back(Particle(diameters[i]));
  }

  // write header with system parameters to output file
  output.write<int>(getNumberParticles());
  output.write<double>(getPotentialParameter());
  output.write<double>(getPropulsionVelocity());
  output.write<double>(getTransDiffusivity());
  output.write<double>(getRotDiffusivity());
  output.write<double>(getPersistenceLength());
  output.write<double>(getPackingFraction());
  output.write<double>(getSystemSize());
  output.write<int>(randomSeed);
  output.write<double>(getTimeStep());

  // write frames
  output.write<int>(time0->size()); // number of initial frames
  for (auto t0 = time0->begin(); t0 != time0->end(); t0++) {
    output.write<int>(*t0); // frame index
  }
  output.write<int>(deltat->size()); // number of lag times
  for (auto t = deltat->begin(); t != deltat->end(); t++) {
    output.write<int>(*t); // lag time
  }
  output.write<int>(frameIndices.size()); // number of frames
  for (std::vector<int>::const_iterator frame = frameIndices.begin();
    frame != frameIndices.end(); frame++) {
    output.write<int>(*frame); // frame index
  }

  // write particles' diameters
  for (int i=0; i < getNumberParticles(); i++) {
    output.write<double>(getParticle(i)->diameter());
  }

  // initialise cell list
  initialiseCellList();
  // copy positions, orientations, and self-propulsion vectors, and update cell list
  copyState(system);
  // copy dumps
  copyDump(system);
}

SystemN::SystemN(
  int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
    std::vector<int>* time0, std::vector<int>* deltat,
  std::string inputFilename, int inputFrame, double dt,
  int seed, std::string filename,
  bool conf) :
  frameIndices(
    getLogFrames(init, Niter, dtMin, dtMax, nMax, intMax, time0, deltat)),
  param(DatN(inputFilename, false), dt, conf),
  confined(conf),
  randomSeed(seed), randomGenerator(randomSeed),
  particles(0),
  cellList(),
  output(filename), velocitiesDumps(0),
  dumpFrame(-1) {

  // load data
  DatN inputDat(inputFilename, false); // data object

  // set diameters
  std::vector<double> diameters = inputDat.getDiameters();
  for (int i=0; i < getNumberParticles(); i++) {
    particles.push_back(Particle(diameters[i]));
  }

  // resize velocity dumps
  velocitiesDumps.resize(getNumberParticles()); // resize vector of locations of velocity dumps

  // set positions and orientations
  for (int i=0; i < getNumberParticles(); i++) {
    // positions
    for (int dim=0; dim < 2; dim++) {
      particles[i].position()[dim] = inputDat.getPosition(inputFrame, i, dim);
    }
    // orientations
    particles[i].orientation()[0] = inputDat.getOrientation(inputFrame, i);
    // self-propulsion vector
    for (int dim=0; dim < 2; dim++) {
      particles[i].propulsion()[dim] =
        inputDat.getPropulsion(inputFrame, i, dim);
    }
  }

  // write header with system parameters to output file
  output.write<int>(getNumberParticles());
  output.write<double>(getPotentialParameter());
  output.write<double>(getPropulsionVelocity());
  output.write<double>(getTransDiffusivity());
  output.write<double>(getRotDiffusivity());
  output.write<double>(getPersistenceLength());
  output.write<double>(getPackingFraction());
  output.write<double>(getSystemSize());
  output.write<int>(randomSeed);
  output.write<double>(getTimeStep());

  // write frames
  output.write<int>(time0->size()); // number of initial frames
  for (auto t0 = time0->begin(); t0 != time0->end(); t0++) {
    output.write<int>(*t0); // frame index
  }
  output.write<int>(deltat->size()); // number of lag times
  for (auto t = deltat->begin(); t != deltat->end(); t++) {
    output.write<int>(*t); // lag time
  }
  output.write<int>(frameIndices.size()); // number of frames
  for (std::vector<int>::const_iterator frame = frameIndices.begin();
    frame != frameIndices.end(); frame++) {
    output.write<int>(*frame); // frame index
  }

  // write particles' diameters
  for (int i=0; i < getNumberParticles(); i++) {
    output.write<double>(getParticle(i)->diameter());
  }

  // initialise cell list
  initialiseCellList();
}

SystemN::SystemN(
  int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
    std::vector<int>* time0, std::vector<int>* deltat,
  std::string inputFilename, int inputFrame, Parameters* parameters,
  int seed, std::string filename,
  bool conf) :
  frameIndices(
    getLogFrames(init, Niter, dtMin, dtMax, nMax, intMax, time0, deltat)),
  param(parameters),
  confined(conf),
  randomSeed(seed), randomGenerator(randomSeed),
  particles(0),
  cellList(),
  output(filename), velocitiesDumps(0),
  dumpFrame(-1) {

  // load data
  DatN inputDat(inputFilename, false); // data object
  int ratioNumberParticles = getNumberParticles()/inputDat.getNumberParticles(); // ratio of number of particles
  int nCopyCells = round(sqrt(ratioNumberParticles));
  if ( ratioNumberParticles != nCopyCells*nCopyCells ) {
    // ratio of number of particles has to be a perfect square to copy correctly
    throw std::invalid_argument(
      "Ratio of number of particles is not a perfect square.");
  }
  double ratioSystemSize = getSystemSize()/inputDat.getSystemSize()/nCopyCells; // ratio from old to new system size (per cell)
  auto mapParticleIndex = // mapping from particle index to input particle index
    [&inputDat](int i){ return i%inputDat.getNumberParticles(); };
  auto copyCellIndex = // mapping from particle to index of copy cell
    [&inputDat](int i){ return i/inputDat.getNumberParticles(); };

  // set diameters
  std::vector<double> diameters = inputDat.getDiameters();
  for (int i=0; i < getNumberParticles(); i++) {
    particles.push_back(Particle(diameters[mapParticleIndex(i)]));
  }

  // resize velocity dumps
  velocitiesDumps.resize(getNumberParticles()); // resize vector of locations of velocity dumps

  // set positions and orientations
  for (int i=0; i < getNumberParticles(); i++) {
    // positions
    for (int dim=0; dim < 2; dim++) {
      particles[i].position()[dim] =
        ratioSystemSize*(
          inputDat.getPosition(
            inputFrame, mapParticleIndex(i), dim)
          + inputDat.getSystemSize()*(dim == 0 ?
            copyCellIndex(i) % nCopyCells : copyCellIndex(i) / nCopyCells));
    }
    // orientations
    particles[i].orientation()[0] =
      inputDat.getOrientation(inputFrame, mapParticleIndex(i));
    // self-propulsion vector
    for (int dim=0; dim < 2; dim++) {
      particles[i].propulsion()[dim] =
        inputDat.getPropulsion(inputFrame, mapParticleIndex(i), dim);
    }
  }
  #if AOUP // system of AOUPs
  if ( (
      getTransDiffusivity() != inputDat.getTransDiffusivity()
      && getRotDiffusivity() != 0.0 )
    || (
      getRotDiffusivity() != inputDat.getRotDiffusivity()
      && getRotDiffusivity() != 0.0 )
    || getNumberParticles() != inputDat.getNumberParticles() ) {
    initPropulsionAOUP<SystemN>(this);
  }
  #endif

  // write header with system parameters to output file
  output.write<int>(getNumberParticles());
  output.write<double>(getPotentialParameter());
  output.write<double>(getPropulsionVelocity());
  output.write<double>(getTransDiffusivity());
  output.write<double>(getRotDiffusivity());
  output.write<double>(getPersistenceLength());
  output.write<double>(getPackingFraction());
  output.write<double>(getSystemSize());
  output.write<int>(randomSeed);
  output.write<double>(getTimeStep());

  // write frames
  output.write<int>(time0->size()); // number of initial frames
  for (auto t0 = time0->begin(); t0 != time0->end(); t0++) {
    output.write<int>(*t0); // frame index
  }
  output.write<int>(deltat->size()); // number of lag times
  for (auto t = deltat->begin(); t != deltat->end(); t++) {
    output.write<int>(*t); // lag time
  }
  output.write<int>(frameIndices.size()); // number of frames
  for (std::vector<int>::const_iterator frame = frameIndices.begin();
    frame != frameIndices.end(); frame++) {
    output.write<int>(*frame); // frame index
  }

  // write particles' diameters
  for (int i=0; i < getNumberParticles(); i++) {
    output.write<double>(getParticle(i)->diameter());
  }

  // initialise cell list
  initialiseCellList();
}

SystemN::SystemN(
  int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
    std::vector<int>* time0, std::vector<int>* deltat,
  std::string inputFilename, int inputFrame, Parameters* parameters,
    std::vector<double> const diameters,
  int seed, std::string filename,
  bool conf) :
  frameIndices(
    getLogFrames(init, Niter, dtMin, dtMax, nMax, intMax, time0, deltat)),
  param(parameters),
  confined(conf),
  randomSeed(seed), randomGenerator(randomSeed),
  particles(0),
  cellList(),
  output(filename), velocitiesDumps(0),
  dumpFrame(-1) {

  // load data
  DatN inputDat(inputFilename, false); // data object
  int ratioNumberParticles = getNumberParticles()/inputDat.getNumberParticles(); // ratio of number of particles
  int nCopyCells = round(sqrt(ratioNumberParticles));
  if ( ratioNumberParticles != nCopyCells*nCopyCells ) {
    // ratio of number of particles has to be a perfect square to copy correctly
    throw std::invalid_argument(
      "Ratio of number of particles is not a perfect square.");
  }
  double ratioSystemSize = getSystemSize()/inputDat.getSystemSize()/nCopyCells; // ratio from old to new system size (per cell)
  auto mapParticleIndex = // mapping from particle index to input particle index
    [&inputDat](int i){ return i%inputDat.getNumberParticles(); };
  auto copyCellIndex = // mapping from particle to index of copy cell
    [&inputDat](int i){ return i/inputDat.getNumberParticles(); };

  // set diameters
  if ( getL_WCA(getPackingFraction(), diameters) != getSystemSize() ) {
    throw std::invalid_argument("Unconsistent diameters.");
  }
  for (int i=0; i < getNumberParticles(); i++) {
    particles.push_back(Particle(diameters[i]));
  }

  // resize velocity dumps
  velocitiesDumps.resize(getNumberParticles()); // resize vector of locations of velocity dumps

  // set positions and orientations
  for (int i=0; i < getNumberParticles(); i++) {
    // positions
    for (int dim=0; dim < 2; dim++) {
      particles[i].position()[dim] =
        ratioSystemSize*(
          inputDat.getPosition(
            inputFrame, mapParticleIndex(i), dim)
          + inputDat.getSystemSize()*(dim == 0 ?
            copyCellIndex(i) % nCopyCells : copyCellIndex(i) / nCopyCells));
    }
    // orientations
    particles[i].orientation()[0] =
      inputDat.getOrientation(inputFrame, mapParticleIndex(i));
    // self-propulsion vector
    for (int dim=0; dim < 2; dim++) {
      particles[i].propulsion()[dim] =
        inputDat.getPropulsion(inputFrame, mapParticleIndex(i), dim);
    }
  }
  #if AOUP // system of AOUPs
  if ( (
      getTransDiffusivity() != inputDat.getTransDiffusivity()
      && getRotDiffusivity() != 0.0 )
    || (
      getRotDiffusivity() != inputDat.getRotDiffusivity()
      && getRotDiffusivity() != 0.0 )
    || getNumberParticles() != inputDat.getNumberParticles() ) {
    initPropulsionAOUP<SystemN>(this);
  }
  #endif

  // write header with system parameters to output file
  output.write<int>(getNumberParticles());
  output.write<double>(getPotentialParameter());
  output.write<double>(getPropulsionVelocity());
  output.write<double>(getTransDiffusivity());
  output.write<double>(getRotDiffusivity());
  output.write<double>(getPersistenceLength());
  output.write<double>(getPackingFraction());
  output.write<double>(getSystemSize());
  output.write<int>(randomSeed);
  output.write<double>(getTimeStep());

  // write frames
  output.write<int>(time0->size()); // number of initial frames
  for (auto t0 = time0->begin(); t0 != time0->end(); t0++) {
    output.write<int>(*t0); // frame index
  }
  output.write<int>(deltat->size()); // number of lag times
  for (auto t = deltat->begin(); t != deltat->end(); t++) {
    output.write<int>(*t); // lag time
  }
  output.write<int>(frameIndices.size()); // number of frames
  for (std::vector<int>::const_iterator frame = frameIndices.begin();
    frame != frameIndices.end(); frame++) {
    output.write<int>(*frame); // frame index
  }

  // write particles' diameters
  for (int i=0; i < getNumberParticles(); i++) {
    output.write<double>(getParticle(i)->diameter());
  }

  // initialise cell list
  initialiseCellList();
}

// DESTRUCTORS

SystemN::~SystemN() {}

// METHODS

std::vector<int> const* SystemN::getFrames() { return &frameIndices; }

Parameters* SystemN::getParameters() { return &param; }
std::vector<double> SystemN::getDiameters() const {
  std::vector<double> diameters(0);
  for (int i=0; i < getNumberParticles(); i++) {
    diameters.push_back(particles[i].diameter());
  }
  return diameters;
}

int SystemN::getNumberParticles() const {
  return param.getNumberParticles(); }
double SystemN::getPotentialParameter() const {
  return param.getPotentialParameter(); }
double SystemN::getPropulsionVelocity() const {
  return param.getPropulsionVelocity(); }
double SystemN::getTransDiffusivity() const {
  return param.getTransDiffusivity(); }
double SystemN::getRotDiffusivity() const {
  return param.getRotDiffusivity(); }
double SystemN::getPersistenceLength() const {
  return param.getPersistenceLength(); }
double SystemN::getPackingFraction() const {
  return param.getPackingFraction(); }
double SystemN::getSystemSize() const {
  return param.getSystemSize(); }
double SystemN::getTimeStep() const {
  return param.getTimeStep(); }

int SystemN::getRandomSeed() const { return randomSeed; }
Random* SystemN::getRandomGenerator() { return &randomGenerator; }

Particle* SystemN::getParticle(int const& index) { return &(particles[index]); }
std::vector<Particle> SystemN::getParticles() { return particles; }
std::vector<double*> SystemN::getPositions() {
  std::vector<double*> positions(0);
  for (int i=0; i < getNumberParticles(); i++) {
    positions.push_back(particles[i].position());
  }
  return positions;
}

CellList* SystemN::getCellList() { return &cellList; }
void SystemN::initialiseCellList(double const& cutOff) {
  const std::vector<double> diameters = getDiameters();
  cellList = CellList(
    getNumberParticles(), getSystemSize(),
    cutOff*(*std::max_element(diameters.begin(), diameters.end())));
  updateCellList();
}
void SystemN::updateCellList() {
  cellList.listConstructor<double*>(getPositions());
}

void SystemN::flushOutputFile() { output.flush(); }
std::string SystemN::getOutputFile() const { return output.getOutputFile(); }

Write* SystemN::getOutput()
  { return &output; }
std::vector<long int>* SystemN::getVelocitiesDumps()
  { return &velocitiesDumps; }

int* SystemN::getDump() { return &dumpFrame; }

void SystemN::resetDump() {
  // Reset time-extensive quantities over trajectory.

  dumpFrame = 0;
}

void SystemN::copyDump(SystemN* system) {
  // Copy dumps from other system.
  // WARNING: This also copies the index of last frame dumped. Consistency
  //          has to be checked.

  dumpFrame = system->getDump()[0];
}

void SystemN::copyState(std::vector<Particle>& newParticles) {
  // Copy positions and propulsions.

  for (int i=0; i < getNumberParticles(); i++) {
    for (int dim=0; dim < 2; dim++) {
      // POSITIONS
      particles[i].position()[dim] = newParticles[i].position()[dim];
    }
    // ORIENTATIONS
    particles[i].orientation()[0] = newParticles[i].orientation()[0];
    for (int dim=0; dim < 2; dim++) {
      // SELF-PROPULSION VECTORS
      particles[i].propulsion()[dim] = newParticles[i].propulsion()[dim];
    }
  }

  // UPDATING CELL LIST
  updateCellList();
}

void SystemN::copyState(SystemN* system) {
  // Copy positions and propulsions.

  for (int i=0; i < getNumberParticles(); i++) {
    for (int dim=0; dim < 2; dim++) {
      // POSITIONS
      particles[i].position()[dim] = (system->getParticle(i))->position()[dim];
    }
    // ORIENTATIONS
    particles[i].orientation()[0] = (system->getParticle(i))->orientation()[0];
    for (int dim=0; dim < 2; dim++) {
      // SELF-PROPULSION VECTORS
      particles[i].propulsion()[dim] =
        (system->getParticle(i))->propulsion()[dim];
    }
  }

  // UPDATING CELL LIST
  updateCellList();
}

void SystemN::saveInitialState() {
  // Saves initial state of particles to output file.

  for (int i=0; i < getNumberParticles(); i++) { // output all particles
    // WRAPPED POSITIONS
    for (int dim=0; dim < 2; dim++) { // output position in each dimension
      output.write<double>(particles[i].position()[dim]);
    }
    // ORIENTATIONS
    output.write<double>(particles[i].orientation()[0]); // output orientation
    // VELOCITIES
    velocitiesDumps[i] = output.tellp(); // location to dump velocities at next time step
    for (int dim=0; dim < 2; dim++) { // output velocity in each dimension
      output.write<double>(0.0); // zero by default for initial frame
    }
    // SELF-PROPULSION VECTORS
    for (int dim=0; dim < 2; dim++) { // output position in each dimension
      output.write<double>(particles[i].propulsion()[dim]);
    }
    // UNWRAPPED POSITIONS
    for (int dim=0; dim < 2; dim++) { // output position in each dimension
      output.write<double>(particles[i].position()[dim]);
    }
  }

  // reset dump
  resetDump();
}

void SystemN::saveNewState(std::vector<Particle>& newParticles) {
  // Saves new state of particles to output file then copy it.

  int cross;

  // DUMP FRAME
  dumpFrame++;

  ////////////
  // SAVING //
  ////////////

  kineticEnergy = 0;
  for (int i=0; i < getNumberParticles(); i++) {

    // COMPUTATION

    for (int dim=0; dim < 2; dim++) {
      // KINETIC ENERGY
      kineticEnergy += pow(particles[i].velocity()[dim], 2.0);
      // COORDINATES
      // compute crossings
      cross = wrapCoordinate<SystemN>(this, newParticles[i].position()[dim]);
      particles[i].cross()[dim] += cross;
      // keep particles in the box
      newParticles[i].position()[dim] -= cross*getSystemSize();
    }

    // DUMP

    // VELOCITIES
    if ( isInSortedVec<int>(&frameIndices, dumpFrame - 1) || dumpFrame == 1 ) {
      for (int dim=0; dim < 2; dim++) {
        output.write<double>(
          particles[i].velocity()[dim],
          velocitiesDumps[i] + dim*sizeof(double));
      }
    }

    if ( isInSortedVec<int>(&frameIndices, dumpFrame) ) {
      // WRAPPED POSITION
      for (int dim=0; dim < 2; dim++) {
        output.write<double>(newParticles[i].position()[dim]);
      }
      // ORIENTATION
      output.write<double>(newParticles[i].orientation()[0]);
      // VELOCITIES
      velocitiesDumps[i] = output.tellp(); // location to dump velocities at next time step
      for (int dim=0; dim < 2; dim++) {
        output.write<double>(0.0); // zero by default until rewrite at next time step
      }
      // SELF-PROPULSION VECTORS
      for (int dim=0; dim < 2; dim++) {
        output.write<double>(newParticles[i].propulsion()[dim]);
      }
      // UNWRAPPED POSITION
      for (int dim=0; dim < 2; dim++) {
        output.write<double>(newParticles[i].position()[dim]
          + particles[i].cross()[dim]*getSystemSize());
      }
    }

  }

  //////////////
  // CHECKING //
  //////////////

  if (
    kineticEnergy > kineticEnergyFactor
      *getNumberParticles()*pow(getPropulsionVelocity(), 2.0) ) {
    throw std::invalid_argument("Exceeded kinetic energy limit. <v^2> = "
      + std::to_string(kineticEnergy/getNumberParticles()));
  }

  /////////////
  // COPYING //
  /////////////

  copyState(newParticles);
}

/**********
 * ROTORS *
 **********/

// CONSTRUCTORS

Rotors::Rotors() :
  numberParticles(0), rotDiffusivity(0), torqueParameter(0),
  timeStep(0), framesOrder(0), dumpRotors(0), dumpPeriod(0),
  randomSeed(0), randomGenerator(),
  orientations(), torques(),
  output(""), biasingParameter(0), dumpFrame(-1),
  orderSum {0, 0, 0}, orderSumSq {0, 0, 0}
  #if BIAS == 1
  #ifdef CONTROLLED_DYNAMICS
  , biasIntegral {0, 0}
  #endif
  #endif
  {}

Rotors::Rotors(
  int N, double Dr, double dt, int seed, double g, std::string filename,
  int nOrder, bool dump, int period) :
  numberParticles(N), rotDiffusivity(Dr), torqueParameter(g),
  timeStep(dt), framesOrder(nOrder > 0 ? nOrder : (int)
    1/(getRotDiffusivity()*getTimeStep()*period)), dumpRotors(dump),
    dumpPeriod(period),
  randomSeed(seed), randomGenerator(randomSeed),
  orientations(N), torques(N, 0.0),
  output(filename), biasingParameter(0), dumpFrame(-1),
  orderSum {0, 0, 0}, orderSumSq {0, 0, 0}
  #if BIAS == 1
  #ifdef CONTROLLED_DYNAMICS
  , biasIntegral {0, 0}
  #endif
  #endif
  {

  // give random orientations to rotors
  for (int i=0; i < numberParticles; i++) { // loop over rotors
    orientations[i] = 2*M_PI*randomGenerator.random01();
  }

  // write header with system parameters to output file
  output.write<int>(numberParticles);
  output.write<double>(rotDiffusivity);
  output.write<double>(torqueParameter);
  output.write<double>(timeStep);
  output.write<int>(framesOrder);
  output.write<bool>(dumpRotors);
  output.write<int>(dumpPeriod);
  output.write<int>(randomSeed);
}

Rotors::Rotors(
  Rotors* rotors, int seed, std::string filename, int nOrder, bool dump,
  int period) :
  numberParticles(rotors->getNumberParticles()),
    rotDiffusivity(rotors->getRotDiffusivity()),
    torqueParameter(rotors->getTorqueParameter()),
  timeStep(rotors->getTimeStep()), framesOrder(nOrder > 0 ? nOrder : (int)
    1/(getRotDiffusivity()*getTimeStep()*period)), dumpRotors(dump),
    dumpPeriod(period),
  randomSeed(seed), randomGenerator(randomSeed),
  orientations(getNumberParticles()), torques(getNumberParticles(), 0.0),
  output(filename), biasingParameter(0), dumpFrame(-1),
  orderSum {0, 0, 0}, orderSumSq {0, 0, 0}
  #if BIAS == 1
  #ifdef CONTROLLED_DYNAMICS
  , biasIntegral {0, 0}
  #endif
  #endif
  {

  // give random orientations to rotors
  for (int i=0; i < numberParticles; i++) { // loop over rotors
    orientations[i] = 2*M_PI*randomGenerator.random01();
  }

  // write header with system parameters to output file
  output.write<int>(numberParticles);
  output.write<double>(rotDiffusivity);
  output.write<double>(torqueParameter);
  output.write<double>(timeStep);
  output.write<int>(framesOrder);
  output.write<bool>(dumpRotors);
  output.write<int>(dumpPeriod);
  output.write<int>(randomSeed);
}

// DESTRUCTORS

Rotors::~Rotors() {}

// METHODS

int Rotors::getNumberParticles() const { return numberParticles; }
double Rotors::getRotDiffusivity() const { return rotDiffusivity; }

double Rotors::getTorqueParameter() { return torqueParameter; }
void Rotors::setTorqueParameter(double g) { torqueParameter = g; }

double Rotors::getTimeStep() const { return timeStep; }

double* Rotors::getOrientation(int const& index)
  { return &(orientations[index]); }
double* Rotors::getTorque(int const& index) { return &(torques[index]); }

Random* Rotors::getRandomGenerator() { return &randomGenerator; }

double Rotors::getBiasingParameter() { return biasingParameter; }
void Rotors::setBiasingParameter(double s) {
  biasingParameter = s;
  #if BIAS == 1
  #ifdef CONTROLLED_DYNAMICS
  setTorqueParameter(getBiasingParameter());
  #endif
  #endif
}

int* Rotors::getDump() { return &dumpFrame; }

void Rotors::resetDump() {
  // Reset time-extensive quantities over trajectory.

  dumpFrame = 0;

  orderSum[0] = 0;
  orderSumSq[0] = 0;

  orderSum[2] = 0;
  orderSumSq[2] = 0;
}

void Rotors::copyDump(Rotors* rotors) {
  // Copy dumps from other system.
  // WARNING: This also copies the index of last frame dumped. Consistency
  //          has to be checked.

  dumpFrame = rotors->getDump()[0];

  orderSum[2] = rotors->getTotalOrder()[0];
  orderSumSq[2] = rotors->getTotalOrderSq()[0];
}

double Rotors::getOrder() { return orderSum[1]; }
double Rotors::getOrderSq() { return orderSumSq[1]; }

double* Rotors::getTotalOrder() { return &(orderSum[2]); }
double* Rotors::getTotalOrderSq() { return &(orderSumSq[2]); }

#if BIAS == 1
#ifdef CONTROLLED_DYNAMICS
double Rotors::getBiasIntegral() { return biasIntegral[1]; }
#endif
#endif

void Rotors::copyState(std::vector<double>& newOrientations) {
  // Copy orientations.

  for (int i=0; i < getNumberParticles(); i++) {
    // ORIENTATIONS
    orientations[i] = newOrientations[i];
  }
}

void Rotors::copyState(Rotors* rotors) {
  // Copy orientations.

  for (int i=0; i < getNumberParticles(); i++) {
    // ORIENTATIONS
    orientations[i] = rotors->getOrientation(i)[0];
  }
}

void Rotors::saveInitialState() {
  // Saves initial state of rotors to output file.

  // output
  if ( dumpRotors ) {
      for (int i=0; i < numberParticles; i++) {
        output.write<double>(orientations[i]);
      }
  }

  // reset dump
  resetDump();
}

void Rotors::saveNewState(std::vector<double>& newOrientations) {
  // Saves new state of rotors to output file then copy it.

  // DUMP FRAME
  dumpFrame++;

  ////////////
  // SAVING //
  ////////////

  for (int i=0; i < getNumberParticles(); i++) { // output all rotors

    if ( dumpRotors && dumpFrame % dumpPeriod == 0 ) {

      // ORIENTATION
      output.write<double>(newOrientations[i]);
    }
  }

  // ORDER PARAMETER
  double oldOrder = getOrderParameterNorm(orientations);
  double oldOrderSq = pow(oldOrder, 2.0);
  double newOrder = getOrderParameterNorm(newOrientations);
  double newOrderSq = pow(newOrder, 2.0);
  orderSum[0] += (oldOrder + newOrder)/2;
  orderSumSq[0] += (oldOrderSq + newOrderSq)/2;
  // BIAS INTEGRAL
  #if BIAS == 1
  #ifdef CONTROLLED_DYNAMICS
  double globalPhaseOld = getGlobalPhase(orientations);
  double globalPhaseNew = getGlobalPhase(newOrientations);
  for (int i=0; i < getNumberParticles(); i++) {
    biasIntegral[0] += oldOrderSq
      *pow(sin(orientations[i] - globalPhaseOld), 2.0)/2.0;
    biasIntegral[0] += newOrderSq
      *pow(sin(newOrientations[i] - globalPhaseNew), 2.0)/2.0;
  }
  #endif
  #endif

  if ( dumpFrame % (framesOrder*dumpPeriod) == 0 ) {
    // compute normalised rates since last dump
    orderSum[1] = orderSum[0]/(framesOrder*dumpPeriod);
    orderSumSq[1] = orderSumSq[0]/(framesOrder*dumpPeriod);
    // output normalised rates
    output.write<double>(orderSum[1]);
    output.write<double>(orderSumSq[1]);
    // update time extensive quantities over trajectory since last reset
    orderSum[2] += orderSum[0];
    orderSumSq[2] += orderSumSq[0];
    // reset sums
    orderSum[0] = 0;
    orderSumSq[0] = 0;
    // bias integral
    #if BIAS == 1
    #ifdef CONTROLLED_DYNAMICS
    biasIntegral[1] = getTimeStep()*biasIntegral[0];
    biasIntegral[0] = 0;
    #endif
    #endif
  }

  /////////////
  // COPYING //
  /////////////

  copyState(newOrientations);
}


///////////////
// FUNCTIONS //
///////////////

double getL_WCA(double phi, std::vector<double> const& diameters) {
  // Returns the length of a square system with packing fraction `phi'
  // containing particles with `diameters', considering the actual diameter
  // as the WCA diameter of interaction.

  double totalArea = 0.;
  for (auto i = diameters.begin(); i != diameters.end(); i++) {
    totalArea += M_PI*pow((*i)*pow(2., 1./6.), 2)/4.;
  }
  return sqrt(totalArea/phi);
}

double getL_WCA(double phi, int N, double diameter) {
  // Returns the length of a square system with packing fraction `phi'
  // containing  `N' particles with same `diameter', considering the actual
  // diameter as the WCA diameter of interaction.

  return getL_WCA(phi, std::vector<double>(N, diameter));
}

double getLc_WCA(double phi, std::vector<double> const& diameters) {
  // Returns the diameter of a circular system with packing fraction `phi'
  // containing particles with `diameters', considering the actual diameter
  // as the WCA diameter of interaction.

  double sumSigma2 = 0.;
  for (auto i = diameters.begin(); i != diameters.end(); i++) {
    sumSigma2 += pow((*i)*pow(2., 1./6.), 2);
  }
  return sqrt(sumSigma2/phi);
}

std::vector<double> getDiametersI(int N, double I, int seed) {
  // Returns vector of `N' uniformly distributed diameters with polydispersity
  // index `I'.

  // array of diameters
  std::vector<double> diameters (N, 1.0);
  if ( N > 1 ) {
    for (int i=0; i < N; i++) {
      diameters[i] = 1 - sqrt(3)*I + 2*sqrt(3)*I*i/(N - 1);
    }
  }
  // randomisation of diameters order
  Random randomGenerator(seed);
  std::random_shuffle(diameters.begin(), diameters.end(),
    [&randomGenerator](int max) { return randomGenerator.randomInt(max); });

  return diameters;
}

std::vector<double> getOrderParameter(std::vector<Particle>& particles) {
  // Returns order parameter.

  std::vector<double> order(2, 0);

  int N = particles.size();
  for (int i=0; i < N; i++) { // loop over particles
    for (int dim=0; dim < 2; dim++) { // loop over dimensions
      order[dim] += cos(particles[i].orientation()[0] - dim*M_PI/2)/N;
    }
  }

  return order;
}

std::vector<double> getOrderParameter(std::vector<double>& orientations) {
  // Returns order parameter.

  std::vector<double> order(2, 0);

  int N = orientations.size();
  for (int i=0; i < N; i++) { // loop over particles
    for (int dim=0; dim < 2; dim++) { // loop over dimensions
      order[dim] += cos(orientations[i] - dim*M_PI/2)/N;
    }
  }

  return order;
}

double getOrderParameterNorm(std::vector<Particle>& particles) {
  // Returns order parameter norm.

  std::vector<double> order = getOrderParameter(particles);

  return sqrt(pow(order[0], 2.0) + pow(order[1], 2.0));
}

double getOrderParameterNorm(std::vector<double>& orientations) {
  // Returns order parameter norm.

  std::vector<double> order = getOrderParameter(orientations);

  return sqrt(pow(order[0], 2.0) + pow(order[1], 2.0));
}

double getGlobalPhase(std::vector<Particle>& particles) {
  // Returns global phase.

  std::vector<double> order = getOrderParameter(particles);

  return getAngle(order[0]/sqrt(pow(order[0], 2) + pow(order[1], 2)), order[1]);
}

double getGlobalPhase(std::vector<double>& orientations) {
  // Returns global phase.

  std::vector<double> order = getOrderParameter(orientations);

  return getAngle(order[0]/sqrt(pow(order[0], 2) + pow(order[1], 2)), order[1]);
}

std::vector<int> getLogFrames(
  int init, int Niter, int dtMin, int* dtMax, int nMax, int intMax,
  std::vector<int>* time0, std::vector<int>* dt) {
  // Returns vector of logarithmically spaced frames in overlapping linearly
  // spaced blocks.
  // Saves in `time0' the initial frames and in `dt' the lag times.
  // NOTE: `dtMax' may be modified according to other parameters.

  std::vector<int> frames;

  if ( dtMin == 0 && nMax == 0 ) {
    // LINEARLY SPACED FRAMES

    if ( Niter % dtMax[0] != 0 ) {
      throw std::invalid_argument(
        "No integer multiple of " + std::to_string(dtMax[0])
        + " in a total of " + std::to_string(Niter) + ".");
    }

    time0->clear();
    if ( intMax == 1 ) {
      time0->push_back((int) init);
    }
    else {
      for (int i=0; i < intMax; i++) {
        time0->push_back((int) init + i*dtMax[0]);
      }
    }
    sortVec(time0);

    dt->clear();
    dt->push_back(dtMin);
    dt->push_back(dtMax[0]);
    sortVec(dt);

  }
  else {
    // LOGARITHMICALLY SPACED FRAMES

    dtMax[0] = std::min(dtMax[0], Niter - (intMax - 1));

    time0->clear();
    if ( intMax == 1 ) {
      time0->push_back((int) init);
    }
    else {
      long int t;
      for (long int i=0; i < intMax; i++) {
        t = init + i*(Niter - dtMax[0])/(intMax - 1);
        time0->push_back((int) t);
      }
    }
    sortVec(time0);

    dt->clear();
    dt->push_back(dtMin);
    for (int i=1; i < nMax - 1; i++) {
      dt->push_back((int) dtMin + exp(i*log(dtMax[0] - dtMin)/(nMax - 1)));
    }
    dt->push_back(dtMax[0]);
    sortVec(dt);

  }

  for (auto t0 = time0->begin(); t0 != time0->end(); t0++) {
    frames.push_back(*t0);
    for (auto t = dt->begin(); t != dt->end(); t++) {
      frames.push_back(*t0 + *t);
    }
  }
  frames.push_back(init + Niter);

  return *sortVec(removeVec(&frames, 0));
}

template<> void WCA_force<System>(
  System* system, int const& index1, int const& index2,
  double* force, double* diff) {
  // Writes to `force' the force deriving from the WCA potential between
  // particles `index1' and `index2' in `system'.

  force[0] = 0.0;
  force[1] = 0.0;

  double dist = getDistance<System>(system, index1, index2, diff); // dimensionless distance between particles

  if (dist < pow(2., 1./6.)) { // distance lower than cut-off

    // compute force
    double coeff = 48.0/pow(dist, 14.0) - 24.0/pow(dist, 8.0);
    for (int dim=0; dim < 2; dim++) {
      force[dim] = diffPeriodic<System>(system,
          (system->getParticle(index2))->position()[dim],
          (system->getParticle(index1))->position()[dim])
        *coeff;
    }
  }
}

#include <cmath>
#include <math.h>
#include <vector>

#include "iteration.hpp"
#include "particle.hpp"


///////////////////////////////
// ACTIVE BROWNIAN PARTICLES //
///////////////////////////////

void iterate_ABP_WCA(System* system, int Niter) {
  // Updates system to next step according to the dynamics of active Brownian
  // particles with WCA potential, using custom dimensionless parameters
  // relations.

  Parameters* parameters = system->getParameters();

  bool const considerTorque = ( system->getTorqueParameter() != 0 );

  std::vector<Particle> newParticles(parameters->getNumberParticles());

  double selfPropulsion; // self-propulsion force
  double noise; // noise realisation

  #if HEUN // HEUN'S SCHEME

  double selfPropulsionCorrection; // correction to the self-propulsion force

  std::vector<double> positions (2*parameters->getNumberParticles(), 0.0); // positions backup
  std::vector<double> forces (2*parameters->getNumberParticles(), 0.0); // forces backup
  std::vector<double> orientations; // orientations backup
  std::vector<double> torques; // torques backup
  if ( considerTorque ) { // only need to use this memory when torque parameter is not 0
    orientations.assign(parameters->getNumberParticles(), 0.0);
    torques.assign(parameters->getNumberParticles(), 0.0);
  }
  #endif

  for (int iter=0; iter < Niter; iter++) {

    // COMPUTATION
    for (int i=0; i < parameters->getNumberParticles(); i++) {

      // POSITIONS
      for (int dim=0; dim < 2; dim++) {
        // initialise velocity
        (system->getParticle(i))->velocity()[dim] = 0.0;
        // initialise new positions with previous ones
        newParticles[i].position()[dim] =
          (system->getParticle(i))->position()[dim];
        // add self-propulsion
        selfPropulsion =
          #if CONTROLLED_DYNAMICS
          (1.0 - 2.0*system->getBiasingParameter()
            /3.0/parameters->getPersistenceLength())*
          #endif
          cos((system->getParticle(i))->orientation()[0] - dim*M_PI/2);
        (system->getParticle(i))->velocity()[dim] += selfPropulsion;
        newParticles[i].position()[dim] +=
          parameters->getTimeStep()*selfPropulsion;
        // add noise
        noise = (system->getRandomGenerator())->gauss_cutoff();
        (system->getParticle(i))->velocity()[dim] +=
          sqrt(2.0/3.0/parameters->getPersistenceLength())
          *noise;
        newParticles[i].position()[dim] +=
          sqrt(parameters->getTimeStep()
            *2.0/3.0/parameters->getPersistenceLength())
          *noise;
        // initialise force
        (system->getParticle(i))->force()[dim] = 0.0;
      }

      // ORIENTATIONS
      // initialise new orientation with previous one
      newParticles[i].orientation()[0] =
        (system->getParticle(i))->orientation()[0];
      // add noise
      newParticles[i].orientation()[0] +=
        sqrt(parameters->getTimeStep()*2.0/parameters->getPersistenceLength())
          *(system->getRandomGenerator())->gauss_cutoff();
      if ( considerTorque ) {
        // initialise torque
        (system->getParticle(i))->torque()[0] = 0.0;
      }
    }

    // FORCES AND ALIGNING TORQUES
    ABP_WCA<System>(system); // compute forces
    if ( considerTorque ) {
      aligningTorque<System>(system,
        [&system](int index) {
          return (system->getParticle(index))->orientation(); },
        [&system](int index) {
          return (system->getParticle(index))->torque(); }); // compute torques
    }

    for (int i=0; i < parameters->getNumberParticles(); i++) {
      for (int dim=0; dim < 2; dim++) {
        (system->getParticle(i))->velocity()[dim] +=
          (system->getParticle(i))->force()[dim]
          /3.0/parameters->getPersistenceLength(); // add force
        newParticles[i].position()[dim] +=
          (system->getParticle(i))->force()[dim]
          *parameters->getTimeStep()/3.0/parameters->getPersistenceLength(); // add force displacement
      }
      if ( considerTorque ) {
        newParticles[i].orientation()[0] +=
          (system->getParticle(i))->torque()[0]*parameters->getTimeStep(); // add torque rotation
      }
    }

    // HEUN'S SCHEME
    #if HEUN
    for (int i=0; i < parameters->getNumberParticles(); i++) {

      for (int dim=0; dim < 2; dim++) {
        // POSITIONS
        positions[2*i + dim] = (system->getParticle(i))->position()[dim]; // save initial position
        (system->getParticle(i))->position()[dim] =
          newParticles[i].position()[dim]; // integrate position as if using Euler's scheme
        // FORCES
        forces[2*i + dim] = (system->getParticle(i))->force()[dim]; // save computed force at initial position
        (system->getParticle(i))->force()[dim] = 0.0; // re-initialise force
      }

      if ( considerTorque ) {
        // ORIENTATIONS
        orientations[i] = (system->getParticle(i))->orientation()[0]; // save initial orientation
        (system->getParticle(i))->orientation()[0] =
          newParticles[i].orientation()[0]; // integrate position as if using Euler's scheme
        // TORQUES
        torques[i] = (system->getParticle(i))->torque()[0]; // save computed force at initial position
        (system->getParticle(i))->torque()[0] = 0.0; // re-initialise torque
      }
    }

    // FORCES AND ALIGNING TORQUES
    ABP_WCA<System>(system); // re-compute forces
    if ( considerTorque ) {
      aligningTorque<System>(system,
        [&system](int index) {
          return (system->getParticle(index))->orientation(); },
        [&system](int index) {
          return (system->getParticle(index))->torque(); }); // re-compute torques
    }

    for (int i=0; i < parameters->getNumberParticles(); i++) {

      // CORRECTION TO INTERPARTICLE FORCE
      for (int dim=0; dim < 2; dim++) {
        (system->getParticle(i))->velocity()[dim] +=
          ((system->getParticle(i))->force()[dim] - forces[2*i + dim])
          /3.0/parameters->getPersistenceLength()/2; // velocity
        newParticles[i].position()[dim] +=
          ((system->getParticle(i))->force()[dim] - forces[2*i + dim])
          *parameters->getTimeStep()/3.0/parameters->getPersistenceLength()/2; // position
        (system->getParticle(i))->force()[dim] =
          ((system->getParticle(i))->force()[dim] + forces[2*i + dim])/2; // force
      }

      // CORRECTION TO SELF-PROPULSION FORCE
      for (int dim=0; dim < 2; dim++) {
        selfPropulsionCorrection = 1.0;
        #if CONTROLLED_DYNAMICS
        selfPropulsionCorrection *=
          (1.0 - 2.0*system->getBiasingParameter()
            /3.0/parameters->getPersistenceLength());
        #endif
        if ( considerTorque ) {
          selfPropulsionCorrection *=
            (cos(newParticles[i].orientation()[0] - dim*M_PI/2)
              - cos(orientations[2*i + dim] - dim*M_PI/2));
        }
        else {
          selfPropulsionCorrection *=
            (cos(newParticles[i].orientation()[0] - dim*M_PI/2)
              - cos((system->getParticle(i))->orientation()[0] - dim*M_PI/2));
        }
        selfPropulsionCorrection /= 2;
        (system->getParticle(i))->velocity()[dim] +=
          selfPropulsionCorrection; // velocity
        newParticles[i].position()[dim] +=
          parameters->getTimeStep()*selfPropulsionCorrection; // position
      }

      // CORRECTION TO TORQUE
      if ( considerTorque ) {
        newParticles[i].orientation()[0] +=
          ((system->getParticle(i))->torque()[0] - torques[i])
          *parameters->getTimeStep()/2; // orientation
        (system->getParticle(i))->torque()[0] =
          ((system->getParticle(i))->torque()[0] + torques[i])/2; // torque
      }

      // RESET INITIAL POSITIONS AND ORIENTATION
      for (int dim=0; dim < 2; dim++) {
        (system->getParticle(i))->position()[dim] = positions[2*i + dim]; // position
      }
      if ( considerTorque ) {
        (system->getParticle(i))->orientation()[0] = orientations[i]; // orientation
      }
    }
    #endif

    // SAVE AND COPY
    system->saveNewState(newParticles);
  }
}

void iterate_ABP_WCA(System0* system, int Niter) {
  // Updates system to next step according to the dynamics of active Brownian
  // particles with WCA potential.

  Parameters* parameters = system->getParameters();

  std::vector<Particle> newParticles(0);
  for (int i=0; i < parameters->getNumberParticles(); i++) {
    newParticles.push_back(Particle((system->getParticle(i))->diameter()));
  }

  double selfPropulsion; // self-propulsion force
  double noise; // noise realisation

  #if HEUN // HEUN'S SCHEME

  double selfPropulsionCorrection; // correction to the self-propulsion force

  std::vector<double> positions (2*parameters->getNumberParticles(), 0.0); // positions backup
  std::vector<double> forces (2*parameters->getNumberParticles(), 0.0); // forces backup
  #endif

  for (int iter=0; iter < Niter; iter++) {

    // COMPUTATION
    for (int i=0; i < parameters->getNumberParticles(); i++) {

      // POSITIONS
      for (int dim=0; dim < 2; dim++) {
        // initialise velocity
        (system->getParticle(i))->velocity()[dim] = 0.0;
        // initialise new positions with previous ones
        newParticles[i].position()[dim] =
          (system->getParticle(i))->position()[dim];
        // add self-propulsion
        selfPropulsion =
          parameters->getPropulsionVelocity()*
          cos((system->getParticle(i))->orientation()[0] - dim*M_PI/2);
        (system->getParticle(i))->velocity()[dim] += selfPropulsion;
        newParticles[i].position()[dim] +=
          parameters->getTimeStep()*selfPropulsion;
        // add noise
        noise = (system->getRandomGenerator())->gauss_cutoff();
        (system->getParticle(i))->velocity()[dim] +=
          sqrt(2.0*parameters->getTransDiffusivity())
          *noise;
        newParticles[i].position()[dim] +=
          sqrt(parameters->getTimeStep()
            *2.0*parameters->getTransDiffusivity())
          *noise;
        // initialise force
        (system->getParticle(i))->force()[dim] = 0.0;
      }

      // ORIENTATIONS
      // initialise new orientation with previous one
      newParticles[i].orientation()[0] =
        (system->getParticle(i))->orientation()[0];
      // add noise
      newParticles[i].orientation()[0] +=
        sqrt(parameters->getTimeStep()*2.0*parameters->getRotDiffusivity())
          *(system->getRandomGenerator())->gauss_cutoff();
    }

    // FORCES
    ABP_WCA<System0>(system); // compute forces

    for (int i=0; i < parameters->getNumberParticles(); i++) {
      for (int dim=0; dim < 2; dim++) {
        (system->getParticle(i))->velocity()[dim] +=
          (system->getParticle(i))->force()[dim]
          *parameters->getPotentialParameter(); // add force
        newParticles[i].position()[dim] +=
          (system->getParticle(i))->force()[dim]
          *parameters->getTimeStep()*parameters->getPotentialParameter(); // add force displacement
      }
    }

    // HEUN'S SCHEME
    #if HEUN
    for (int i=0; i < parameters->getNumberParticles(); i++) {

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
    ABP_WCA<System0>(system); // re-compute forces

    for (int i=0; i < parameters->getNumberParticles(); i++) {

      // CORRECTION TO INTERPARTICLE FORCE
      for (int dim=0; dim < 2; dim++) {
        (system->getParticle(i))->velocity()[dim] +=
          ((system->getParticle(i))->force()[dim] - forces[2*i + dim])
          *parameters->getPotentialParameter(); // velocity
        newParticles[i].position()[dim] +=
          ((system->getParticle(i))->force()[dim] - forces[2*i + dim])
          *parameters->getTimeStep()*parameters->getPotentialParameter(); // position
        (system->getParticle(i))->force()[dim] =
          ((system->getParticle(i))->force()[dim] + forces[2*i + dim])/2; // force
      }

      // CORRECTION TO SELF-PROPULSION FORCE
      for (int dim=0; dim < 2; dim++) {
        selfPropulsionCorrection =
          parameters->getPropulsionVelocity()*
          (cos(newParticles[i].orientation()[0] - dim*M_PI/2)
          - cos((system->getParticle(i))->orientation()[0] - dim*M_PI/2))
          /2;
        (system->getParticle(i))->velocity()[dim] +=
          selfPropulsionCorrection; // velocity
        newParticles[i].position()[dim] +=
          parameters->getTimeStep()*selfPropulsionCorrection; // position
      }

      // RESET INITIAL POSITIONS
      for (int dim=0; dim < 2; dim++) {
        (system->getParticle(i))->position()[dim] = positions[2*i + dim]; // position
      }
    }
    #endif

    // SAVE AND COPY
    system->saveNewState(newParticles);
  }
}


/////////////////////////////////
// INTERACTING BROWNIAN ROTORS //
/////////////////////////////////

void iterate_rotors(Rotors* rotors, int Niter) {
  // Updates system to next step according to the dynamics of interacting
  // Brownian rotors.

  bool const considerTorque = ( rotors->getTorqueParameter() != 0 );

  std::vector<double> newOrientations(rotors->getNumberParticles());

  #if HEUN // HEUN'S SCHEME
  std::vector<double> orientations(rotors->getNumberParticles(), 0.0); // orientations backup
  std::vector<double> torques(rotors->getNumberParticles(), 0.0); // torques backup
  #endif

  for (int iter=0; iter < Niter; iter++) {

    // COMPUTATION
    for (int i=0; i < rotors->getNumberParticles(); i++) {
      // initialise new orientations with previous ones
      newOrientations[i] = rotors->getOrientation(i)[0];
      // reset torques
      rotors->getTorque(i)[0] = 0.0;
      // add noise
      newOrientations[i] +=
        sqrt(2.0*rotors->getRotDiffusivity()*rotors->getTimeStep())
        *(rotors->getRandomGenerator())->gauss_cutoff();
    }
    // compute aligning torques
    if ( considerTorque ) {
      aligningTorque<Rotors>(rotors,
        [&rotors](int index) {
          return rotors->getOrientation(index); },
        [&rotors](int index) {
          return rotors->getTorque(index); }); // compute torques
    }
    // add torque
    for (int i=0; i < rotors->getNumberParticles(); i++) {
      newOrientations[i] +=
        rotors->getTorque(i)[0]*rotors->getTimeStep();
    }

    // HEUN'S SCHEME
    #if HEUN
    for (int i=0; i < rotors->getNumberParticles(); i++) {
      // ORIENTATIONS
      orientations[i] = rotors->getOrientation(i)[0]; // save initial orientation
      rotors->getOrientation(i)[0] = newOrientations[i]; // integration orientation as if using Euler's scheme
      // TORQUES
      torques[i] = rotors->getTorque(i)[0]; // save computed torque at initial orientation
      rotors->getTorque(i)[0] = 0; // re-initialise torques
    }
    // re-compute aligning torques
    if ( considerTorque ) {
      aligningTorque<Rotors>(rotors,
        [&rotors](int index) {
          return rotors->getOrientation(index); },
        [&rotors](int index) {
          return rotors->getTorque(index); }); // compute torques
    }
    for (int i=0; i < rotors->getNumberParticles(); i++) {
      // correction to orientations
      newOrientations[i] +=
        (rotors->getTorque(i)[0] - torques[i])*rotors->getTimeStep()
        /2;
      // correction to torques
      rotors->getTorque(i)[0] =
        (rotors->getTorque(i)[0] + torques[i])
        /2;
      // reset initial orientations
      rotors->getOrientation(i)[0] = orientations[i];
    }
    #endif

    // SAVE
    rotors->saveNewState(newOrientations);
  }
}

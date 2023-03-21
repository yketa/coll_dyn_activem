#ifndef ITERATION_HPP
#define ITERATION_HPP

#include <cmath>
#include <math.h>
#include <vector>

#include "maths.hpp"
#include "particle.hpp"

// FUNCTIONS

template<class SystemClass> void confinement_WCA(SystemClass* system) {
  // Compute interactions with WCA circular walls.

  if ( ! system->isConfined() ) { return; }

  double coeff = (system->getParameters())->getPotentialParameter();
  double r[2];
  double rn, dr;
  for (int i=0; i < system->getNumberParticles(); i++) {
    for (int dim=0; dim < 2; dim++) {
      r[dim] = algDistPeriod(
        system->getSystemSize()/2, (system->getParticle(i))->position()[dim],
        system->getSystemSize());
    }
    rn = sqrt(r[0]*r[0] + r[1]*r[1]);
    dr = 2*((system->getParameters())->getConfiningLength()/2 - rn)
      /(system->getParticle(i))->diameter();
    if ( dr < pow(2., 1./6.) ) {
      for (int dim=0; dim < 2; dim++) {
        std::cerr << "init: " << (system->getParticle(i))->force()[dim] << std::endl;
        (system->getParticle(i))->force()[dim] +=
          -(coeff/(system->getParticle(i))->diameter())
            *(192/pow(1/dr, 13) - 96/pow(1/dr, 7))*(r[dim]/rn);
        std::cerr << (system->getParticle(i))->force()[dim] << std::endl << std::endl;
      }
    }
  }
}

template<class SystemClass> void system_WCA(SystemClass* system) {
  // Compute interactions with WCA potentials between all particles of the
  // system.

  double force[2];
  double diff[2];
  pairs_system<SystemClass>(system, // pairwise interaction forces
    [&system, &force, &diff](int index1, int index2)
      { add_WCA_force<SystemClass>(system, index1, index2,
          &force[0], &diff[0]);
        });
  confinement_WCA<SystemClass>(system); // confining forces
}

template<class SystemClass, typename F, typename G> void aligningTorque(
  SystemClass* system, F getOrientation, G getTorque) {
  // Compute aligning torques between all particles of the system given a
  // function returning a pointer to the orientation and an other to the applied
  // torque on a particle specified by its index.

  std::vector<double> orientations(0);
  for (int i=0; i < system->getNumberParticles(); i++) {
    orientations.push_back(getOrientation(i)[0]);
  }
  std::vector<double> order = getOrderParameter(orientations);
  double nu = sqrt(pow(order[0], 2.0) + pow(order[1], 2.0));
  double phi = getAngle(order[0]/nu, order[1]);

  for (int i=0; i < system->getNumberParticles(); i++) {
    getTorque(i)[0] +=
      2*system->getTorqueParameter()*nu*sin(orientations[i] - phi);
  }
}

///////////////////////////////
// ACTIVE BROWNIAN PARTICLES //
///////////////////////////////

template<class SystemClass> void iterate_ABP_WCA(
  SystemClass* system, int Niter) {
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
    system_WCA<SystemClass>(system); // compute forces

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
    system->updateCellList(); // update cell list since positions have changed
    system_WCA<SystemClass>(system); // re-compute forces

    for (int i=0; i < parameters->getNumberParticles(); i++) {

      // CORRECTION TO INTERPARTICLE FORCE
      for (int dim=0; dim < 2; dim++) {
        (system->getParticle(i))->velocity()[dim] +=
          ((system->getParticle(i))->force()[dim] - forces[2*i + dim])
          *parameters->getPotentialParameter()/2; // velocity
        newParticles[i].position()[dim] +=
          ((system->getParticle(i))->force()[dim] - forces[2*i + dim])
          *parameters->getTimeStep()*parameters->getPotentialParameter()/2; // position
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

    // SELF-PROPULSION VECTORS
    for (int i=0; i < parameters->getNumberParticles(); i++) {
      for (int dim=0; dim < 2; dim++) {
        newParticles[i].propulsion()[dim] = parameters->getPropulsionVelocity()
          *cos(newParticles[i].orientation()[0] - dim*M_PI/2);
      }
    }

    // SAVE AND COPY
    system->saveNewState(newParticles);
  }
}


/////////////////////////////////////////
// ACTIVE ORNSTEIN-UHLENBECK PARTICLES //
/////////////////////////////////////////

template<class SystemClass> void iterate_AOUP_WCA(
  SystemClass* system, int Niter) {
  // Updates system to next step according to the dynamics of active
  // Ornstein-Uhlenbeck particles with WCA potential.

  Parameters* parameters = system->getParameters();

  std::vector<Particle> newParticles(0);
  for (int i=0; i < parameters->getNumberParticles(); i++) {
    newParticles.push_back(Particle((system->getParticle(i))->diameter()));
  }

  #if HEUN // HEUN'S SCHEME

  double selfPropulsionCorrection; // correction to the self-propulsion force

  std::vector<double> positions (2*parameters->getNumberParticles(), 0.0); // positions backup
  std::vector<double> forces (2*parameters->getNumberParticles(), 0.0); // forces backup
  #endif

  for (int iter=0; iter < Niter; iter++) {

    // COMPUTATION
    for (int i=0; i < parameters->getNumberParticles(); i++) {

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
          parameters->getTimeStep()*(system->getParticle(i))->propulsion()[dim];
        // initialise force
        (system->getParticle(i))->force()[dim] = 0.0;

        // SELF-PROPULSION VECTORS
        // initialise new self-propulsion vectors with previous ones
        newParticles[i].propulsion()[dim] =
          (system->getParticle(i))->propulsion()[dim];
        // add drift
        newParticles[i].propulsion()[dim] +=
          -parameters->getTimeStep()*parameters->getRotDiffusivity()
          *(system->getParticle(i))->propulsion()[dim];
        // add diffusion
        newParticles[i].propulsion()[dim] +=
          sqrt(2.0*parameters->getTimeStep()
            *pow(parameters->getRotDiffusivity(), 2.0)
            *parameters->getTransDiffusivity())
          *(system->getRandomGenerator())->gauss_cutoff();
      }
    }

    // FORCES
    system_WCA<SystemClass>(system); // compute forces

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
    system->updateCellList(); // update cell list since positions have changed
    system_WCA<SystemClass>(system); // re-compute forces

    for (int i=0; i < parameters->getNumberParticles(); i++) {

      // CORRECTION TO INTERPARTICLE FORCE
      for (int dim=0; dim < 2; dim++) {
        (system->getParticle(i))->velocity()[dim] +=
          ((system->getParticle(i))->force()[dim] - forces[2*i + dim])
          *parameters->getPotentialParameter()/2; // velocity
        newParticles[i].position()[dim] +=
          ((system->getParticle(i))->force()[dim] - forces[2*i + dim])
          *parameters->getTimeStep()*parameters->getPotentialParameter()/2; // position
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
          parameters->getTimeStep()*selfPropulsionCorrection; // position
        newParticles[i].propulsion()[dim] +=
          -parameters->getTimeStep()*parameters->getRotDiffusivity()
          *selfPropulsionCorrection; // self-propulsion vector
      }

      // RESET INITIAL POSITIONS
      for (int dim=0; dim < 2; dim++) {
        (system->getParticle(i))->position()[dim] = positions[2*i + dim]; // position
      }
    }
    #endif

    // ORIENTATION
    for (int i=0; i < parameters->getNumberParticles(); i++) {
      newParticles[i].orientation()[0] = getAngleVector(
          newParticles[i].propulsion()[0], newParticles[i].propulsion()[1]);
    }

    // SAVE AND COPY
    system->saveNewState(newParticles);
  }
}


/////////////////////////////////
// INTERACTING BROWNIAN ROTORS //
/////////////////////////////////

void iterate_rotors(Rotors* rotors, int Niter);
  // Updates system to next step according to the dynamics of interacting
  // Brownian rotors.


#endif

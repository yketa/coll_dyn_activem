#ifndef ITERATION_HPP
#define ITERATION_HPP

#include "particle.hpp"

// ABPs

void iterate_ABP_WCA(System* system, int Niter);
  // Updates system to next step according to the dynamics of active Brownian
  // particles with WCA potential, using custom dimensionless parameters
  // relations.

void iterate_ABP_WCA(System0* system, int Niter);
  // Updates system to next step according to the dynamics of active Brownian
  // particles with WCA potential.

// AOUPs

void iterate_AOUP_WCA(System0* system, int Niter);
  // Updates system to next step according to the dynamics of active
  // Ornstein-Uhlenbeck particles with WCA potential.

// ROTORS

void iterate_rotors(Rotors* rotors, int Niter);
  // Updates system to next step according to the dynamics of interacting
  // Brownian rotors.

// FUNCTIONS

template<class SystemClass> void system_WCA(SystemClass* system) {
  // Compute interactions with WCA potentials between all particles of the
  // system.

  pairs_system<SystemClass>(system,
    [&system](int index1, int index2) { system->WCA_force(index1, index2); });
}

template<class SystemClass, typename F, typename G> void aligningTorque(
  SystemClass* system, F getOrientation, G getTorque) {
  // Compute aligning torques between all particles of the system given a
  // function returning a pointer to the orientation and an other to the applied
  // torque on a particle specified by its index.

  double torque;
  for (int i=0; i < system->getNumberParticles(); i++) {
    for (int j=i + 1; j < system->getNumberParticles(); j++) {
      torque = 2.0*system->getTorqueParameter()/system->getNumberParticles()
        *sin(getOrientation(i)[0] - getOrientation(j)[0]);
      getTorque(i)[0] += torque;
      getTorque(j)[0] -= torque;
    }
  }
}

#endif

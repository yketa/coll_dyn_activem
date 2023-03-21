#ifndef FIRE_HPP
#define FIRE_HPP

#include <math.h>
#include <algorithm>
#include <vector>

#include "iteration.hpp"
#include "particle.hpp"

#ifdef DEBUG
#include <iostream>
#endif

////////////////
// PROTOTYPES //
////////////////

template<class SystemClass> void FIRE_WCA(
  SystemClass* system, long double Emin, int iterMax, long double dtmin = 0,
  long double dt0 = 0, long double dtmax = 0,
  long double finc = 1.1, long double fdec = 0.5,
  long double alpha0 = 0.1, long double falpha = 0.99,
  int Nmin = 5) {
  // Uses the FIRE algorithm to minimise the WCA potential energy of a given
  // system.
  // (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#FIRE%20algorithm)

  #ifdef DEBUG
  std::cerr << "##FIRE minimisation algorithm" << std::endl;
  std::cerr << "#Emin: " << Emin << " iterMax: " << iterMax;
  std::cerr << " dtmin: " << dtmin << " Nmin: " << Nmin;
  std::cerr << std::endl;
  #endif

  // INITIALISATION

  std::vector<double> velocities (2*system->getNumberParticles(), 0.0); // array of global velocities
  long double normVelocities; // norm of global velocities
  long double normForces; // norm of global forces
  double power; // total power of interacting forces

  if ( dt0 <= 0 ) dt0 = system->getTimeStep(); // initial time step
  if ( dtmax <= 0 ) dtmax = 10*dt0; // maximum time step
  #ifdef DEBUG
  std::cerr << "#dt0: " << dt0 << " dtmax: " << dtmax;
  std::cerr << " finc: " << finc << " fdec: " << fdec << std::endl;
  std::cerr << "#alpha0: " << alpha0 << " falpha: " << falpha << std::endl;
  std::cerr << std::endl;
  #endif

  long double dt = dt0; // time step
  long double alpha = alpha0; // strength of force towards steepest descent

  // MINIMISATION LOOP

  double NPpos = 0; // number of consecutive steps with positive power
  double iter = 0; // number of iterations
  double potential = WCA_potential<SystemClass>(system);
  while ( potential > Emin && iter < iterMax && dt > dtmin ) { // while energy is not minimised and time step is still big for a maximum of iterMax iterations
    iter++;
    #ifdef DEBUG
    std::cerr << "#iter: " << iter << " E: " << potential << std::endl;
    #endif

    // COMPUTE FORCES
    for (int i = 0; i < system->getNumberParticles(); i++) {
      for (int dim = 0; dim < 2; dim++) {
        (system->getParticle(i))->force()[dim] = 0.0; // reset force
      }
    }
    system_WCA<SystemClass>(system);

    // COMPUTE POWER
    power = 0;
    for (int i = 0; i < system->getNumberParticles(); i++) { // loop over particles
      for (int dim = 0; dim < 2; dim++) { // loop over dimensions
        power += (system->getParticle(i))->force()[dim]*velocities[2*i + dim];
      }
    }
    #ifdef DEBUG
    std::cerr << "#power: " << power << std::endl;
    #endif

    // P > 0
    if ( power > 0 ) {
      #ifdef DEBUG
      std::cerr << "## P > 0" << std::endl;
      #endif
      NPpos++;
      for (int i = 0; i < system->getNumberParticles(); i++) {

        // compute norms of velocity and force
        normVelocities = 0;
        normForces = 0;
        for (int dim = 0; dim < 2; dim++) {
          normVelocities += pow(velocities[2*i + dim], 2);
          normForces += pow((system->getParticle(i))->force()[dim], 2);
        }
        normVelocities = sqrt(normVelocities);
        normForces = sqrt(normForces);

        // modify velocity
        if ( normForces > 0 ) {
          for (int dim = 0; dim < 2; dim++) {
            velocities[2*i + dim] =                          // v_{t+1} =
              (1.0 - alpha)*velocities[2*i + dim]            // (1 - alpha)*v_t
              + alpha*(system->getParticle(i))->force()[dim] // + alpha*F(r_t)
                *normVelocities/normForces;                  // *|v_t|/|F(r_t)|
          }
        }
        else {
          for (int dim = 0; dim < 2; dim++) {
            velocities[2*i + dim] =                // v_{t+1} =
              (1.0 - alpha)*velocities[2*i + dim]; // (1 - alpha)*v_t
          }
        }
      }

      // maximum number of iterations at positive power
      if ( NPpos > Nmin ) {
        NPpos = 0;
        dt = std::min(dt*finc, dtmax);
        alpha = falpha*alpha;
      }
    }

    // P <= 0
    else {
      #ifdef DEBUG
      std::cerr << "## P <= 0" << std::endl;
      #endif
      NPpos = 0;

      std::vector<double> velocities (2*system->getNumberParticles(), 0.0);
      dt *= fdec;
      alpha = alpha0;
    }

    #ifdef DEBUG
    std::cerr << "#dt: " << dt << " alpha: " << alpha << std::endl << std::endl;
    #endif

    // INTEGRATION TO NEXT TIME STEP [EULER EXPLICIT METHOD]
    for (int i = 0; i < system->getNumberParticles(); i++) {
      for (int dim = 0; dim < 2; dim++) {
        // positions
        (system->getParticle(i))->position()[dim] += dt*velocities[2*i + dim];
        (system->getParticle(i))->position()[dim] -=
          system->getSystemSize()*wrapCoordinate<SystemClass>(system, // taking boundary condition into account
            (system->getParticle(i))->position()[dim]);
        // velocities
        velocities[2*i + dim] += dt*(system->getParticle(i))->force()[dim];
      }
    }

    // UPDATE CELL LIST
    system->updateCellList();

    // POTENTIAL ENERGY AT NEXT TIME STEP
    potential = WCA_potential<SystemClass>(system);
  }
}

#endif

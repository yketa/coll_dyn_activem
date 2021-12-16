#ifndef ADD_HPP
#define ADD_HPP

#include <cmath>
#include <vector>
#include <limits>

#include "alglib.hpp"
#include "maths.hpp"
#include "particle.hpp"
#include "readwrite.hpp"

/////////////
// CLASSES //
/////////////

class ADD;


/*  ADD
 *  ---
 *  Provides methods to evolve the activity-driven dynamics of AOUPs.
 *  Uses the ALGLIB library to perform minimisations of the effective
 *  potential.
 */

class ADD {

  public:

    // CONSTRUCTORS

    ADD
      (int const& N, double const& L, std::vector<double> const& sigma,
        double const& f,
        double const& dt, int const& init, int const& Niter,
          int const& dtMin, int const& dtMax,
          int const& nMax, int const& intMax,
        int const& seed = 0, std::string filename = "") :
      numberParticles(N), systemSize(L), diameters(sigma),
      positions(2*numberParticles, 0),
      propulsions(2*numberParticles, 0), propulsions0(2*numberParticles, 0),
      randomSeed(seed), randomGenerator(randomSeed), cellList(),
      velocity(f), timeStep(dt),
      output(filename != "" ? filename + ".add" : ""),
      system(
        init, Niter, dtMin, new int(dtMax), nMax, intMax,
          new std::vector<int>(), new std::vector<int>(),
        new Parameters(
          numberParticles, 1, velocity, 0, 0, 0,
            numberParticles/pow(systemSize, 2),
          const_cast<std::vector<double>&>(diameters), systemSize, timeStep),
        const_cast<std::vector<double>&>(diameters), seed, filename),
      iterMax(100*numberParticles) {

      // propulsions
      for (int i=0; i < numberParticles; i++) {
        for (int dim=0; dim < 2; dim++) {
          propulsions[2*i + dim] = velocity*randomGenerator.gauss(0, 1);
        }
      }

      // cell list
      cellList = CellList(numberParticles, systemSize,
          rcut*(*std::max_element(diameters.begin(), diameters.end())));
    }

    // DESTRUCTORS

    ~ADD() {;}

    // METHODS

    int const getNumberParticles() { return numberParticles; } // returns number of particles
    double const getSystemSize() { return systemSize; } // returns size of the system
    std::vector<double> const getDiameters() { return diameters; } // returns vector of diameters

    double* getPosition(int const& index) { return &positions[2*index]; } // return pointer to position
    std::vector<double*> getPositions() {
      // Returns vector of pointers to positions.

      std::vector<double*> positionsPTR(0);
      for (int i=0; i < numberParticles; i++) {
        positionsPTR.push_back(getPosition(i));
      }
      return positionsPTR;
    }
    double* getPropulsion(int const& index) { return &propulsions[2*index]; } // return pointer to propulsion
    std::vector<double*> getPropulsions() {
      // Returns vector of pointers to propulsions.

      std::vector<double*> propulsionsPTR(0);
      for (int i=0; i < numberParticles; i++) {
        propulsionsPTR.push_back(&propulsions[2*i]);
      }
      return propulsionsPTR;
    }

    int const getRandomSeed() { return randomSeed; } // returns random seed
    Random* getRandomGenerator() { return &randomGenerator; } // returns pointer to random generator

    CellList* getCellList() { return &cellList; } // returns pointer to cell list
    void updateCellList() { cellList.listConstructor<double*>(getPositions()); } // updates cell list with positions

    double const getVelocity() { return velocity; } // returns velocity

    Write* getOuput() { return &output; } // returns pointer to output object
    SystemN* getSystem() { return &system; } // returns pointer ot system
    void saveInitialState() {
      // Saves first frame.

      for (int i=0; i < numberParticles; i++) {
        for (int dim=0; dim < 2; dim++) {
          (system.getParticle(i))->position()[dim] = positions[2*i + dim];
          (system.getParticle(i))->propulsion()[dim] = propulsions[2*i + dim];
        }
        (system.getParticle(i))->orientation()[0] =
          getAngleVector(propulsions[2*i], propulsions[2*i + 1]);
      }
      system.saveInitialState();
    }
    void saveNewState() {
      // Saves new frame.

      std::vector<Particle> newParticles;
      for (int i=0; i < numberParticles; i++) {
        newParticles.push_back(system.getParticle(i));
        for (int dim=0; dim < 2; dim++) {
          newParticles[i].position()[dim] += // WARNING: we assume that particles do not move more further than a half box length
            algDistPeriod( // equivalent position at distance lower than half box
              newParticles[i].position()[dim],
              positions[2*i + dim] // wrapped coordinate
                - (wrapCoordinate<SystemN>(&system, positions[2*i + dim])
                  *systemSize),
              systemSize);
          newParticles[i].propulsion()[dim] = propulsions[2*i + dim];
        }
        newParticles[i].orientation()[0] =
          getAngleVector(propulsions[2*i], propulsions[2*i + 1]);
      }
      system.saveNewState(newParticles);
      for (int i=0; i < numberParticles; i++) {
        for (int dim=0; dim < 2; dim++) {
          positions[2*i + dim] = (system.getParticle(i))->position()[dim];
        }
      }
    }

    std::vector<double> difference(const double* r0) {
      // Returns vectors of displacements from `positions' to `r0'.

      std::vector<double> dr(2*numberParticles, 0);
      for (int dim=0; dim < 2; dim++) {
        for (int i=0; i < numberParticles; i++) {
          dr[2*i + dim] = algDistPeriod(
            r0[2*i + dim],
            positions[2*i + dim] // wrapped coordinate
              - (wrapCoordinate<SystemN>(&system, positions[2*i + dim])
                *systemSize),
            systemSize);
        }
      }
      return dr;
    }

    double difference2(const double* r0) {
      // Returns squared displacements from `positions' to `r0'.

      const std::vector<double> dr = difference(r0);
      double dr2 = 0;
      for (int i=0; i < numberParticles; i++) {
        for (int dim=0; dim < 2; dim++) {
          dr2 += pow(dr[2*i + dim], 2);
        }
      }
      return dr2;
    }

    double potential() {
      // Returns potential energy.

      // parameters
      std::vector<double> const* sigma = &diameters;
      std::vector<double*> rPTR = getPositions();
      double const L = systemSize;
      double const A = a;
      double const C0 = c0;
      double const C1 = c1;
      double const C2 = c2;
      double const RCUT = rcut;

      // potential
      double U = 0;
      updateCellList();
      cellList.pairs(
        [&U, &sigma, &rPTR, &L, &A, &C0, &C1, &C2, &RCUT]
        (int const& index1, int const& index2) { // do for each individual pair
          // rescaled diameter
          double sigmaij = (sigma->at(index1) + sigma->at(index2))/2
            *(1 - 0.2*fabs(sigma->at(index1) - sigma->at(index2)));
          // distance
          double diff[2];
          double dist =
            dist2DPeriod(rPTR[index1], rPTR[index2], L, &diff[0]);
          // potential
          if ( dist/sigmaij < RCUT ) {
            // rescaled distances
            double rAinv = 1./pow((dist/sigmaij), A);
            double r2 = pow((dist/sigmaij), 2);
            double r4 = r2*r2;
            // potential
            U += rAinv + C0 + C1*r2 + C2*r4;
          }
        });
      return U;
    }

    std::vector<double> gradientUeff() {
      // Returns effective potential energy gradient.

      // parameters
      std::vector<double> const* sigma = &diameters;
      std::vector<double*> rPTR = getPositions();
      double const L = systemSize;
      double const A = a;
      double const C0 = c0;
      double const C1 = c1;
      double const C2 = c2;
      double const RCUT = rcut;

      // potential
      std::vector<double> grad(2*numberParticles, 0);
      // propulsion part
      double av_prop[2] = {0, 0};
      for (int dim=0; dim < 2; dim++) {
        for (int i=0; i < numberParticles; i++) {
          grad[2*i + dim] = -propulsions[2*i + dim]; // actual gradient of propulsion
          av_prop[dim] += propulsions[2*i + dim]/numberParticles; // average propulsion
        }
        for (int i=0; i < numberParticles; i++) {
          grad[2*i + dim] += av_prop[dim]; // correction to gradient from average propulsion
        }
      }
      // interaction part
      updateCellList();
      cellList.pairs(
        [&grad, &sigma, &rPTR, &L, &A, &C0, &C1, &C2, &RCUT]
        (int const& index1, int const& index2) { // do for each individual pair
          // rescaled diameter
          double sigmaij = (sigma->at(index1) + sigma->at(index2))/2
            *(1 - 0.2*fabs(sigma->at(index1) - sigma->at(index2)));
          // distance
          double diff[2];
          double dist =
            dist2DPeriod(rPTR[index1], rPTR[index2], L, &diff[0]);
          // potential
          if ( dist/sigmaij < RCUT ) {
            // rescaled distances
            double rAinv = 1./pow((dist/sigmaij), A);
            double r2 = pow((dist/sigmaij), 2);
            double r4 = r2*r2;
            // gradient of potential
            for (int dim=0; dim < 2; dim++) {
              grad[2*index1 + dim] +=
                (diff[dim]/(dist*dist))*(A*rAinv - 2*C1*r2 - 4*C2*r4);
              grad[2*index2 + dim] +=
                -(diff[dim]/(dist*dist))*(A*rAinv - 2*C1*r2 - 4*C2*r4);
            }
          }
        });
      return grad;
    }

    double gradientUeff2(std::vector<double> const& grad) {
      // Returns squared norm of effective potential energy gradient.

      double grad2 = 0;
      for (double g : grad) { grad2 += pow(g, 2); }
      return grad2;
    }
    double gradientUeff2() {
      // Returns squared norm of effective potential energy gradient.

      const std::vector<double> grad = gradientUeff();
      return gradientUeff2(grad);
    }

    void minimiseUeff(int const& iter = 0) {
      // Minimises effective potential with respect to positions.

      // parameters
      CellList* cl = &cellList;
      std::vector<double> const* sigma = &diameters;
      std::vector<double> const r0 = positions;
      std::vector<double*> rPTR = getPositions();
      std::vector<double>* prop = &propulsions;
      int const N = numberParticles;
      double const L = systemSize;
      double const A = a;
      double const C0 = c0;
      double const C1 = c1;
      double const C2 = c2;
      double const RCUT = rcut;
      double const potential0 = potential();
      std::vector<double> gradUeff; // gradient of effective potential
      double gradUeff2; // squared gradient of effective potential
      #ifndef ADD_MD
      //////////////////////////////////
      // MINIMISATION USING CG (+ MD) //
      //////////////////////////////////
      // std::cout << "ADD-CG—————" << std::endl;
      // potential
      auto potential_force =
        [&cl, &sigma, &r0, &rPTR, &prop, &N, &L, &A, &C0, &C1, &C2, &RCUT]
        (double* r, double* U, double* gradU) {
          // positions
          for (int i=0; i < N; i++) {
            for (int dim=0; dim < 2; dim++) {
              rPTR[i][dim] = r[2*i + dim]; // `rPTR' should already point to `positions'
            }
          }
          cl->listConstructor<double*>(rPTR);
          // propulsion part
          U[0] = 0;
          double av_prop[2] = {0, 0};
          for (int dim=0; dim < 2; dim++) {
            for (int i=0; i < N; i++) {
              U[0] += -prop->at(2*i + dim)
                *algDistPeriod(r0[2*i + dim], rPTR[i][dim], L);
              gradU[2*i + dim] = -prop->at(2*i + dim); // actual gradient of propulsion
              av_prop[dim] += prop->at(2*i + dim)/N; // average propulsion
            }
            for (int i=0; i < N; i++) {
              U[0] += av_prop[dim]
                *algDistPeriod(r0[2*i + dim], rPTR[i][dim], L); // correction to potential from average propulsion
              gradU[2*i + dim] += av_prop[dim]; // correction to gradient from average propulsion
            }
          }
          // repulsive part
          cl->pairs(
            [&U, &gradU, &sigma, &rPTR, &L, &A, &C0, &C1, &C2, &RCUT]
            (int const& index1, int const& index2) { // do for each individual pair
              // rescaled diameter
              double sigmaij = (sigma->at(index1) + sigma->at(index2))/2
                *(1 - 0.2*fabs(sigma->at(index1) - sigma->at(index2)));
              // distance
              double diff[2];
              double dist =
                dist2DPeriod(rPTR[index1], rPTR[index2], L, &diff[0]);
              // potential
              if ( dist/sigmaij < RCUT ) {
                // rescaled distances
                double rAinv = pow((sigmaij/dist), A);
                double r2 = pow((dist/sigmaij), 2);
                double r4 = r2*r2;
                // potential
                U[0] += rAinv + C0 + C1*r2 + C2*r4;
                // gradient of potential
                for (int dim=0; dim < 2; dim++) {
                  gradU[2*index1 + dim] +=
                    (diff[dim]/(dist*dist))*(A*rAinv - 2*C1*r2 - 4*C2*r4);
                  gradU[2*index2 + dim] +=
                    -(diff[dim]/(dist*dist))*(A*rAinv - 2*C1*r2 - 4*C2*r4);
                }
              }
            });
      };
      // minimisation
      alglib::mincgreport report;
      CGMinimiser Uminimiser(potential_force, 2*numberParticles,
        pow(gradMax, 2)/numberParticles, 0, 0, iter > 0 ? iter : iterMax);
      report = Uminimiser.minimise(&positions[0]);
      int termination = report.terminationtype;
      int iterations = report.iterationscount;
      // MD on failure
      gradUeff = gradientUeff();
      gradUeff2 = gradientUeff2(gradUeff);
      if (
        termination == 5 || termination == 7
        #ifndef ADD_NO_LIMIT
        || difference2(&(r0[0])) > numberParticles*dr2Max
        #endif
        || sqrt(gradUeff2/numberParticles) > gradMax ) {
        std::cerr << "[minimisation failure] sqrt(gradUeff2/N) = "
          << sqrt(gradUeff2/numberParticles) << std::endl;
        // restart from initial positions
        for (int i=0; i < numberParticles; i++) {
          for (int dim=0; dim < 2; dim++) {
            positions[2*i + dim] = r0[2*i + dim];
          }
        }
        gradUeff = gradientUeff();
        gradUeff2 = gradientUeff2(gradUeff);
        // perform MD
        int iterMD = 0;
        while (
          iterMD < iterMaxMD
          && sqrt(gradUeff2/numberParticles) > gradMaxMD ) {
          std::cerr << "MD: " << iterMD << std::endl
            << "sqrt(gradUeff2/N) = " << sqrt(gradUeff2/numberParticles)
            << std::endl;
          for (int step=0; step < iterMinMD; step++) {
            for (int i=0; i < numberParticles; i++) {
              for (int dim=0; dim < 2; dim++) {
                positions[2*i + dim] -= dtMD*gradUeff[2*i + dim];
              }
            }
            gradUeff = gradientUeff();
            iterMD++;
          }
          gradUeff2 = gradientUeff2(gradUeff);
        }
        iterations += iterMD;
        // re-minimise
        CGMinimiser Uminimiser(potential_force, 2*numberParticles,
          pow(gradMax, 2)/numberParticles, 0, 0, iter > 0 ? iter : iterMax);
        report = Uminimiser.minimise(&positions[0]);
        termination = (int) report.terminationtype;
        iterations += (int) report.iterationscount;
        gradUeff2 = gradientUeff2();
      }
      // failures
      if ( termination == 5 ) {
        throw std::invalid_argument(
          "Maximum number of iterations ("
            + std::to_string(iterMax) + ") reached.");
      }
      else if ( sqrt(gradUeff2/numberParticles) > gradMax ) {
        throw std::invalid_argument(
          "Maximum scaled gradient of effective potential"
            + std::string("((|\\nabla U_eff|^2)^{1/2} = ")
            + std::to_string(gradMax) + ") exceeded.");
      }
      else if ( termination == 7 ) {
        throw std::invalid_argument(
          "Minimisation failed. (error code: 7)");
      }
      #else
      ///////////////////////////
      // MINIMISATION USING MD //
      ///////////////////////////
      // std::cout << "ADD-MD—————" << std::endl;
      int iterations = 0;
      gradUeff = gradientUeff();
      gradUeff2 = gradientUeff2(gradUeff);
      while (
        #ifndef ADD_NO_LIMIT
        iterations < iterMaxMD &&
        #endif
        sqrt(gradUeff2/numberParticles) > gradMax ) {
        for (int step=0; step < iterMinMD; step++) {
          for (int i=0; i < numberParticles; i++) {
            for (int dim=0; dim < 2; dim++) {
              positions[2*i + dim] -= dtMD*gradUeff[2*i + dim];
            }
          }
          gradUeff = gradientUeff();
          iterations++;
        }
        gradUeff2 = gradientUeff2(gradUeff);
      }
      int termination = iterations >= iterMaxMD ? 5 : 0;
      #endif
      // measurements
      double const potential1 = potential();
      double dEp = potential0 - potential1; // energy drop
      double dms = 0; // mean squared displacement
      double dr;
      double av_prop0[2] = {0, 0};
      for (int dim=0; dim < 2; dim++) {
        for (int i=0; i < numberParticles; i++) {
          av_prop0[dim] += propulsions0[2*i + dim]/numberParticles;
        }
        for (int i=0; i < numberParticles; i++) {
          dr = algDistPeriod(
            r0[2*i + dim],
            positions[2*i + dim] // wrapped coordinate
              - (wrapCoordinate<SystemN>(&system, positions[2*i + dim])
                *systemSize),
            systemSize);
          dEp += (propulsions0[2*i + dim] - av_prop0[dim])*dr;
          dms += pow(dr, 2)/numberParticles;
        }
      }
      // output
      output.write<int>(termination);
      output.write<int>(iterations);
      output.write<double>(potential1/numberParticles);
      output.write<double>(gradUeff2);
      output.write<double>(dEp);
      output.write<double>(sqrt(dms));
    }

    void iteratePropulsion() {
      // Perform an iteration over scaled time scale `timeStep' of the
      // propulsion vectors.

      for (int i=0; i < numberParticles; i++) {
        for (int dim=0; dim < 2; dim++) {
          propulsions0[2*i + dim] = propulsions[2*i + dim];
          propulsions[2*i + dim] = (1 - timeStep)*propulsions[2*i + dim]
            + velocity*sqrt(2*timeStep)*randomGenerator.gauss_cutoff();
        }
      }
    }

  private:

    int const numberParticles; // number of particles
    double const systemSize; // size of the system
    std::vector<double> const diameters; // diameters of particles

    std::vector<double> positions; // vector of 2N position coordinates
    std::vector<double> propulsions; // vector of 2N propulsion coordinates
    std::vector<double> propulsions0; // vector of previous 2N propulsion coordinates

    int const randomSeed; // random seed
    Random randomGenerator; // random number generator

    CellList cellList; // cell list

    double const velocity; // self-propulsion velocity
    double const timeStep; // integration time step

    Write output; // output object
    SystemN system; // system object saving purposes

    int const a = 12; // potential parameter
    double const rcut = 1.25; // potential cut-off radius
    double const c0 = -(8 + a*(a + 6))/(8*pow(rcut, a)); // constant part of potential
    double const c1 = (a*(a + 4))/(4*pow(rcut, a + 2)); // quadratic part of potential
    double const c2 = -(a*(a + 2))/(8*pow(rcut, a + 4)); // quartic part of potential

    double const gradMax = 1e-5; // tolerance for scaled gradient of effective potential

    double const dr2Max = 0.1; // tolerance for squared displacement per particle [CG minimisation]
    double const gradMaxMD = 1e-4; // threshold on scaled gradient of effective potential for molecular dynamics [(MD during) CG minimisation]
    long int const iterMax; // maximum number of minimisation iterations (0 => no limit) [(MD during) MD minimisation]

    double const dtMD = 5e-4; // time step for molecular dynamics
    int const iterMinMD = 1e3; // number of molecular dynamics steps before checking scaled gradient of effective potential
    int const iterMaxMD = 400/dtMD; // maximum number of molecular dynamics steps

};

#endif

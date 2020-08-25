#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cloningserial.hpp"
#include "env.hpp"
#include "particle.hpp"
#include "readwrite.hpp"

int main() {

	// cloning parameters
	double tmax = getEnvDouble("TMAX", 1); // dimensionless time to simulate
	int nc = getEnvInt("NC", 10); // number of clones
	double sValue = getEnvDouble("SVALUE", 0); // biasing parameter
	int seed = getEnvInt("SEED", 0); // master random seed
	int nRuns = getEnvInt("NRUNS", 1); // number of different runs
	int cloneMethod = 2; // should keep this set to 2 (!!)
	int initSim = getEnvInt("INITSIM", 1); // number of initial elementary number of iterations to "randomise" the systems

	// openMP parameters
	#ifdef _OPENMP
	int threads = getEnvInt("THREADS", -1); // number of threads
	printf("# compiled with openMP\n");
	if ( threads > 0 ) {
		printf("# setting threads %d\n",threads);
		omp_set_num_threads(threads);
	}
	printf("# running on %d threads\n",omp_get_max_threads());
	#endif

	// physical parameters
	int N = getEnvInt("N", 100); // number of rotors in the system
	double Dr = getEnvDouble("DR", 1.0/2.0); // rotational diffusivity

	// simulation parameters
	int tau = getEnvInt("TAU", 100); // elementary number of steps
	double dt = getEnvDouble("DT", 0.001); // time step

	// output to file
	std::string filename = getEnvString("FILE", ""); // output file name
	Write output(filename); // output class
	output.write<double>(tmax);
	output.write<int>(nc);
	output.write<double>(sValue);
	output.write<int>(seed);
	output.write<int>(nRuns);
	output.write<int>(cloneMethod);
	output.write<int>(initSim);
	output.write<int>(N);
	output.write<double>(Dr);
	output.write<int>(tau);
	output.write<double>(dt);
	output.write<int>(BIAS);

	// dummy system
	Rotors dummy(N, Dr, dt);

	printf("## CloningSerial Code: tmax %.3e numClones %d runs %d s %.3e tau %d Delta.t %.3e\n",tmax,nc,nRuns,sValue,tau,dt);

	// cloning object (default cloning method is [eq])
	CloningSerial<Rotors> clones(nc, tau, sValue, cloneMethod);

	// set up the clones etc, using dummySystem to get system sizes, hop rates, etc
	clones.init(&dummy, seed);
	std::cout << "## master seed " << seed << std::endl;

	#if BIAS == 0
	double sFactor = N*tau*dt;
	double sOffset = 0;
	#endif
	#if BIAS == 1
	#ifdef CONTROLLED_DYNAMICS
	double sFactor = sValue/Dr;
	double sOffset = tau*dt;
	#else
	double sFactor = N*tau*dt;
	double sOffset = 0;
	#endif
	#endif

	for (int run = 0; run<nRuns;run++) {

		// go! (this includes generating "random" [different] initial conditions for the clones)
		clones.doCloning(tmax, initSim,

			// ITERATION FUNCTION
			[](Rotors* rotors, int Niter) { iterate_rotors(rotors, Niter); },

			// GET WEIGHT FUNCTION
			[&sFactor, &sOffset](Rotors* rotors) {

				double sWeight = 0;

				// biasing with order parameter
				#if BIAS == 0
				sWeight = rotors->getBiasingParameter() // sw = s
					*rotors->getOrder()                   // *nu
					*sFactor;                             // *N*tau
				#endif
				// biasing with squared order parameter
				#if BIAS == 1
				#ifdef CONTROLLED_DYNAMICS
				sWeight = rotors->getBiasingParameter()           // sw = s
					*(sOffset - sFactor*rotors->getBiasIntegral()); // *(tau - s/Dr)*int nu^2 sin(theta-phi)^2)
				#else
				sWeight = rotors->getBiasingParameter() // sw = s
					*rotors->getOrderSq()                 // *nu^2
					*sFactor;                             // *N*tau
				#endif
				#endif

				return sWeight;
			},

			// CONTROL FUNCTION
			[](std::vector<Rotors*>& rotors, int pullOffset, int pushOffset) {;}
		);

		clones.outputOP.assign(2, 0.0);
		for (int i=0; i < nc; i++) {
			clones.outputOP[0] += (clones.finalSystem(i))->getTotalOrder()[0]
				/(clones.finalSystem(i))->getDump()[0]; // order parameter
			clones.outputOP[1] += (clones.finalSystem(i))->getTotalOrderSq()[0]
				/(clones.finalSystem(i))->getDump()[0]; // squared order parameter
		}

		for (unsigned int j=0;j<2;j++) { clones.outputOP[j] /= nc; }

		std::cout << std::endl;
		std::cout << "##s "    << sValue << std::endl
		          << "##bias " << BIAS << std::endl
		          << "#SCGF "  << clones.outputPsi/N << std::endl
		          << "#nu "    << clones.outputOP[0] << std::endl
		          << "#nu^2 "  << clones.outputOP[1] << std::endl << std::endl
		          << "##time " << clones.outputWalltime << std::endl;

		// output to file
		output.write<double>(clones.outputPsi);
		output.write<double>(clones.outputOP[0]);
		output.write<double>(clones.outputOP[1]);
		output.write<double>(clones.outputWalltime);
	}

}

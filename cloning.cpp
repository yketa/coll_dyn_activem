#include <iostream>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cloningserial.hpp"
#include "env.hpp"
#include "particle.hpp"
#include "readwrite.hpp"

template<> void CloningSerial<System>::
	loadState() {
	// Load cloning configurations from input file.

	// DUMP LENGTHS
	long int headerLength = 3*sizeof(int) + sizeof(double);
	long int headerRunLength = sizeof(int) + sizeof(double)
		+ (1 + 4)*sizeof(double);
	long int parametersRunCloneLength = sizeof(int) + 4*sizeof(double)
		+ sizeof(std::default_random_engine);
	long int particleRunCloneLength = 3*sizeof(double);
	long int dumpRunCloneLength = sizeof(int) + 8*sizeof(double);
	long int runLength =
		headerRunLength + (2*nc)*
			(parametersRunCloneLength + dumpRunCloneLength
				+ loadInput.read<int>(headerLength + headerRunLength) // number of particles
					*particleRunCloneLength);
	long int cloneLength =
		parametersRunCloneLength + dumpRunCloneLength
			+ loadInput.read<int>(headerLength + headerRunLength) // number of particles
				*particleRunCloneLength;
	long int nRuns = (loadInput.getFileSize() - headerLength)/runLength;

	if ( runIndex >= nRuns ) {
		throw std::invalid_argument("Not enough runs in input file.");
	}

	// FILE CHECKS
	if ( nc != loadInput.read<int>((long int) 0) ) {
		throw std::invalid_argument("Invalid number of clones.");
	}
	if ( cloneMethod != loadInput.read<int>() ) {
		throw std::invalid_argument("Invalid cloning method.");
	}
	if ( tau != loadInput.read<int>() ) {
		throw std::invalid_argument("Invalid cloning step size.");
	}
	if ( sValue != loadInput.read<double>() ) {
		throw std::invalid_argument("Invalid biasing parameter.");
	}
	if ( loadInput.getFileSize() != headerLength + nRuns*runLength ) {
		throw std::invalid_argument("Invalid file size.");
	}

	// CLONING STATE
	arrswitch = loadInput.read<int>(headerLength + runLength*runIndex);
	outputPsiOffset[0] = loadInput.read<double>();
	outputPsiOffset[1] = loadInput.read<double>();

	// CLONES
	if ( runIndex == 0 ) deleteClones();
  for (int i=0;i<2*nc;i++) {

		// CREATE OR CHECK SYSTEM
		if ( runIndex == 0 ) { // create system on first run
	    systems[i] = new System(
				loadInput.read<int>(        // N
					headerLength              // -- header
					+ runLength*runIndex      // -- previous runs
					+ headerRunLength         // -- cloning output header
					+ i*cloneLength),         // -- other clones
				loadInput.read<double>(),   // lp
				loadInput.read<double>(),   // phi
				loadInput.read<double>(),   // g
				loadInput.read<double>(),   // dt
				-1, tau, cloneFilename(i)); // create new system from copy of dummy, with random seed from processSeeds, computing active work and order parameter for every tau iterations
		}
		else { // check system parameters on following runs (and change torque parameter)
			if (
				loadInput.read<int>(                                                  // N
					headerLength                                                        // -- header
					+ runLength*runIndex                                                // -- previous runs
					+ headerRunLength                                                   // -- cloning output header
					+ i*cloneLength)                                                    // -- other clones
				!= systems[i]->getNumberParticles() ) {
				throw std::invalid_argument("Invalid number of particles.");
			}
			if ( loadInput.read<double>() != systems[i]->getPersistenceLength() ) { // lp
				throw std::invalid_argument("Invalid persistence length.");
			}
			if ( loadInput.read<double>() != systems[i]->getPackingFraction() ) {   // phi
				throw std::invalid_argument("Invalid packing fraction.");
			}
			double g = loadInput.read<double>(); systems[i]->setTorqueParameter(g); // g
			if ( loadInput.read<double>() != systems[i]->getTimeStep() ) {          // dt
				throw std::invalid_argument("Invalid time step.");
			}
		}

		// RANDOM GENERATOR
		systems[i]->setGenerator(loadInput.read<std::default_random_engine>());

		// COPY POSITIONS AND ORIENTATION
		for (int j=0; j < systems[i]->getNumberParticles(); j++) {
			for (int dim=0; dim < 2; dim++) {
				(systems[i]->getParticle(j))->position()[dim] =
					loadInput.read<double>();
			}
			(systems[i]->getParticle(j))->orientation()[0] = loadInput.read<double>();
		}
		if ( runIndex == 0 ) systems[i]->saveInitialState();

		// COPY DUMPS
		systems[i]->getDump()[0] = loadInput.read<int>();                    // total dumps
		systems[i]->getTotalWork()[0] = loadInput.read<double>();            // active work
		systems[i]->getTotalWorkForce()[0] = loadInput.read<double>();       // force part of the active work
		systems[i]->getTotalWorkOrientation()[0] = loadInput.read<double>(); // orientation part of the active work
		systems[i]->getTotalOrder()[0] = loadInput.read<double>();           // order parameter
		systems[i]->getTotalOrder0()[0] = loadInput.read<double>();          // order parameter along x-axis
		systems[i]->getTotalOrder1()[0] = loadInput.read<double>();          // order parameter along y-axis
		systems[i]->getTotalTorqueIntegral1()[0] = loadInput.read<double>(); // first torque integral
		systems[i]->getTotalTorqueIntegral2()[0] = loadInput.read<double>(); // second torque integral
  }

	outSyncRandomGenerator();
}

template<> void CloningSerial<System>::
	saveState() {
	// Save cloning configurations to output file.

	// CLONING ALGORITHM PARAMETERS
	saveOutput.write<int>(arrswitch);
	saveOutput.write<double>(iter*tau*systems[0]->getTimeStep());

	// CLONING OUTPUT
	std::vector<double> output(4, 0.0);
	for (int i=0; i < nc; i++) {
		output[0] += finalSystem(i)->getTotalWork()[0]
			/((finalSystem(i)->getTimeStep())*(finalSystem(i)->getDump()[0])); // normalised rate of active work
		output[1] += finalSystem(i)->getTotalWorkForce()[0]
			/((finalSystem(i)->getTimeStep())*(finalSystem(i)->getDump()[0])); // force part of the normalised rate of active work
		output[2] += finalSystem(i)->getTotalWorkOrientation()[0]
			/((finalSystem(i)->getTimeStep())*(finalSystem(i)->getDump()[0])); // orientation part of the normalised rate of active work
		output[3] += finalSystem(i)->getTotalOrder()[0]
			/(finalSystem(i)->getDump()[0]); // order parameter
	}
	saveOutput.write<double>(outputPsi);
	for (unsigned int j=0;j<4;j++) saveOutput.write<double>(output[j]/nc);

	// CLONES
	for (int i=0; i < 2*nc; i++) {
		// PHYSICAL PARAMETERS
		saveOutput.write<int>(systems[i]->getNumberParticles());
		saveOutput.write<double>(systems[i]->getPersistenceLength());
		saveOutput.write<double>(systems[i]->getPackingFraction());
		saveOutput.write<double>(systems[i]->getTorqueParameter());
		saveOutput.write<double>(systems[i]->getTimeStep());
		// RANDOM GENERATOR
		saveOutput.write<std::default_random_engine>
			((systems[i]->getRandomGenerator())->getGenerator());
		// POSITIONS AND ORIENTATIONS
		for (int j=0; j < systems[i]->getNumberParticles(); j++) {
			for (int dim=0; dim < 2; dim++) {
				saveOutput.write<double>
					((systems[i]->getParticle(j))->position()[dim]);
			}
			saveOutput.write<double>
				((systems[i]->getParticle(j))->orientation()[0]);
		}
		// DUMPS
		saveOutput.write<int>(systems[i]->getDump()[0]);
		saveOutput.write<double>(systems[i]->getTotalWork()[0]);
		saveOutput.write<double>(systems[i]->getTotalWorkForce()[0]);
		saveOutput.write<double>(systems[i]->getTotalWorkOrientation()[0]);
		saveOutput.write<double>(systems[i]->getTotalOrder()[0]);
		saveOutput.write<double>(systems[i]->getTotalOrder0()[0]);
		saveOutput.write<double>(systems[i]->getTotalOrder1()[0]);
		saveOutput.write<double>(systems[i]->getTotalTorqueIntegral1()[0]);
		saveOutput.write<double>(systems[i]->getTotalTorqueIntegral2()[0]);
	}
}

template<> void CloningSerial<System>::
	writeTrajFiles(Write& clonesLog) {
  // Write trajectory files for cloning loop.

  int period = 1;

  // BUILD TRAJECTORIES

  clonesLog.flush();
  Read log(clonesLog.getOutputFile());

  std::vector<std::vector<int>> trajectories (nc);
  std::vector<int> parents (nc);
  for (int i=0; i < nc; i++) { parents[i] = i; }

  for (int t=0; t < iter; t++) {
    for (int i=0; i < nc; i++) {
      int newParent = log.read<int>(
        -((t + 1)*nc - parents[i])*sizeof(int),
        std::ios_base::end);
      trajectories[i].insert(trajectories[i].begin(), newParent);
      parents[i] = newParent;
    }
  }
  log.close();

  // WRITE .dat FILES

  std::vector<Dat*> dat;
  std::vector<std::vector<double>> activeWork;
  std::vector<std::vector<double>> activeWorkForce;
  std::vector<std::vector<double>> activeWorkOri;
  std::vector<std::vector<double>> orderParameter;
	std::vector<std::vector<double>> orderParameter0;
	std::vector<std::vector<double>> orderParameter1;
  std::vector<std::vector<double>> torqueIntegral1;
  std::vector<std::vector<double>> torqueIntegral2;
  for (int i=0; i < 2*nc; i++) {
    systems[i]->flushOutputFile();
    dat.push_back(new Dat(systems[i]->getOutputFile(), true));
    dat[i]->close();
    activeWork.push_back(dat[i]->getActiveWork());
    activeWorkForce.push_back(dat[i]->getActiveWorkForce());
    activeWorkOri.push_back(dat[i]->getActiveWorkOri());
    orderParameter.push_back(dat[i]->getOrderParameter());
		orderParameter0.push_back(dat[i]->getOrderParameter0());
		orderParameter1.push_back(dat[i]->getOrderParameter1());
    torqueIntegral1.push_back(dat[i]->getTorqueIntegral1());
    torqueIntegral2.push_back(dat[i]->getTorqueIntegral2());
  }

  for (int i=0; i < nc; i++) {

    Write output(std::experimental::filesystem::path(
      std::experimental::filesystem::path(clonesDirectory) /
      [](int index, int rIndex)
        { return
          std::string(4 - std::to_string(rIndex).length(), '0')
            + std::to_string(rIndex) + std::string(".")
          + std::string(6 - std::to_string(index).length(), '0')
            + std::to_string(index) + std::string(".dat"); }
        (i, runIndex)
      ).u8string());

    // header
    output.write<int>(systems[i]->getNumberParticles());
    output.write<double>(systems[i]->getPersistenceLength());
    output.write<double>(systems[i]->getPackingFraction());
    output.write<double>(systems[i]->getSystemSize());
    output.write<double>(systems[i]->getTorqueParameter());
    output.write<int>(-1);
    output.write<double>(systems[i]->getTimeStep());
    output.write<int>(1);
    output.write<bool>(true);
    output.write<int>(tau);

    // initial frame
    int initFrame = (dat[i]->getFrames() - 1) - period*((int) (iter + 1)/2);
    int initClone = trajectories[i][0];
    dat[initClone]->open();
    for (int p=0; p < systems[i]->getNumberParticles(); p++) { // output all particles
      // POSITIONS
      for (int dim=0; dim < 2; dim++) { // output position in each dimension
        output.write<double>(dat[initClone]->getPosition(initFrame, p, dim));
      }
      // ORIENTATIONS
      output.write<double>(dat[initClone]->getOrientation(initFrame, p)); // output orientation
      // VELOCITIES
      for (int dim=0; dim < 2; dim++) { // output velocity in each dimension
        output.write<double>(dat[initClone]->getVelocity(initFrame, p, dim)); // output velocity
      }
    }
    dat[initClone]->close();

    // other frames
    for (int frame=1; frame <= iter; frame++) {

      int currClone = trajectories[i][frame - 1] + nc*(1 - (frame%2));
      int currFrame = (dat[currClone]->getFrames() - 1)
        - ((int) (iter - frame)/2)*period;

      dat[currClone]->open();
      for (int p=0; p < systems[i]->getNumberParticles(); p++) { // output all particles
        // POSITIONS
        for (int dim=0; dim < 2; dim++) { // output position in each dimension
          output.write<double>(
            dat[currClone]->getPosition(currFrame, p, dim));
        }
        // ORIENTATIONS
        output.write<double>(
          dat[currClone]->getOrientation(currFrame, p)); // output orientation
        // VELOCITIES
        for (int dim=0; dim < 2; dim++) { // output velocity in each dimension
          output.write<double>(
            dat[currClone]->getVelocity(currFrame, p, dim)); // output velocity
        }
      }
      dat[currClone]->close();

      // active work, polarisation, and torque integrals
      output.write<double>(activeWork[currClone][currFrame/period - 1]);
      output.write<double>(activeWorkForce[currClone][currFrame/period - 1]);
      output.write<double>(activeWorkOri[currClone][currFrame/period - 1]);
      output.write<double>(orderParameter[currClone][currFrame/period - 1]);
			output.write<double>(orderParameter0[currClone][currFrame/period - 1]);
			output.write<double>(orderParameter1[currClone][currFrame/period - 1]);
      output.write<double>(torqueIntegral1[currClone][currFrame/period - 1]);
      output.write<double>(torqueIntegral2[currClone][currFrame/period - 1]);
    }

    // close file
    output.close();
  }

  // delete pointers to input files
  for (int i=0; i < 2*nc; i++) delete dat[i];
}

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
	int N = getEnvInt("N", 100); // number of particles in the system
	double lp = getEnvDouble("LP", 5); // dimensionless persistence length
	double phi = getEnvDouble("PHI", 0.65); // packing fraction

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
	output.write<double>(lp);
	output.write<double>(phi);
	output.write<int>(tau);
	output.write<double>(dt);
	output.close();

	// save trajectories
	std::string clonesDirectory = getEnvString("CLONES_DIRECTORY", ""); // if different than "" then clones trajectories are saved to this directory

	// load initial cloning state
	std::string loadFile = getEnvString("LOAD_FILE", ""); // if != "": input class from where initial configurations are loaded
	// save final cloning state
	std::string saveFile = getEnvString("SAVE_FILE", ""); // if != "": write class to where final configurations at each call of doCloning are saved

	// parameters class
	Parameters parameters(N, lp, phi, dt);
	// dummy system
	System dummy(&parameters);

	printf("## CloningSerial Code: tmax %.3e numClones %d runs %d s %.3e tau %d Delta.t %.3e\n",tmax,nc,nRuns,sValue,tau,dt);
	#if BIAS_POLARISATION
	std::cout << "## Biasing with respect to polarisation." << std::endl;
	#if CONTROLLED_DYNAMICS
	std::cout << "## Modified rotational EOM." << std::endl;
	#endif
	#else
	std::cout << "## Biasing with respect to the active work." << std::endl;
	#if CONTROLLED_DYNAMICS == 1
	std::cout << "## Modified translational EOM." << std::endl;
	#endif
	#if CONTROLLED_DYNAMICS == 2
	std::cout << "## Modified translational and rotational EOM with 1st method." << std::endl;
	#endif
	#if CONTROLLED_DYNAMICS == 3
	std::cout << "## Modified translational and rotational EOM with 2nd method." << std::endl;
	#endif
	#endif
	// see cloning.py for more info

	// cloning object
	CloningSerial<System> clones(nc, tau, sValue, cloneMethod,
		clonesDirectory, loadFile, saveFile);

	// set up the clones
	if ( loadFile == "" ) { clones.init(&dummy, seed); }

	std::cout << "## master seed " << seed << std::endl;

  double sFactor = N*tau*dt;

	for (int run = 0; run<nRuns;run++) {

		#if (BIAS_POLARISATION && CONTROLLED_DYNAMICS)\
			|| (! BIAS_POLARISATION &&\
				(CONTROLLED_DYNAMICS == 2 || CONTROLLED_DYNAMICS == 3))
		#ifdef TORQUE_DUMP
		Write torqueDump(getEnvString("TORQUE_DUMP_FILE", "torque.dump"));
		torqueDump.close();
		#endif
		#endif

		// go! (this includes generating "random" [different] initial conditions for the clones)
		clones.doCloning(tmax, initSim,

			// ITERATION FUNCTION
			[](System* system, int Niter) { iterate_ABP_WCA(system, Niter); },

			// GET WEIGHT FUNCTION
			[&sFactor](System* system) {

			double sWeight;

			#if BIAS_POLARISATION

			sWeight = system->getBiasingParameter()*system->getOrder(); // s nu = s nu

			#ifdef CONTROLLED_DYNAMICS
			sWeight += system->getTorqueParameter()* // s nu += g
				(1.0/system->getNumberParticles()      // (1/N
				- system->getTorqueIntegral1()         // - I_1
				- system->getTorqueParameter()*        // - g
				system->getPersistenceLength()*        // lp
				system->getTorqueIntegral2());         // I_2)
			#endif

			#else

			#if CONTROLLED_DYNAMICS
			sWeight = system->getBiasingParameter()*( // sw = s(
				1.0 - system->getBiasingParameter()/    // 1 - s/
				(3.0*system->getPersistenceLength())    // (3*lp)
				+ system->getWorkForce());              // + w_f))
			#if CONTROLLED_DYNAMICS == 2 || CONTROLLED_DYNAMICS == 3
			sWeight += system->getTorqueParameter()*  // sw += g
				(1.0/system->getNumberParticles()       // (1/N
				- system->getTorqueIntegral1()          // - I_1
				- system->getTorqueParameter()*         // - g
				system->getPersistenceLength()*         // lp
				system->getTorqueIntegral2());          // I_2)
			#endif
			#else
			sWeight = system->getBiasingParameter()*system->getWork();
			#endif

			#endif

			return sFactor*sWeight;
			},

			// CONTROL FUNCTION
			[&nc
			#if (BIAS_POLARISATION && CONTROLLED_DYNAMICS)\
				|| (! BIAS_POLARISATION &&\
					(CONTROLLED_DYNAMICS == 2 || CONTROLLED_DYNAMICS == 3))
			#ifdef TORQUE_DUMP
			, &torqueDump
			#endif
			#endif
			](std::vector<System*>& systems, int pullOffset, int pushOffset) {

				// TORQUE PARAMETER VALUE

				// ORDER PARAMETER METHOD
				#if (BIAS_POLARISATION && CONTROLLED_DYNAMICS)\
					|| (! BIAS_POLARISATION && CONTROLLED_DYNAMICS == 2)

				// order parameter squared
				double nusq = 0;
				#ifdef _OPENMP
				#pragma omp parallel for reduction (+:nusq)
				#endif
				for (int i=0; i<nc; i++) {
					nusq += systems[pushOffset + i]->getTotalTorqueIntegral1()[0]
						/systems[pushOffset + i]->getDump()[0];
				}
				nusq /= nc;

				// define and set g
				double g = (1.0/(systems[0]->getNumberParticles()*nusq) - 1.0)
					/systems[0]->getPersistenceLength();
				#ifdef _OPENMP
				#pragma omp parallel for
				#endif
				for (int i=0; i<nc; i++) {
					systems[pullOffset + i]->setTorqueParameter(g);
				}

				// output
				#ifdef TORQUE_DUMP
				torqueDump.open();
				torqueDump.write<double>(systems[pullOffset]->getTorqueParameter());
				torqueDump.close();
				#endif

				#endif

				// POLYNOMIAL METHOD
				#if ! BIAS_POLARISATION && CONTROLLED_DYNAMICS == 3

				// polynomial coefficients
				double torqueIntegral1 (0.0), torqueIntegral2 (0.0); // torque integrals
				double workForce (0.0); // force part of the normalised rate of active work
				#ifdef _OPENMP
				#pragma omp parallel for reduction (+:torqueIntegral1,torqueIntegral2,workForce)
				#endif
				for (int i=0; i<nc; i++) {
					torqueIntegral1 +=
						systems[pushOffset + i]->getTotalTorqueIntegral1()[0]
							/systems[pullOffset + i]->getDump()[0];
					torqueIntegral2 +=
						systems[pushOffset + i]->getTotalTorqueIntegral2()[0]
							/systems[pullOffset + i]->getDump()[0];
					workForce +=
						systems[pushOffset + i]->getTotalWorkForce()[0]
							/(systems[pushOffset + i]->getTimeStep()
								*systems[pushOffset + i]->getDump()[0]);
				}
				torqueIntegral1 /= nc;
				torqueIntegral2 /= nc;
				workForce /= nc;

				// define and set g
				double g = -(torqueIntegral1 - 1.0/systems[0]->getNumberParticles())
					/(2*systems[0]->getPersistenceLength()*torqueIntegral2);
				#ifdef _OPENMP
				#pragma omp parallel for
				#endif
				for (int i=0; i<nc; i++) {
					systems[pullOffset + i]->setTorqueParameter(g);
				}

				// output
				#ifdef TORQUE_DUMP
				torqueDump.open();
				torqueDump.write<double>(systems[pullOffset]->getTorqueParameter());
				torqueDump.close();
				#endif

				#endif
			}
		);

		clones.outputOP.assign(4, 0.0);
		for (int i=0; i < nc; i++) {
			clones.outputOP[0] += (clones.finalSystem(i))->getTotalWork()[0]
				/((clones.finalSystem(i))->getTimeStep()
					*(clones.finalSystem(i))->getDump()[0]); // normalised rate of active work
			clones.outputOP[1] += (clones.finalSystem(i))->getTotalWorkForce()[0]
				/((clones.finalSystem(i))->getTimeStep()
					*(clones.finalSystem(i))->getDump()[0]); // force part of the normalised rate of active work
			clones.outputOP[2] += (clones.finalSystem(i))->getTotalWorkOrientation()[0]
				/((clones.finalSystem(i))->getTimeStep()
					*(clones.finalSystem(i))->getDump()[0]); // orientation part of the normalised rate of active work
			clones.outputOP[3] += (clones.finalSystem(i))->getTotalOrder()[0]
				/(clones.finalSystem(i))->getDump()[0]; // order parameter
		}

		for (unsigned int j=0;j<4;j++) { clones.outputOP[j] /= nc; }

		std::cout << std::endl;
		std::cout << "##s "    << sValue << std::endl
		          << "#SCGF "  << clones.outputPsi/N << std::endl
		          << "#w "     << clones.outputOP[0] << std::endl
		          << "#wf "    << clones.outputOP[1] << std::endl
							<< "#wo "		 << clones.outputOP[2] << std::endl
						  << "#nu "    << clones.outputOP[3] << std::endl << std::endl
		          << "##time " << clones.outputWalltime << std::endl;

		// output to file
		output.open();
		output.write<double>(clones.outputPsi);
		output.write<double>(clones.outputOP[0]);
		output.write<double>(clones.outputOP[1]);
		output.write<double>(clones.outputOP[2]);
		output.write<double>(clones.outputOP[3]);
		output.write<double>(clones.outputWalltime);
		output.close();
	}

}

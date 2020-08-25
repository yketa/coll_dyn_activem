#! /bin/bash

# This bash script enables one to launch cloning simulations in a more efficient
# and automatised way.

# Execute with `source exponents.sh` or `. exponents.sh` in order to correctly
# export functions to the current shell.

function chainCloning() {
	HELP='
	Launches chained cloning simulations — saving and loading configuration files.
	```
	chainCloning [cloning script] [directory] [# of launches] [env variables]
	```
	'
	if [[ "$1" == "help" || "$1" == "" ]];
		then echo "$HELP"; return;
	fi

	unset CHAINCLONING;

	SCRIPT=$1;
	DIRECTORY=$2;
	LAUNCHES=$3;
	export ${@:4};	# export environment variables

	function confName() {
		# Name of configuration file given the launch index.
		echo ${DIRECTORY}/$(printf '%03i' $1)'.{:03d}.clo.conf';
	}

	for launch in $(seq 0 $(($LAUNCHES - 1))); do

		OUT_DIR=$DIRECTORY LAUNCH=$launch SLURM=True SLURM_CHAIN=$CHAINCLONING \
		LOAD_FILE=$(if (( $launch > 0 )); then confName $(($launch - 1)); fi) \
		SAVE_FILE=$(confName $launch) \
		setsid python $SCRIPT &> ${DIRECTORY}/out;

		while [[ $(cat ${DIRECTORY}/out) == "" ]]; do
			continue;
		done
		export CHAINCLONING=$(head -n1 ${DIRECTORY}/out);

	done

}
export -f chainCloning

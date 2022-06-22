#! /bin/bash

# Submits job to a Slurm job scheduler.
# (see https://slurm.schedmd.com/sbatch.html)

# SCRIPT DIRECTORY

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# DEFAULT PARAMETERS
# We are concerned here with the `fawcett' computer of the Faculty of
# Mathematics at the University of Cambridge.
# (see https://www.maths.cam.ac.uk/computing/faculty-hpc-system-fawcett)

_SIM_DIR=${SCRIPT_DIR}/build                  # default simulation directory
_ERROR_DIR=${_SIM_DIR}/queueing_system_error  # default error output directory
_OUT_FILE=/dev/null                           # standard output file

_PARTITION=skylake  # default partition for the ressource allocation
_GRES=              # default generic consumable ressources
_NODES=1            # default required number of nodes
_NTASKS=1           # default number of MPI ranks running per node
_ARRAY_SIZE=        # default job array size
_ARRAY_TASKS=       # default maximum number of simultaneous tasks in the array
_TIME=              # default required time
_MEMORY=            # default real memory required per node

# HELP MENU

usage() {

less <<< "Submit job to a Slurm job scheduler.
(see https://slurm.schedmd.com/sbatch.html)

SYNOPSIS

  [bash] slurm.sh [OPTIONS] [ENVIRONMENT VARIABLES] [SCRIPT]

OPTIONS

  -h    Display this help.

  -w    Pause this script until completion of the job.

  -j    Job name on Slurm scheduler.
        DEFAULT: script name after last '/'
  -c    Execute after job with this ID has succesfully executed.
        DEFAULT: (not specified)

  -d    Directory in which to execute the job.
        DEFAULT: _SIM_DIR
  -o    Error output directory.
        NOTE: Error files are named according to job ID.
        DEFAULT: _ERROR_DIR
  -f    Standard output file.
        DEFAULT: _OUT_FILE

  -p    Partition for the resource allocation.
        DEFAULT: _PARTITION
  -g    Generic consumable resources.
        DEFAULT: _GRES
  -n    Required number of nodes.
        DEFAULT: _NODES
  -r    Number of MPI ranks running per node.
        Number of threads for OpenMP parallelised jobs.
        DEFAULT: _NTASKS
  -a    Job array size.
        (see https://slurm.schedmd.com/job_array.html)
        NOTE: SLURM_ARRAY_TASK_ID is set as task id (between 0 and size - 1).
        DEFAULT: _ARRAY_SIZE
  -s    Maximum number of simultaneously running tasks in the job array.
        (see https://slurm.schedmd.com/job_array.html)
        NOTE: An empty string will not set this maximum.
        DEFAULT: _ARRAY_TASKS
  -t    Required time.
        DEFAULT: _TIME
  -m    Real memory required per node.
        NOTE: MaxMemPerNode allocates maximum memory.
        DEFAULT: _MEMORY
"
}

# OPTIONS

while getopts "hwj:c:d:o:f:p:g:n:r:a:s:t:m:" OPTION; do
  case $OPTION in

    h)  # help menu
      usage; exit 0;;

    w)  # wait
      WAIT=true;;

    j)  # job name
      JOB_NAME=$OPTARG;;
    c)  # chained job
      CHAIN=$OPTARG;;

    d)  # simulation directory
      SIM_DIR=$OPTARG;;
    o)  # error output directory
      ERROR_DIR=$OPTARG;;
    f)  # standard output file
      OUT_FILE=$OPTARG;;

    p)  # partition
      PARTITION=$OPTARG;;
    g)  # generic consumable resources
      GRES=$OPTARG;;
    n)  # nodes
      NODES=$OPTARG;;
    r)  # tasks
      NTASKS=$OPTARG;;
    a)  # array size
      ARRAY_SIZE=$OPTARG;;
    s)  # array tasks
      ARRAY_TASKS=$OPTARG;;
    t)  # time
      TIME=$OPTARG;;
    m)  # real memory
      MEMORY=$OPTARG;;

  esac
done
shift $(expr $OPTIND - 1);

if [[ -z "$@" ]]; then
  echo 'No script submitted.';
  usage;
  exit 1;
fi

SCRIPT=$@ # script to execute

# JOB PARAMETERS

JOB_NAME=${JOB_NAME-${SCRIPT##*/}}  # job name

SIM_DIR=${SIM_DIR-$_SIM_DIR}; mkdir -p "$SIM_DIR";          # simulation directory
ERROR_DIR=${ERROR_DIR-$_ERROR_DIR}; mkdir -p "$ERROR_DIR";  # error output directory
OUT_FILE=${OUT_FILE-$_OUT_FILE}                             # standard output file

PARTITION=${PARTITION-$_PARTITION}        # partition for the resource allocation
GRES=${GRES-$_GRES}                       # generic consumable resources
NODES=${NODES-$_NODES}                    # required number of nodes
NTASKS=${NTASKS-$_NTASKS}                 # maximum ntasks to be invoked on each core
ARRAY_SIZE=${ARRAY_SIZE-$_ARRAY_SIZE}     # job array size
ARRAY_TASKS=${ARRAY_TASKS-$_ARRAY_TASKS}  # maximum number of simultaneous tasks in the array
TIME=${TIME-$_TIME}                       # required time
MEMORY=${MEMORY-$_MEMORY}                 # real memory required per node

# SUBMIT JOB

sbatch ${WAIT:+-W} ${CHAIN:+-d afterok:$CHAIN} <<EOF
#! /bin/bash
#SBATCH --job-name='$JOB_NAME'
#SBATCH -D '$SIM_DIR'
#SBATCH --error='${ERROR_DIR}/%j.out'
#SBATCH --output='$OUT_FILE'
#SBATCH --partition=$PARTITION
${GRES:+#SBATCH --gres=$GRES}
#SBATCH --nodes=$NODES
#SBATCH --ntasks=$NTASKS
${ARRAY_SIZE:+#SBATCH --array=0-$(($ARRAY_SIZE-1))${ARRAY_TASKS+%$ARRAY_TASKS}}
${TIME:+#SBATCH --time=$TIME}
${MEMORY:+#SBATCH --mem=$MEMORY}

export OMP_NUM_THREADS=$NTASKS

# PRINT JOB PARAMETERS TO ERROR OUTPUT FILE
(>&2 printf '%-21s: %s\n' 'SUBMIT DIRECTORY' '$(pwd)')
(>&2 printf '%-21s: %s\n' 'DATE' '$(date)')
(>&2 echo)
(>&2 printf '%-21s: %s\n' 'JOB NAME' '$JOB_NAME')
(>&2 echo)
(>&2 printf '%-21s: %s\n' 'SIMULATION DIRECTORY' '$SIM_DIR')
(>&2 printf '%-21s: %s\n' 'OUTPUT FILE' '$OUT_FILE')
(>&2 echo)
(>&2 printf '%-21s: %s\n' 'PARTITION' '$PARTITION')
(>&2 printf '%-21s: %s\n' 'GRES' '$GRES')
(>&2 printf '%-21s: %s\n' 'NODES REQUIRED' '$NODES')
(>&2 printf '%-21s: %s\n' 'TASKS PER NODE' '$NTASKS')
(>&2 printf '%-21s: %s\n' 'ARRAY SIZE' '$ARRAY_SIZE')
(>&2 printf '%-21s: %s\n' 'TASKS IN ARRAY' '$ARRAY_TASKS')
(>&2 printf '%-21s: %s\n' 'TIME REQUIRED' '$TIME')
(>&2 printf '%-21s: %s\n' 'MEMORY REQUIRED' '$MEMORY')
(>&2 echo)
(>&2 printf '%-21s: %s\n' 'SCRIPT' '$SCRIPT')
(>&2 echo)

trap 'kill -15 "\${PID}"; wait "\${PID}";' SIGINT SIGTERM # terminate script when cancelled

module load singularity/3.6.3

$SCRIPT & # launching script
PID="\$!"
wait "\${PID}"
EOF

${WAIT:+wait} # wait until completion of the job

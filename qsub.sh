#! /bin/bash

# Submits job to a Sun Grid Engine queuing system.
# (see http://gridscheduler.sourceforge.net/htmlman/htmlman1/qsub.html)

# SCRIPT DIRECTORY

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# DEFAULT PARAMETERS
# We are concerned here with the `squid' computer of the Laboratoire Charles
# Coulomb at the Université de Montpellier.

_SIM_DIR=${SCRIPT_DIR}/build                  # default simulation directory
_ERROR_DIR=${_SIM_DIR}/queueing_system_error  # default error output directory
_OUT_FILE=/dev/null                           # standard output file

_QUEUE=simons.q   # default cluster queue(s)
_SHELL=/bin/bash  # default interpreting shell for the job

# HELP MENU

usage() {

less <<< "Submit job to a Sun Grid Engine queuing system.
(see http://gridscheduler.sourceforge.net/htmlman/htmlman1/qsub.html)

SYNOPSIS

  [bash] qsub.sh [OPTIONS] [ENVIRONMENT VARIABLES] [SCRIPT]

OPTIONS

  -h    Display this help.

  -w    Pause this script until completion of the job.

  -j    Job name on Sun Grid Engine queuing system.
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

  -p    Queue(s) to exectue this job. (see \`free_cpu_load\`)
        DEFAULT: _QUEUE
  -S    Interpreting shell for the job.
        DEFAULT: _SHELL
  -r    Number of tasks.
        Number of threads for OpenMP parallelised jobs.
        DEFAULT:
  -a    Job array size.
        NOTE: SGE_TASK_ID is set as task id (between 0 and size - 1).
        DEFAULT:
  -s    Maximum number of simultaneously running tasks in the job array.
        NOTE: An empty string will not set this maximum.
        DEFAULT:
"
}

# OPTIONS

while getopts "hwj:c:d:o:f:p:S:r:a:s:" OPTION; do
  case $OPTION in

    h)  # help menu
      usage; exit 0;;

    w)  # wait
      SYNC=yes;;

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

    p)  # queue
      QUEUE=$OPTARG;;
    S)  # shell
      SHELL=$OPTARG;;
    r)  # parallel environment
      PARALLEL_ENV=$OPTARG;;
    a)  # array size
      ARRAY_SIZE=$OPTARG;;
    s)  # array tasks
      ARRAY_TASKS=$OPTARG;;

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

QUEUE=${QUEUE-$_QUEUE}  # cluster queue(s)
SHELL=${SHELL-$_SHELL}  # interpreting shell for the job

# SUBMIT JOB

qsub <<EOF
#! /bin/bash
#$ -N '$JOB_NAME'
#$ -wd '$SIM_DIR'
#$ -e '${ERROR_DIR}/\$JOB_ID.out'
#$ -o '$OUT_FILE'
#$ -q $QUEUE
#$ -S $SHELL
${CHAIN:+#$ -hold_jid $CHAIN}
${PARALLEL_ENV:+#$ -pe $PARALLEL_ENV}
${ARRAY_SIZE:+#$ -t 0-$(($ARRAY_SIZE-1)):1}
${ARRAY_SIZE:+${ARRAY_TASKS:+#$ -tc $ARRAY_TASKS}}
${SYNC:+#$ -sync $SYNC}

${PARALLEL_ENV:+export OMP_NUM_THREADS=$PARALLEL_ENV}

# PRINT JOB PARAMETERS TO ERROR OUTPUT FILE
(>&2 printf '%-21s: %s\n' 'SUBMIT DIRECTORY' '$(pwd)')
(>&2 printf '%-21s: %s\n' 'DATE' '$(date)')
(>&2 echo)
(>&2 printf '%-21s: %s\n' 'JOB NAME' '$JOB_NAME')
(>&2 echo)
(>&2 printf '%-21s: %s\n' 'SIMULATION DIRECTORY' '$SIM_DIR')
(>&2 printf '%-21s: %s\n' 'OUTPUT FILE' '$OUT_FILE')
(>&2 echo)
(>&2 printf '%-21s: %s\n' 'QUEUE' '$QUEUE')
(>&2 printf '%-21s: %s\n' 'SHELL' '$SHELL')
(>&2 printf '%-21s: %s\n' 'NUMBER OF TASKS' '${PARALLEL_ENV:+$PARALLEL_ENV}')
(>&2 printf '%-21s: %s\n' 'ARRAY SIZE' '${ARRAY_SIZE:+$ARRAY_SIZE}')
(>&2 echo)
(>&2 printf '%-21s: %s\n' 'SCRIPT' '$SCRIPT')
(>&2 echo)

trap 'kill -15 "\${PID}"; wait "\${PID}";' SIGINT SIGTERM # terminate script when cancelled

$SCRIPT & # launching script
PID="\$!"
wait "\${PID}"
EOF

${SYNC:+wait} # wait until completion of the job

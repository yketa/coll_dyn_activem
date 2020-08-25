#! /bin/bash

# This bash shell script enables one to use functions
# coll_dyn_activem.exponents.float_to_letters and
# coll_dyn_activem.exponents.letters_to_float from the shell.

# Execute with `source exponents.sh` or `. exponents.sh` in order to correctly
# export functions to the current shell.

# Information about litteral notation of floats used in this project can be
# found in coll_dyn_activem/exponents.py.

PYTHON=${PYTHON-python} # Python executable
# NOTE: Using a weird syntax for function declarations in order to enforce this
#       choice of executable every time this script is executed.

eval "$(cat <<EOF
letters_to_float() {
  # Converts litteral expression to float expression.
  $PYTHON -c "from coll_dyn_activem.exponents import letters_to_float; print(letters_to_float('\$1'))"
}
EOF
)"
export -f letters_to_float

eval "$(cat <<EOF
float_to_letters() {
  # Converts float expression to litteral expression.
  $PYTHON -c "from coll_dyn_activem.exponents import float_to_letters; print(float_to_letters(\$1))"
}
EOF
)"
export -f float_to_letters

#!/bin/bash
set -e

DIR=$(dirname ${BASH_SOURCE[0]})

VENV_NAME="aas-overcooked-env"
VENV_PATH="$DIR/$VENV_NAME"
PYTHON_VERSION_NUMBER="3.10"
PYTHON_VERSION="python$PYTHON_VERSION_NUMBER"
REQUIREMENTS_FILE_PATH="$DIR/requirements.txt"

function ansi() {
	echo -ne "\x1b[1;$1m"
}

RED=$(ansi 31)
GREEN=$(ansi 32)
YELLOW=$(ansi 33)
BLUE=$(ansi 34)
MAGENTA=$(ansi 35)
CYAN=$(ansi 36)
RESET=$(ansi 0)

if which conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda env list | grep -qE "^$VENV_NAME\s"; then
        conda create -y -n $VENV_NAME python=$PYTHON_VERSION_NUMBER
        echo "Created conda environment"
    else
        echo "Environment ${MAGENTA}$VENV_NAME${RESET} already exists"
    fi
    echo -n "Run ${YELLOW}'conda activate $VENV_NAME'${RESET}, "
    echo "then ${YELLOW}pip install -r $REQUIREMENTS_FILE_PATH${RESET} to use the environment."
    exit
fi

if which "$PYTHON_VERSION" >/dev/null 2>&1; then
    echo "Found required Python version (${GREEN}$PYTHON_VERSION${RESET})"
else
    echo "${RED}Expected $PYTHON_VERSION to be installed but it is not present.${RESET}" >&2
    echo "${RED}This script (and by extension Overcooked) requires ${PYTHON_VERSION}.${RESET}" >&2
    exit 1
fi

if ! [ -d $VENV_PATH ]; then
    $PYTHON_VERSION -m venv $VENV_PATH
fi
echo "Created venv environment"
echo -n "Run ${YELLOW}'source $VENV_PATH/bin/activate'${RESET}, "
echo "then ${YELLOW}$PYTHON_VERSION -m pip install -r $REQUIREMENTS_FILE_PATH${RESET} to use the environment"

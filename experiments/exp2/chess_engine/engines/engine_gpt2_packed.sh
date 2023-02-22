#!/bin/bash
if ! [ -z "$VENV_PATH" ]; then
    . $VENV_PATH/bin/activate
fi
export BASE_PATH=$(readlink -f $(dirname $0)/..)
export PYTHONPATH=$BASE_PATH

python $PYTHONPATH/engines/engine_gpt2_packed.py

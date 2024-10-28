#!/bin/bash

read -p "Which subversion of python are you using? (10/11): " answer
if [[ $answer == 10* ]]; then
    python3.10 -m venv venv
else
    python3.11 -m venv venv
fi
source venv/bin/activate

read -p "Are you using an Nvidia GPU? (Y/N): " answer
if [[ $answer == [Yy]* ]]; then
    pip install -r requirements.txt --extra-index-url=https://download.pytorch.org/whl/cu121
    echo "gpu=true" > config.txt
else
    pip install -r requirements.txt
    echo "gpu-false" > config.txt
fi

if [ -z "$1" ]; then
    read -p "Press Enter to continue..."
fi
deactivate

#!/bin/bash

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Installation not found. Installing..."
    chmod +x install.sh
    ./install.sh no_pause
    read -p "Press enter to continue..."
    deactivate
    exit 0
fi

while IFS== read -r key value; do
    if [ "$value" == "true" ]; then
        pip install -U -r requirements.txt --extra-index-url=https://download.pytorch.org/whl/cu121
    else
        pip install -U -r requirements.txt
    fi
done < config.txt

read -p "Press enter to continue..."
deactivate
exit 0

#!/bin/bash

if [ -d "venv" ]; then
    source venv/bin/activate
else
    chmod +x install.sh
    ./install.sh no_pause
    source venv/bin/activate
fi
python run.py --batch_size=1 --gen_threshold=0.35 --char_threshold=0.75 --model=vit-large --subfolder=False --noUnderscores=True --sortAlphabetically=False
read -p "Press any key to continue..."
deactivate

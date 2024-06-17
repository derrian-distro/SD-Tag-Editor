#!/bin/bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
deactivate

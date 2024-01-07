#!/bin/bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install onnxruntime # necessary due to cuda incompatibility with onnxruntime-gpu with cuda 12.1 on Linux, causes speed loss but alternative is not working at all
deactivate

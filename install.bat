@echo off
python -m venv venv
call .\venv\Scripts\activate
python -s -m pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
pause
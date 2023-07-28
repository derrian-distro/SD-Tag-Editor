@echo off
python -m venv venv
call .\venv\Scripts\activate
python -s -m pip install --upgrade --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu121 -r requirements.txt pygit2
pause
@echo off

python -m venv venv
call venv/Scripts/activate

choice /c YN /m "Are you using an Nvidia GPU?"
if errorlevel ==2 goto other_install

pip install -r requirements.txt --extra-index-url=https://download.pytorch.org/whl/cu126
echo gpu=true>config.txt
goto end

:other_install
pip install -r requirements.txt
echo gpu=false>config.txt

:end
if "%~1" == "" (
    pause
)

@echo off

if exist venv (
    call venv\Scripts\activate
) else (
    echo Installation not found. Installing...
    install.bat no_pause
    goto end
)
set "firstLine="
for /F "tokens=2 delims==" %%x in (config.txt) do (
    if "%%x"=="true" (
        goto gpu
    ) else (
        goto no_gpu
    )
)

:gpu
pip install -U -r requirements.txt --extra-index-url=https://download.pytorch.org/whl/cu121
goto end

:no_gpu
pip install -U -r requirements.txt

:end
pause

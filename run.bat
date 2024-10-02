@echo off

if exist venv (
	call venv\Scripts\activate
) else (
	call install.bat no_pause
	call venv\Scripts\activate
)

python run.py --batch_size=1 --gen_threshold=0.35 --char_threshold=0.75 --model=vit-large --subfolder=False --noUnderscores=True --sortAlphabetically=False
pause
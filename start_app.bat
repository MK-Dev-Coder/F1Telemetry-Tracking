@echo off
rem Starts the F1 Strategy Optimizer. Keep this window open while you use the app (and while you play F1 25).
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo Starting the app... the browser opens in a few seconds.
start "" cmd /c "timeout /t 8 >nul & start http://127.0.0.1:8000"
python api.py
pause

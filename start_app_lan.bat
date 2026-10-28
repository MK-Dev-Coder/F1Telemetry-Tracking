@echo off
rem Starts the F1 Strategy Optimizer so phones and tablets on your home network can open it too (the dash for a
rem second screen). Use it at home only: anyone on the same network can open the app while it runs.
rem The first time, Windows asks whether Python may use the network: allow it on private networks.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo Starting the app for this PC and your home network... the browser opens in a few seconds.
start "" cmd /c "timeout /t 8 >nul & start http://127.0.0.1:8000"
python api.py --lan
pause

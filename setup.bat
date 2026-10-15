@echo off
rem One-time setup: create .venv and install packages
cd /d %~dp0
set PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PY%" set PY=python
"%PY%" -m venv .venv || goto :err
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :err
.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu || goto :err
echo.
echo Setup complete. Run run_daily.bat (collect + dashboard) or run_lab.bat (analytics app).
pause
exit /b 0
:err
echo Setup failed - check the error above.
pause
exit /b 1

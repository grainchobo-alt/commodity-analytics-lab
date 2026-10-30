@echo off
rem Double-click to open the dashboard. If it is already running, just opens the browser.
cd /d %~dp0
netstat -ano | findstr ":8501 " | findstr LISTENING >nul
if %errorlevel%==0 goto open
start "Commodity Dashboard - closing this window stops the dashboard" /min .venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
for /l %%i in (1,1,60) do (
  timeout /t 1 /nobreak >nul
  netstat -ano | findstr ":8501 " | findstr LISTENING >nul && goto open
)
:open
start "" http://localhost:8501

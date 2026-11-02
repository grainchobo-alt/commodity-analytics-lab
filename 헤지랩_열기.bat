@echo off
rem Hedge lab draft (futures hedge ratio, VaR, hedge instruments, crush), port 8507. Separate from the main dashboard.
cd /d %~dp0
netstat -ano | findstr ":8507 " | findstr LISTENING >nul
if %errorlevel%==0 goto open
start "Hedge Lab - closing this window stops it" /min .venv\Scripts\python.exe -m streamlit run hedge_lab.py --server.port 8507 --server.headless true --browser.gatherUsageStats false --theme.base light --theme.primaryColor "#1E3A2F"
for /l %%i in (1,1,60) do (
  timeout /t 1 /nobreak >nul
  netstat -ano | findstr ":8507 " | findstr LISTENING >nul && goto open
)
:open
start "" http://localhost:8507

@echo off
rem Glass theme dashboard (port 8502). Same data and features as the default dashboard.
cd /d %~dp0
netstat -ano | findstr ":8502 " | findstr LISTENING >nul
if %errorlevel%==0 goto open
start "Commodity Dashboard GLASS - closing this window stops it" /min .venv\Scripts\python.exe -m streamlit run app_glass.py --server.port 8502 --server.headless true --browser.gatherUsageStats false --theme.base light --theme.primaryColor "#1E3A2F"
for /l %%i in (1,1,60) do (
  timeout /t 1 /nobreak >nul
  netstat -ano | findstr ":8502 " | findstr LISTENING >nul && goto open
)
:open
start "" http://localhost:8502

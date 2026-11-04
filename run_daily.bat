@echo off
rem Daily: incremental collect -> clean -> roll adjust -> quality report -> Excel export -> dashboard
rem Full re-download: collect.py --full
cd /d %~dp0
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe collect.py || goto :err
.venv\Scripts\python.exe clean.py || goto :err
.venv\Scripts\python.exe roll_adjust.py || goto :err
.venv\Scripts\python.exe data_quality.py > data\data_quality_console.txt || goto :err
.venv\Scripts\python.exe export_excel.py || goto :err
.venv\Scripts\python.exe build_dashboard.py || goto :err
rem Loss-limit alert (risk_alert.py): writes data\risk_alert_latest.md + appends data\risk_alerts.csv. Takes a few minutes (HMM/DCC); a failure here does not stop the batch
.venv\Scripts\python.exe risk_alert.py > data\risk_alert_console.txt 2>&1 || echo WARNING - risk_alert.py failed, see data\risk_alert_console.txt
start "" dashboard.html
exit /b 0
:err
echo FAILED - see messages above.
pause
exit /b 1

@echo off
title ISRO AI Mission Control - Exhibition
cd /d "%~dp0"
echo ================================================
echo      ISRO AI MISSION CONTROL - EXHIBITION
echo ================================================
echo.
echo Installing / checking required Python packages...
py -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo ERROR: Package installation failed.
  echo Check that Python 3.10+ is installed and try again.
  pause
  exit /b 1
)
echo.
echo Starting dashboard...
py -m streamlit run app.py
pause

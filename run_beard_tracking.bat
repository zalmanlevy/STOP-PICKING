@echo off
cd /d "%~dp0"

REM Check for common venv names
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

python beard_tracking.py
pause

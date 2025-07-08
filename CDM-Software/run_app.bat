@echo off
echo Starting Flask Application...
echo.

REM Try different Python commands
echo Trying python...
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found python, starting application...
    python web_application_demo.py
    goto :end
)

echo Trying py...
py --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found py, starting application...
    py web_application_demo.py
    goto :end
)

echo Trying python3...
python3 --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found python3, starting application...
    python3 web_application_demo.py
    goto :end
)

echo ERROR: Python not found in PATH!
echo Please install Python or add it to your PATH variable.
echo.
echo You can download Python from: https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation.
echo.
pause

:end
echo.
echo Application finished.
pause 
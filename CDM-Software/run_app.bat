@echo off
echo ========================================
echo ConMED-RL Flask Application Launcher
echo ========================================
echo.

echo NOTE: This script is for local development and testing.
echo For production deployment, please use Docker (see Docker-Deployment/).
echo.

REM Check if virtual environment should be activated
if exist "venv\Scripts\activate.bat" (
    echo Found virtual environment, activating...
    call venv\Scripts\activate.bat
    echo.
)

REM Try different Python commands
echo Checking Python installation...
echo.

echo Trying python...
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found python, checking dependencies...
    python -c "import flask, torch, numpy, sklearn, PIL" >nul 2>&1
    if %errorlevel% == 0 (
        echo Dependencies OK, starting application...
        echo.
        echo Starting Flask app on http://localhost:5000
        echo Press Ctrl+C to stop the application
        echo.
        python web_application_demo.py
        goto :end
    ) else (
        echo ERROR: Missing required dependencies!
        echo Please install dependencies: pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
)

echo Trying py...
py --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found py, checking dependencies...
    py -c "import flask, torch, numpy, sklearn, PIL" >nul 2>&1
    if %errorlevel% == 0 (
        echo Dependencies OK, starting application...
        echo.
        echo Starting Flask app on http://localhost:5000
        echo Press Ctrl+C to stop the application
        echo.
        py web_application_demo.py
        goto :end
    ) else (
        echo ERROR: Missing required dependencies!
        echo Please install dependencies: py -m pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
)

echo Trying python3...
python3 --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found python3, checking dependencies...
    python3 -c "import flask, torch, numpy, sklearn, PIL" >nul 2>&1
    if %errorlevel% == 0 (
        echo Dependencies OK, starting application...
        echo.
        echo Starting Flask app on http://localhost:5000
        echo Press Ctrl+C to stop the application
        echo.
        python3 web_application_demo.py
        goto :end
    ) else (
        echo ERROR: Missing required dependencies!
        echo Please install dependencies: python3 -m pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
)

echo ERROR: Python not found in PATH!
echo.
echo Please install Python or add it to your PATH variable.
echo You can download Python from: https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation.
echo.
echo Alternative: Use Docker for easier deployment
echo See Docker-Deployment/README.md for instructions.
echo.
pause
exit /b 1

:end
echo.
echo Application finished.
echo.
echo TIP: For production deployment, consider using Docker:
echo   cd ../Docker-Deployment
echo   docker-compose -f docker-compose.prod.yml up -d
echo.
pause 
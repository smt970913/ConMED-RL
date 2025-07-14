@echo off
setlocal enabledelayedexpansion

echo ========================================
echo ConMED-RL Research Environment Setup
echo ========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed or not in PATH!
    echo Please install Docker Desktop and try again.
    pause
    exit /b 1
)

REM Check if docker-compose is available
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker Compose is not available!
    echo Please install Docker Compose and try again.
    pause
    exit /b 1
)

REM Navigate to Docker-Deployment directory
cd /d "%~dp0\.."

echo Current directory: %CD%
echo.

REM Clean up old containers and images
echo [INFO] Cleaning up old containers and images...
docker-compose -f docker-compose.research.yml down 2>nul
docker system prune -f

echo.
echo [INFO] Building ConMED-RL Research Environment...
docker-compose -f docker-compose.research.yml build --no-cache

if %errorlevel% neq 0 (
    echo ERROR: Docker build failed!
    echo Check the error messages above for details.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.

REM Ask user what to do next
echo What would you like to do next?
echo [1] Start research environment (Jupyter Lab)
echo [2] Start development environment (Jupyter + Flask)
echo [3] Just build (exit)
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo [INFO] Starting research environment...
    docker-compose -f docker-compose.research.yml up -d
    if !errorlevel! equ 0 (
        echo.
        echo ========================================
        echo Research environment started!
        echo ========================================
        echo Jupyter Lab: http://localhost:8888
        echo Token: conmed-rl-research
        echo Flask App: http://localhost:5000
        echo.
        echo Available directories:
        echo   - /app/ConMedRL/ - Core OCRL framework
        echo   - /app/Data/ - Data processing modules
        echo   - /app/Experiment Notebook/ - Jupyter notebooks
        echo   - /app/CDM-Software/ - Clinical decision support
        echo.
        echo To view logs: docker-compose -f docker-compose.research.yml logs -f
        echo To stop: docker-compose -f docker-compose.research.yml down
    )
) else if "%choice%"=="2" (
    echo.
    echo [INFO] Starting development environment...
    docker-compose -f docker-compose.dev.yml up -d
    if !errorlevel! equ 0 (
        echo.
        echo ========================================
        echo Development environment started!
        echo ========================================
        echo Jupyter Lab: http://localhost:8888
        echo Token: conmed-rl-dev
        echo Flask App: http://localhost:5000
        echo.
        echo To view logs: docker-compose -f docker-compose.dev.yml logs -f
        echo To stop: docker-compose -f docker-compose.dev.yml down
    )
) else (
    echo.
    echo [INFO] Build completed. You can start the environment later with:
    echo   Research: docker-compose -f docker-compose.research.yml up -d
    echo   Development: docker-compose -f docker-compose.dev.yml up -d
)

echo.
pause 
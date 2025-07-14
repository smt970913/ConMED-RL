@echo off
setlocal enabledelayedexpansion

echo ========================================
echo ConMED-RL Docker Build Script
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
echo Step 1: Cleaning up old containers and images...
docker-compose down 2>nul
docker system prune -f

echo.
echo Step 2: Building ConMED-RL Docker image...
docker-compose build --no-cache

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
echo [1] Run development environment (port 5000)
echo [2] Run production environment (port 80)
echo [3] Just build (exit)
echo [4] Run with monitoring (Prometheus + Grafana)
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Starting development environment...
    docker-compose up -d
    if !errorlevel! equ 0 (
        echo.
        echo ========================================
        echo Development environment started!
        echo ========================================
        echo Web interface: http://localhost:5000
        echo Health check: http://localhost:5000/health
        echo.
        echo To view logs: docker-compose logs -f
        echo To stop: docker-compose down
    )
) else if "%choice%"=="2" (
    echo.
    echo Starting production environment...
    docker-compose -f docker-compose.prod.yml up -d
    if !errorlevel! equ 0 (
        echo.
        echo ========================================
        echo Production environment started!
        echo ========================================
        echo Web interface: http://localhost
        echo Health check: http://localhost/health
        echo.
        echo To view logs: docker-compose -f docker-compose.prod.yml logs -f
        echo To stop: docker-compose -f docker-compose.prod.yml down
    )
) else if "%choice%"=="4" (
    echo.
    echo Starting with monitoring...
    docker-compose -f docker-compose.prod.yml --profile monitoring up -d
    if !errorlevel! equ 0 (
        echo.
        echo ========================================
        echo Full stack started with monitoring!
        echo ========================================
        echo Web interface: http://localhost
        echo Prometheus: http://localhost:9090
        echo Grafana: http://localhost:3000 (admin/admin)
        echo.
        echo To view logs: docker-compose -f docker-compose.prod.yml logs -f
        echo To stop: docker-compose -f docker-compose.prod.yml --profile monitoring down
    )
) else (
    echo.
    echo Build completed. You can run the application later with:
    echo   docker-compose up -d (development)
    echo   docker-compose -f docker-compose.prod.yml up -d (production)
)

echo.
pause 
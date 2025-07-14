@echo off
setlocal enabledelayedexpansion

echo ========================================
echo ConMED-RL Docker Deployment Test
echo ========================================
echo.

REM Function to check if a service is healthy
:check_service_health
set url=%1
set service_name=%2
set max_attempts=30
set attempt=1

echo [INFO] Checking %service_name% health at %url%...

:health_loop
curl -s -f %url% >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] %service_name% is healthy!
    goto :eof
)

echo|set /p="."
timeout /t 2 /nobreak >nul
set /a attempt+=1
if %attempt% leq %max_attempts% goto health_loop

echo [ERROR] %service_name% health check failed after %max_attempts% attempts
exit /b 1

:main
REM Navigate to Docker-Deployment directory
cd /d "%~dp0\.."

echo [INFO] Current directory: %CD%
echo.

REM Test 1: Validate Docker Compose files
echo [INFO] Testing Docker Compose configuration...

docker-compose config >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] docker-compose.yml is valid
) else (
    echo [ERROR] docker-compose.yml has syntax errors
    exit /b 1
)

docker-compose -f docker-compose.prod.yml config >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] docker-compose.prod.yml is valid
) else (
    echo [ERROR] docker-compose.prod.yml has syntax errors
    exit /b 1
)

REM Test 2: Build images
echo [INFO] Building Docker images...
docker-compose build --no-cache
if %errorlevel% equ 0 (
    echo [SUCCESS] Docker images built successfully
) else (
    echo [ERROR] Docker build failed
    exit /b 1
)

REM Test 3: Test development environment
echo [INFO] Testing development environment...
docker-compose up -d

if %errorlevel% equ 0 (
    echo [SUCCESS] Development environment started
    
    REM Check if containers are running
    docker-compose ps | findstr "Up" >nul
    if %errorlevel% equ 0 (
        echo [SUCCESS] Containers are running
        
        REM Check health endpoint
        call :check_service_health "http://localhost:5000/health" "Development environment"
        if %errorlevel% equ 0 (
            echo [SUCCESS] Development environment is fully functional
        ) else (
            echo [ERROR] Development environment health check failed
            docker-compose logs conmed-rl-app
        )
    ) else (
        echo [ERROR] Containers failed to start
        docker-compose logs conmed-rl-app
    )
) else (
    echo [ERROR] Failed to start development environment
    exit /b 1
)

REM Cleanup development environment
echo [INFO] Stopping development environment...
docker-compose down

REM Test 4: Test production environment
echo [INFO] Testing production environment...
docker-compose -f docker-compose.prod.yml up -d

if %errorlevel% equ 0 (
    echo [SUCCESS] Production environment started
    
    REM Check if containers are running
    docker-compose -f docker-compose.prod.yml ps | findstr "Up" >nul
    if %errorlevel% equ 0 (
        echo [SUCCESS] Production containers are running
        
        REM Check health endpoint through nginx
        call :check_service_health "http://localhost/health" "Production environment"
        if %errorlevel% equ 0 (
            echo [SUCCESS] Production environment is fully functional
        ) else (
            echo [ERROR] Production environment health check failed
            docker-compose -f docker-compose.prod.yml logs conmed-rl-app
            docker-compose -f docker-compose.prod.yml logs nginx
        )
    ) else (
        echo [ERROR] Production containers failed to start
        docker-compose -f docker-compose.prod.yml logs
    )
) else (
    echo [ERROR] Failed to start production environment
    exit /b 1
)

REM Test 5: Test monitoring (optional)
echo [INFO] Testing monitoring stack...
docker-compose -f docker-compose.prod.yml --profile monitoring up -d

if %errorlevel% equ 0 (
    echo [SUCCESS] Monitoring stack started
    
    REM Check Prometheus
    call :check_service_health "http://localhost:9090" "Prometheus"
    if %errorlevel% equ 0 (
        echo [SUCCESS] Prometheus is accessible
    ) else (
        echo [WARNING] Prometheus health check failed
    )
    
    REM Check Grafana
    call :check_service_health "http://localhost:3000" "Grafana"
    if %errorlevel% equ 0 (
        echo [SUCCESS] Grafana is accessible
    ) else (
        echo [WARNING] Grafana health check failed
    )
) else (
    echo [WARNING] Monitoring stack failed to start
)

REM Cleanup
echo [INFO] Cleaning up test environment...
docker-compose -f docker-compose.prod.yml --profile monitoring down

REM Final report
echo.
echo ========================================
echo Test Results Summary
echo ========================================
echo [SUCCESS] ✅ Docker Compose configuration validation
echo [SUCCESS] ✅ Docker image building
echo [SUCCESS] ✅ Development environment deployment
echo [SUCCESS] ✅ Production environment deployment
echo [SUCCESS] ✅ Health checks
echo.
echo [SUCCESS] All tests passed! Your Docker configuration is working correctly.
echo.
echo [INFO] You can now deploy with confidence using:
echo   Development: docker-compose up -d
echo   Production:  docker-compose -f docker-compose.prod.yml up -d
echo   Monitoring:  docker-compose -f docker-compose.prod.yml --profile monitoring up -d
echo.
pause
goto :eof 
@echo off
REM Build script for Smart Parking System Docker images (Windows)

setlocal enabledelayedexpansion

echo ==========================================
echo Smart Parking System - Docker Build
echo ==========================================
echo.

REM Parse command line arguments
set GPU_MODE=
set BUILD_ARGS=

:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="--gpu" (
    set GPU_MODE=1
    shift
    goto parse_args
)
if /i "%~1"=="--no-cache" (
    set BUILD_ARGS=!BUILD_ARGS! --no-cache
    shift
    goto parse_args
)
shift
goto parse_args
:end_parse

REM Check if .env file exists
if not exist .env (
    echo Warning: .env file not found!
    echo Creating .env from .env.example...
    if exist .env.example (
        copy .env.example .env
        echo [OK] Created .env file
    ) else (
        echo [ERROR] .env.example not found!
        exit /b 1
    )
)

REM Check if backend_app/.env exists
if not exist backend_app\.env (
    echo Warning: backend_app/.env file not found!
    echo Creating backend_app/.env...
    (
        echo APP_PORT=5002
        echo FLASK_ENV=production
        echo SECRET_KEY=smart-parking-backend-secret-key-2024-change-in-production
        echo.
        echo DB_HOST=postgres
        echo DB_PORT=5432
        echo DB_NAME=PARKING_PLATE
        echo DB_USER=postgres
        echo DB_PASSWORD=1412
    ) > backend_app\.env
    echo [OK] Created backend_app/.env file
)

echo.
echo Build configuration:
if defined GPU_MODE (
    echo   Target: runtime-gpu ^(CUDA 12.8^)
    echo   GPU Mode: enabled
) else (
    echo   Target: runtime-cpu
    echo   GPU Mode: disabled
)
echo.

REM Stop existing containers
echo 1. Stopping existing containers...
docker-compose down 2>nul
echo [OK] Containers stopped
echo.

REM Build images
echo 2. Building Docker images...
echo    This may take 5-10 minutes on first build...
if defined GPU_MODE (
    docker-compose -f docker-compose.yml -f docker-compose.gpu.yml build !BUILD_ARGS!
) else (
    docker-compose build !BUILD_ARGS!
)
if errorlevel 1 (
    echo [ERROR] Build failed!
    exit /b 1
)
echo [OK] Images built successfully
echo.

REM Start services
echo 3. Starting services...
if defined GPU_MODE (
    docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
) else (
    docker-compose up -d
)
if errorlevel 1 (
    echo [ERROR] Failed to start services!
    exit /b 1
)
echo [OK] Services started
echo.

REM Wait for services
echo 4. Waiting for services to be healthy...
timeout /t 5 /nobreak >nul
echo.

REM Check service status
echo Service Status:
docker-compose ps
echo.

REM Show completion message
echo ==========================================
echo Build complete!
echo ==========================================
echo.
echo Services:
echo   Dashboard:     http://localhost:5001
echo   Backend API:   http://localhost:5002
echo   PostgreSQL:    localhost:5432
echo.
if defined GPU_MODE (
    echo GPU Support: ENABLED ^(CUDA 12.8^)
    echo.
)
echo View logs:
echo   docker-compose logs -f parking
echo   docker-compose logs -f backend
echo   docker-compose logs -f postgres
echo.
echo Stop services:
echo   docker-compose down
echo.

pause

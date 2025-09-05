@echo off
echo Starting ASL Classifier Backend...
echo.

REM Change to project root directory (parent of scripts)
cd /d "%~dp0\.."

REM Navigate to backend directory
cd backend

REM Check if we can use our optimized startup script
if exist "tests\quick_start.py" (
    echo Using optimized startup script with model validation...
    echo.
    cd tests
    python quick_start.py
) else (
    echo Using fallback startup method...
    echo.
    
    REM Try different ports if 8000 is busy
    echo Trying to start server on available port...
    
    REM Try port 8001 first (our preferred port)
    echo Attempting to start on port 8001...
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload 2>nul
    
    if errorlevel 1 (
        echo Port 8001 busy, trying 8002...
        python -m uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload 2>nul
        
        if errorlevel 1 (
            echo Port 8002 busy, trying 8003...
            python -m uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload
        )
    )
)

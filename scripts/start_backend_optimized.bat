@echo off
echo ============================================================
echo 🧠 ASL CLASSIFIER BACKEND STARTUP
echo ============================================================
echo.

REM Change to project root directory (parent of scripts)
cd /d "%~dp0\.."

REM Navigate to backend/tests directory where our working script is
cd backend\tests

echo Starting optimized backend with model validation...
echo Server will be available on port 8001
echo.

REM Run our working Python startup script
python quick_start.py

echo.
echo Backend stopped.
pause

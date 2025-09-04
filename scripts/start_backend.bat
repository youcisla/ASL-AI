@echo off
echo Starting ASL Classifier Backend...
echo.

REM Change to project root directory (parent of scripts)
cd /d "%~dp0\.."

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found. Please run install.bat first.
    pause
    exit /b 1
)

REM Navigate to backend directory
cd backend

REM Install/update dependencies
echo Installing dependencies...
python -m pip install -r requirements.txt

REM Start the FastAPI server
echo.
echo Starting FastAPI server on http://localhost:8000
echo API documentation available at http://localhost:8000/docs
echo.
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

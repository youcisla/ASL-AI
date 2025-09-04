@echo off
echo ========================================
echo ASL Classifier - Complete Installation
echo ========================================
echo.

REM Change to project root directory (parent of scripts)
cd /d "%~dp0\.."
echo Working directory: %CD%
echo.

REM Check prerequisites
echo Checking prerequisites...

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed
    echo Please install Python 3.11+ from https://www.python.org/
    pause
    exit /b 1
)

node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed  
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo ✓ Prerequisites check passed
echo.

REM Create virtual environment
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
    echo ✓ Virtual environment created
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install backend dependencies
echo.
echo Installing backend dependencies...
cd backend
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install backend dependencies
    pause
    exit /b 1
)
cd..

REM Install notebook dependencies (optional)
echo.
echo Installing notebook dependencies...
pip install -r requirements-notebook.txt
if %errorlevel% neq 0 (
    echo WARNING: Failed to install notebook dependencies (optional)
)

REM Install frontend dependencies
echo.
echo Installing frontend dependencies...
cd frontend
npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install frontend dependencies
    pause
    exit /b 1
)
cd..

REM Setup environment
if not exist ".env" (
    echo Creating .env file...
    copy ".env.example" ".env"
)

REM Create uploads directory
if not exist "uploads" (
    mkdir uploads
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Make sure MongoDB is running (mongod)
echo 2. Train your model using the Jupyter notebook
echo 3. Run start_backend.bat in one terminal
echo 4. Run start_frontend.bat in another terminal
echo.
echo Access URLs:
echo   Frontend: http://localhost:5173
echo   Backend: http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo.
pause

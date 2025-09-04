@echo off
echo ========================================
echo ASL Classifier Setup Script (Native)
echo ========================================
echo.

echo Checking requirements...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://www.python.org/
    pause
    exit /b 1
)

echo ✓ Python is installed

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo ✓ Node.js is installed

REM Check if MongoDB is running (optional check)
echo.
echo NOTE: Make sure MongoDB is installed and running
echo You can:
echo   1. Install MongoDB Community Server locally
echo   2. Use MongoDB Atlas (cloud)
echo   3. Use a local MongoDB Docker container

REM Check if model files exist
if not exist "models\vgg16_asl_final.keras" (
    echo WARNING: Model file not found: models\vgg16_asl_final.keras
    echo Please run the Jupyter notebook to train and save the model
    echo.
)

if not exist "models\labels.json" (
    echo WARNING: Labels file not found: models\labels.json
    echo Please run the Jupyter notebook to generate the labels file
    echo.
)

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo Creating .env file from template...
    copy ".env.example" ".env"
    echo ✓ Created .env file
    echo.
    echo You can edit .env to customize configuration
    echo.
) else (
    echo ✓ .env file exists
)

REM Create uploads directory
if not exist "uploads" (
    mkdir uploads
    echo ✓ Created uploads directory
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
    echo ✓ Created virtual environment
)

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To start the application:
echo.
echo 1. Activate virtual environment:
echo    venv\Scripts\activate
echo.
echo 2. Install backend dependencies:
echo    cd backend
echo    pip install -r requirements.txt
echo.
echo 3. Start MongoDB (if not running):
echo    mongod
echo.
echo 4. Start backend (in one terminal):
echo    cd backend
echo    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
echo.
echo 5. Install frontend dependencies (in another terminal):
echo    cd frontend
echo    npm install
echo.
echo 6. Start frontend:
echo    npm run dev
echo.
echo Access URLs:
echo   Frontend: http://localhost:5173
echo   Backend API: http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo.
pause

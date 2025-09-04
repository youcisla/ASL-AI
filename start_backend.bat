@echo off
echo Starting ASL Classifier Backend...
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Navigate to backend directory
cd backend

REM Install/update dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Start the FastAPI server
echo.
echo Starting FastAPI server on http://localhost:8000
echo API documentation available at http://localhost:8000/docs
echo.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

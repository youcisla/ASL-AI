@echo off
echo Starting ASL Classifier Frontend...
echo.

REM Change to project root directory (parent of scripts)
cd /d "%~dp0\.."

REM Navigate to frontend directory
cd frontend

REM Install/update dependencies
echo Installing dependencies...
npm install

REM Start the development server
echo.
echo Starting Vite development server on http://localhost:5173
echo.
npm run dev

#!/bin/bash
echo "Starting ASL Classifier Backend..."
echo

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Navigate to backend directory
cd backend

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the FastAPI server
echo
echo "Starting FastAPI server on http://localhost:8000"
echo "API documentation available at http://localhost:8000/docs"
echo
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

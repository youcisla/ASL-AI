#!/bin/bash

echo "========================================"
echo "ASL Classifier - Complete Installation"
echo "========================================"
echo

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.11+ from https://www.python.org/"
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

echo "✓ Prerequisites check passed"
echo

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install backend dependencies
echo
echo "Installing backend dependencies..."
cd backend
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install backend dependencies"
    exit 1
fi
cd ..

# Install notebook dependencies (optional)
echo
echo "Installing notebook dependencies..."
pip install -r requirements-notebook.txt
if [ $? -ne 0 ]; then
    echo "WARNING: Failed to install notebook dependencies (optional)"
fi

# Install frontend dependencies
echo
echo "Installing frontend dependencies..."
cd frontend
npm install
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install frontend dependencies"
    exit 1
fi
cd ..

# Setup environment
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp ".env.example" ".env"
fi

# Create uploads directory
if [ ! -d "uploads" ]; then
    mkdir -p uploads
fi

echo
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo
echo "Next steps:"
echo "1. Make sure MongoDB is running (mongod)"
echo "2. Train your model using the Jupyter notebook"
echo "3. Run ./start_backend.sh in one terminal"
echo "4. Run ./start_frontend.sh in another terminal"
echo
echo "Access URLs:"
echo "  Frontend: http://localhost:5173"
echo "  Backend: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo

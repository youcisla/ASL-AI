#!/bin/bash

echo "========================================"
echo "ASL Classifier Setup Script (Native)"
echo "========================================"
echo

echo "Checking requirements..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.11+ from https://www.python.org/"
    exit 1
fi

echo "✓ Python is installed"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

echo "✓ Node.js is installed"

# Check MongoDB availability
echo
echo "NOTE: Make sure MongoDB is installed and running"
echo "You can:"
echo "  1. Install MongoDB Community Server locally"
echo "  2. Use MongoDB Atlas (cloud)"
echo "  3. Use a local MongoDB Docker container: docker run -d -p 27017:27017 mongo:7"

# Check if model files exist
if [ ! -f "models/vgg16_asl_final.keras" ]; then
    echo "WARNING: Model file not found: models/vgg16_asl_final.keras"
    echo "Please run the Jupyter notebook to train and save the model"
    echo
fi

if [ ! -f "models/labels.json" ]; then
    echo "WARNING: Labels file not found: models/labels.json"
    echo "Please run the Jupyter notebook to generate the labels file"
    echo
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp ".env.example" ".env"
    echo "✓ Created .env file"
    echo
    echo "You can edit .env to customize configuration"
    echo
else
    echo "✓ .env file exists"
fi

# Create uploads directory
if [ ! -d "uploads" ]; then
    mkdir -p uploads
    echo "✓ Created uploads directory"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo "✓ Created virtual environment"
fi

echo
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo
echo "To start the application:"
echo
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo
echo "2. Install backend dependencies:"
echo "   cd backend"
echo "   pip install -r requirements.txt"
echo
echo "3. Start MongoDB (if not running):"
echo "   mongod"
echo "   # OR use Docker: docker run -d -p 27017:27017 mongo:7"
echo
echo "4. Start backend (in one terminal):"
echo "   cd backend"
echo "   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo
echo "5. Install frontend dependencies (in another terminal):"
echo "   cd frontend"
echo "   npm install"
echo
echo "6. Start frontend:"
echo "   npm run dev"
echo
echo "Access URLs:"
echo "  Frontend: http://localhost:5173"
echo "  Backend API: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo

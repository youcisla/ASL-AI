#!/bin/bash
echo "Starting ASL Classifier Frontend..."
echo

# Navigate to frontend directory
cd frontend

# Install/update dependencies
echo "Installing dependencies..."
npm install

# Start the development server
echo
echo "Starting Vite development server on http://localhost:5173"
echo
npm run dev

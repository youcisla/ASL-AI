# ASL Hand-Sign Image Classifier

A production-ready monorepo for ASL (American Sign Language) hand-sign image classification using ├── frontend/          # React + TypeScript frontend
├── backend/           # FastAPI backend
├── models/            # ML models and labels
├── uploads/           # File storage (gitignored)
├── venv/              # Python virtual environment
├── .env.example       # Environment configuration template
├── start_backend.*    # Backend startup scripts
├── start_frontend.*   # Frontend startup scripts
└── README.md          # This fileransfer learning.

## Features

- **Frontend**: React + TypeScript + Vite + Bootstrap
- **Backend**: FastAPI with async MongoDB support
- **Database**: MongoDB with Motor (async driver)
- **ML Model**: VGG16 transfer learning for ASL classification

## Prerequisites

- **Python 3.11+** - [Download here](https://www.python.org/)
- **Node.js 18+** - [Download here](https://nodejs.org/)
- **MongoDB** - Choose one:
  - [MongoDB Community Server](https://www.mongodb.com/try/download/community) (local install)
  - [MongoDB Atlas](https://www.mongodb.com/atlas) (cloud)
  - Docker container: `docker run -d -p 27017:27017 mongo:7`

## Quick Start

1. **Setup the model and labels**:
   - Place your trained model at `./models/vgg16_asl_final.keras`
   - Create `./models/labels.json` with your class names in order

2. **Run setup script**:
   ```bash
   # Windows
   setup.bat
   
   # Linux/Mac
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Start MongoDB** (if not running):
   ```bash
   mongod
   # OR use Docker: docker run -d -p 27017:27017 mongo:7
   ```

4. **Start the backend** (in one terminal):
   ```bash
   # Windows
   start_backend.bat
   
   # Linux/Mac
   chmod +x start_backend.sh
   ./start_backend.sh
   ```

5. **Start the frontend** (in another terminal):
   ```bash
   # Windows
   start_frontend.bat
   
   # Linux/Mac
   chmod +x start_frontend.sh
   ./start_frontend.sh
   ```

6. **Access the application**:
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Predict ASL Sign
```bash
curl -F "image=@/path/to/img.png" http://localhost:8000/api/predict
```

### Get Prediction History
```bash
curl http://localhost:8000/api/history?limit=10&offset=0
```

### Submit Feedback
```bash
curl -X POST http://localhost:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"upload_id": "64f1a2b3c4d5e6f7g8h9i0j1", "is_correct": true}'
```

## Development

### Manual Setup

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Backend Development**:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Frontend Development**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

### Run Tests
```bash
cd backend
pytest

# Test the API
python ../test_api.py
```

## Project Structure

```
├── frontend/          # React + TypeScript frontend
├── backend/           # FastAPI backend
├── models/            # ML models and labels
├── uploads/           # File storage (gitignored)
├── docker-compose.yml # Container orchestration
├── .env.example       # Environment configuration template
└── README.md         # This file
```

## Database Schema

### Collections

- **uploads**: Image metadata and file paths
- **predictions**: Model predictions with Top-3 results
- **feedback**: User feedback on predictions

### Indexes

- `uploads.file_hash` (unique)
- `uploads.created_at`
- `predictions.upload_id`
- `predictions.created_at`
- `feedback.upload_id`
- `feedback.created_at`

## Features

- **Drag & Drop Upload**: Intuitive file upload interface
- **Real-time Predictions**: Instant ASL sign classification
- **Top-3 Results**: Shows confidence scores for top 3 predictions
- **Feedback System**: Users can correct wrong predictions
- **History Tracking**: View all previous predictions
- **Responsive Design**: Works on desktop and mobile
- **Native Development**: No Docker required, runs directly on your system

## Model Requirements

The model should be a Keras model that:
- Accepts input shape (224, 224, 3)
- Uses VGG16 preprocessing (`tf.keras.applications.vgg16.preprocess_input`)
- Returns softmax probabilities for ASL classes
- Is saved in Keras format (`.keras` extension)

## Optional Features

- **GridFS Storage**: Set `USE_GRIDFS=true` to store files in MongoDB GridFS
- **Batch Prediction**: Use `backend/dev_batch_predict.py` for bulk predictions

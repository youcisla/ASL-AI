# ASL Classifier - Deployment Guide

This guide walks you through deploying the ASL Hand-Sign Image Classifier from development to production.

## Prerequisites

- Docker Desktop installed
- At least 4GB RAM available
- 2GB free disk space
- Python 3.11+ (for training)
- Jupyter Notebook environment

## Quick Start (5 minutes)

1. **Clone/Download** this repository
2. **Train the model** (see Model Training section)
3. **Run setup script**:
   ```bash
   # Windows
   setup.bat
   
   # Linux/Mac
   chmod +x setup.sh
   ./setup.sh
   ```
4. **Start the application**:
   ```bash
   docker compose up --build
   ```
5. **Access the app** at http://localhost:5173

## Model Training

### Step 1: Prepare Data
- Download ASL alphabet dataset from Kaggle
- Extract to a folder accessible by the notebook
- Ensure folder structure: `train/A/`, `train/B/`, etc.

### Step 2: Run Training Notebook
1. Open `projet-ia.ipynb` in Jupyter
2. Update the dataset path in the first cell
3. Run all cells to train the model
4. **Important**: Run the final cell to save model files:
   - `models/vgg16_asl_final.keras`
   - `models/labels.json`

### Step 3: Verify Model Files
```bash
ls -la models/
# Should show:
# vgg16_asl_final.keras (~60-80MB)
# labels.json (~1KB)
```

## Configuration

### Environment Variables (.env)
```bash
# Database
MONGO_URI=mongodb://mongo:27017
MONGO_DB=asl_db

# Backend
BACKEND_PORT=8000
MODEL_PATH=./models/vgg16_asl_final.keras
LABELS_PATH=./models/labels.json

# Frontend
FRONTEND_PORT=5173
ALLOWED_ORIGINS=http://localhost:5173

# Storage (optional)
USE_GRIDFS=false
MAX_FILE_SIZE_MB=10
```

### Docker Compose Services
- **mongo**: MongoDB database
- **mongo-express**: Database admin UI
- **backend**: FastAPI application
- **frontend**: React application

## Deployment Options

### Option 1: Local Development
```bash
docker compose up --build
```

### Option 2: Background/Production
```bash
docker compose up --build -d
```

### Option 3: Individual Services
```bash
# Start only database
docker compose up mongo -d

# Start backend for development
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Start frontend for development
cd frontend
npm install
npm run dev
```

## Testing

### Automated API Tests
```bash
# Ensure API is running first
python test_api.py
```

### Manual Testing
1. **Health Check**: GET http://localhost:8000/health
2. **Upload Image**: Use frontend or curl:
   ```bash
   curl -F "image=@test_image.png" http://localhost:8000/api/predict
   ```
3. **View History**: Visit http://localhost:5173/history

### Unit Tests
```bash
cd backend
pytest
```

## Access URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | http://localhost:5173 | Web application |
| Backend API | http://localhost:8000 | REST API |
| API Docs | http://localhost:8000/docs | Swagger documentation |
| MongoDB | mongodb://localhost:27017 | Database |
| Mongo Express | http://localhost:8081 | Database admin |

## Production Deployment

### Security Considerations
1. **Change default passwords** in mongo-express
2. **Use environment-specific .env** files
3. **Enable HTTPS** with reverse proxy (nginx/traefik)
4. **Restrict CORS origins** to production domains
5. **Set up monitoring** and logging

### Scaling
1. **Horizontal scaling**: Run multiple backend containers
2. **Load balancing**: Use nginx or cloud load balancer
3. **Database**: Use MongoDB Atlas or managed MongoDB
4. **File storage**: Consider cloud storage for uploads

### Example Production docker-compose.yml
```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    environment:
      - MONGO_URI=mongodb://your-mongodb-instance
      - ALLOWED_ORIGINS=https://your-domain.com
    volumes:
      - ./models:/app/models:ro
    deploy:
      replicas: 2

  frontend:
    build: 
      context: ./frontend
      args:
        VITE_API_BASE: https://api.your-domain.com
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
```

## Troubleshooting

### Common Issues

#### 1. Model file not found
```
FileNotFoundError: Model file not found: ./models/vgg16_asl_final.keras
```
**Solution**: Run the training notebook and execute the final cell.

#### 2. Database connection failed
```
pymongo.errors.ServerSelectionTimeoutError
```
**Solution**: Ensure MongoDB container is running and healthy.

#### 3. Out of memory during prediction
```
ResourceExhaustedError: OOM when allocating tensor
```
**Solution**: Reduce batch size or increase container memory limits.

#### 4. CORS errors in frontend
```
Access to fetch at 'http://localhost:8000' from origin 'http://localhost:5173' has been blocked
```
**Solution**: Check ALLOWED_ORIGINS environment variable.

#### 5. Frontend build fails
```
Module not found: Error: Can't resolve 'react'
```
**Solution**: Run `npm install` in frontend directory.

### Debug Commands

```bash
# Check container logs
docker compose logs backend
docker compose logs frontend
docker compose logs mongo

# Check container status
docker compose ps

# Restart services
docker compose restart backend

# Full cleanup and rebuild
docker compose down -v
docker compose up --build

# Enter container shell
docker compose exec backend bash
docker compose exec frontend sh
```

### Performance Optimization

1. **Model optimization**:
   - Use TensorFlow Lite for faster inference
   - Implement model quantization
   - Cache predictions for duplicate images

2. **Database optimization**:
   - Add appropriate indexes
   - Use database connection pooling
   - Implement result caching

3. **Frontend optimization**:
   - Implement image compression before upload
   - Add service worker for caching
   - Use virtual scrolling for history

## Monitoring

### Health Checks
- Backend: GET `/health`
- Database: MongoDB ping command
- Frontend: HTTP status check

### Metrics to Monitor
- API response times
- Prediction accuracy feedback
- Database connection count
- Memory usage
- Disk space (uploads directory)

### Logging
- Application logs: Docker container logs
- Access logs: nginx/reverse proxy logs
- Error tracking: Consider Sentry integration

## Backup and Recovery

### Database Backup
```bash
docker compose exec mongo mongodump --out /backup/$(date +%Y%m%d)
```

### Model Files Backup
```bash
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/
```

### Uploads Backup
```bash
tar -czf uploads_backup_$(date +%Y%m%d).tar.gz uploads/
```

## Support

### Documentation
- API docs: http://localhost:8000/docs
- Frontend components: See `frontend/src/components/`
- Backend routes: See `backend/app/routes.py`

### Common Configuration
- Model path: Update `MODEL_PATH` in .env
- Database: Update `MONGO_URI` in .env
- CORS: Update `ALLOWED_ORIGINS` in .env
- File size limits: Update `MAX_FILE_SIZE_MB` in .env

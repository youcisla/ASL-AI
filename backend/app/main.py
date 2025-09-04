from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import os

from app.config import settings
from app.database import connect_to_mongo, close_mongo_connection
from app.ml_service import ml_service
from app.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting ASL Classifier API")
    
    # Connect to database
    await connect_to_mongo()
    
    # Initialize ML service
    await ml_service.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down ASL Classifier API")
    await close_mongo_connection()

app = FastAPI(
    title="ASL Classifier API",
    description="ASL Hand-Sign Image Classification API using VGG16 Transfer Learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

# Serve static files (uploaded images)
if os.path.exists(settings.uploads_dir):
    app.mount("/static", StaticFiles(directory=settings.uploads_dir), name="static")

@app.get("/")
async def root():
    return {
        "message": "ASL Classifier API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

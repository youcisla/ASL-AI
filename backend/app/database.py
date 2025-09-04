from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class Database:
    client: AsyncIOMotorClient = None
    database: AsyncIOMotorDatabase = None

db = Database()

async def connect_to_mongo():
    """Create database connection"""
    try:
        db.client = AsyncIOMotorClient(settings.mongo_uri)
        db.database = db.client[settings.mongo_db]
        
        # Test connection
        await db.client.admin.command('ping')
        logger.info(f"Connected to MongoDB at {settings.mongo_uri}")
        
        # Create indexes
        await create_indexes()
        
    except Exception as e:
        logger.error(f"Could not connect to MongoDB: {e}")
        raise

async def close_mongo_connection():
    """Close database connection"""
    if db.client:
        db.client.close()
        logger.info("Disconnected from MongoDB")

async def create_indexes():
    """Create database indexes for optimal performance"""
    try:
        # uploads collection indexes
        await db.database.uploads.create_index("file_hash", unique=True)
        await db.database.uploads.create_index([("created_at", -1)])
        
        # predictions collection indexes
        await db.database.predictions.create_index("upload_id")
        await db.database.predictions.create_index([("created_at", -1)])
        
        # feedback collection indexes
        await db.database.feedback.create_index("upload_id")
        await db.database.feedback.create_index([("created_at", -1)])
        
        logger.info("Database indexes created successfully")
        
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")
        raise

async def get_database() -> AsyncIOMotorDatabase:
    """Get database instance"""
    return db.database

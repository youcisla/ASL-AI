#!/usr/bin/env python3
"""
Batch prediction script for bulk processing of ASL images.
Useful for backfilling the database with predictions from a local folder.
"""

import asyncio
import argparse
import logging
from pathlib import Path
from typing import List
import time

from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
from app.ml_service import MLService, preprocess_image, safe_file_hash, get_image_dimensions
from app.models import Upload, Prediction, PredictionItem
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_images_batch(image_folder: Path, batch_size: int = 10):
    """Process images in batches and store predictions in database"""
    
    # Initialize ML service
    ml_service = MLService()
    await ml_service.initialize()
    
    if not ml_service.is_ready():
        raise RuntimeError("Failed to load model and labels")
    
    # Connect to database
    client = AsyncIOMotorClient(settings.mongo_uri)
    db = client[settings.mongo_db]
    
    try:
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = [
            f for f in image_folder.rglob('*') 
            if f.suffix.lower() in image_extensions
        ]
        
        logger.info(f"Found {len(image_files)} images to process")
        
        processed = 0
        skipped = 0
        errors = 0
        
        # Process in batches
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1} ({len(batch)} images)")
            
            for image_path in batch:
                try:
                    # Read image file
                    with open(image_path, 'rb') as f:
                        file_bytes = f.read()
                    
                    # Check if already processed
                    file_hash = safe_file_hash(file_bytes)
                    existing = await db.uploads.find_one({"file_hash": file_hash})
                    
                    if existing:
                        logger.debug(f"Skipping {image_path.name} (already processed)")
                        skipped += 1
                        continue
                    
                    # Get image dimensions
                    width, height = get_image_dimensions(file_bytes)
                    
                    # Copy file to uploads directory if needed
                    uploads_dir = Path(settings.uploads_dir)
                    uploads_dir.mkdir(exist_ok=True)
                    
                    new_filename = f"{uuid.uuid4()}{image_path.suffix}"
                    new_path = uploads_dir / new_filename
                    
                    with open(new_path, 'wb') as f:
                        f.write(file_bytes)
                    
                    # Create upload record
                    upload_doc = Upload(
                        file_path=str(new_path),
                        file_hash=file_hash,
                        width=width,
                        height=height
                    )
                    
                    result = await db.uploads.insert_one(
                        upload_doc.model_dump(by_alias=True, exclude={"id"})
                    )
                    upload_id = result.inserted_id
                    
                    # Make prediction
                    start_time = time.time()
                    img_array = preprocess_image(file_bytes)
                    top3_results = ml_service.predict_topk(img_array, k=3)
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Store prediction
                    prediction_doc = Prediction(
                        upload_id=upload_id,
                        top1_label=top3_results[0]["label"],
                        top1_prob=top3_results[0]["prob"],
                        top3=[
                            PredictionItem(label=item["label"], prob=item["prob"]) 
                            for item in top3_results
                        ],
                        latency_ms=latency_ms
                    )
                    
                    await db.predictions.insert_one(
                        prediction_doc.model_dump(by_alias=True, exclude={"id"})
                    )
                    
                    processed += 1
                    logger.info(f"Processed {image_path.name}: {top3_results[0]['label']} ({top3_results[0]['prob']:.3f})")
                    
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    errors += 1
                    continue
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        logger.info(f"Batch processing complete:")
        logger.info(f"  Processed: {processed}")
        logger.info(f"  Skipped: {skipped}")
        logger.info(f"  Errors: {errors}")
        
    finally:
        client.close()

async def main():
    parser = argparse.ArgumentParser(description="Batch predict ASL images")
    parser.add_argument("folder", type=Path, help="Folder containing images to process")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    
    args = parser.parse_args()
    
    if not args.folder.exists():
        logger.error(f"Folder does not exist: {args.folder}")
        return
    
    if not args.folder.is_dir():
        logger.error(f"Path is not a directory: {args.folder}")
        return
    
    logger.info(f"Starting batch prediction for folder: {args.folder}")
    await process_images_batch(args.folder, args.batch_size)

if __name__ == "__main__":
    asyncio.run(main())

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.database import get_database
from app.models import (
    PredictResponse, FeedbackRequest, HealthResponse, 
    HistoryResponse, HistoryItem, Upload, Prediction, 
    Feedback, PredictionItem
)
from app.ml_service import ml_service, preprocess_image, safe_file_hash, get_image_dimensions
from app.config import settings
from bson import ObjectId
from pathlib import Path
import uuid
import time
import logging
from typing import List

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncIOMotorDatabase = Depends(get_database)):
    """Health check endpoint"""
    try:
        # Check database connection
        await db.command("ping")
        db_status = "connected"
    except Exception:
        db_status = "disconnected"
    
    return HealthResponse(
        status="ok",
        model_loaded=ml_service.is_ready(),
        num_labels=ml_service.get_num_labels(),
        db=db_status
    )

@router.get("/api/labels")
async def get_labels():
    """Get all available labels"""
    if not ml_service.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ml_service.get_labels()

@router.post("/api/predict", response_model=PredictResponse)
async def predict_image(
    image: UploadFile = File(...),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Predict ASL sign from uploaded image"""
    
    # Check if model is ready
    if not ml_service.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read file bytes
        file_bytes = await image.read()
        
        # Check file size
        if len(file_bytes) > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
            )
        
        # Compute file hash
        file_hash = safe_file_hash(file_bytes)
        
        # Check if we've seen this image before
        existing_upload = await db.uploads.find_one({"file_hash": file_hash})
        
        if existing_upload:
            # Get the most recent prediction for this upload
            prediction = await db.predictions.find_one(
                {"upload_id": existing_upload["_id"]},
                sort=[("created_at", -1)]
            )
            
            if prediction:
                return PredictResponse(
                    upload_id=str(existing_upload["_id"]),
                    prediction_id=str(prediction["_id"]),
                    top3=[
                        PredictionItem(label=item["label"], prob=item["prob"]) 
                        for item in prediction["top3"]
                    ],
                    latency_ms=prediction["latency_ms"]
                )
        
        # New image - process it
        # Get image dimensions
        width, height = get_image_dimensions(file_bytes)
        
        # Save file to disk
        file_extension = Path(image.filename).suffix if image.filename else '.png'
        filename = f"{uuid.uuid4()}{file_extension}"
        file_path = Path(settings.uploads_dir) / filename
        
        # Ensure uploads directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            f.write(file_bytes)
        
        # Create upload record
        upload_doc = Upload(
            file_path=str(file_path),
            file_hash=file_hash,
            width=width,
            height=height
        )
        
        result = await db.uploads.insert_one(upload_doc.model_dump(by_alias=True, exclude={"id"}))
        upload_id = result.inserted_id
        
        # Preprocess image for prediction
        start_time = time.time()
        img_array = preprocess_image(file_bytes)
        
        # Make prediction
        top3_results = ml_service.predict_topk(img_array, k=3)
        latency_ms = (time.time() - start_time) * 1000
        
        # Create prediction record
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
        
        pred_result = await db.predictions.insert_one(
            prediction_doc.model_dump(by_alias=True, exclude={"id"})
        )
        
        return PredictResponse(
            upload_id=str(upload_id),
            prediction_id=str(pred_result.inserted_id),
            top3=[
                PredictionItem(label=item["label"], prob=item["prob"]) 
                for item in top3_results
            ],
            latency_ms=latency_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_image: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/api/feedback")
async def submit_feedback(
    feedback_data: FeedbackRequest,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Submit feedback for a prediction"""
    
    try:
        # Validate upload_id
        if not ObjectId.is_valid(feedback_data.upload_id):
            raise HTTPException(status_code=400, detail="Invalid upload_id")
        
        upload_id = ObjectId(feedback_data.upload_id)
        
        # Check if upload exists
        upload = await db.uploads.find_one({"_id": upload_id})
        if not upload:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        # Create feedback record
        feedback_doc = Feedback(
            upload_id=upload_id,
            is_correct=feedback_data.is_correct,
            correct_label=feedback_data.correct_label,
            notes=feedback_data.notes
        )
        
        await db.feedback.insert_one(feedback_doc.model_dump(by_alias=True, exclude={"id"}))
        
        return {"message": "Feedback submitted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in submit_feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

@router.get("/api/history", response_model=HistoryResponse)
async def get_history(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Get prediction history with pagination"""
    
    try:
        # Get predictions with pagination, sorted by creation date (newest first)
        predictions_cursor = db.predictions.find().sort("created_at", -1).skip(offset).limit(limit)
        predictions = await predictions_cursor.to_list(length=limit)
        
        # Get total count
        total = await db.predictions.count_documents({})
        
        # Get corresponding uploads
        upload_ids = [pred["upload_id"] for pred in predictions]
        uploads_cursor = db.uploads.find({"_id": {"$in": upload_ids}})
        uploads = await uploads_cursor.to_list(length=None)
        
        # Create lookup dict for uploads
        uploads_dict = {upload["_id"]: upload for upload in uploads}
        
        # Build history items
        items = []
        for pred in predictions:
            upload = uploads_dict.get(pred["upload_id"])
            if upload:
                items.append(HistoryItem(
                    upload_id=str(pred["upload_id"]),
                    file_path=upload["file_path"],
                    filename=Path(upload["file_path"]).name,
                    created_at=pred["created_at"],
                    top1=PredictionItem(
                        label=pred["top1_label"],
                        prob=pred["top1_prob"]
                    ),
                    top3=[
                        PredictionItem(label=item["label"], prob=item["prob"]) 
                        for item in pred["top3"]
                    ]
                ))
        
        return HistoryResponse(
            items=items,
            total=total,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        logger.error(f"Error in get_history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

from pydantic import BaseModel, Field, ConfigDict
from pydantic.json_schema import JsonSchemaValue
from typing import List, Optional, Dict, Any, Annotated
from datetime import datetime
from bson import ObjectId


class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.str_schema(),
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema: JsonSchemaValue) -> JsonSchemaValue:
        field_schema.update(type="string")
        return field_schema


class Upload(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    file_path: str
    file_hash: str
    width: int
    height: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PredictionItem(BaseModel):
    label: str
    prob: float


class Prediction(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
        protected_namespaces=()  # Allow model_ prefixed fields
    )
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    upload_id: str  # Changed from PyObjectId to str
    top1_label: str
    top1_prob: float
    top3: List[PredictionItem]
    ai_model_version: str = "asl_cnn_model"  # Renamed from model_version
    model_type: str = "cnn"  # Add model type field
    latency_ms: float
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Feedback(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    upload_id: str  # Changed from PyObjectId to str
    correct_label: Optional[str] = None
    is_correct: bool
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Request/Response models
class PredictResponse(BaseModel):
    upload_id: str
    prediction_id: str
    top3: List[PredictionItem]
    model_type: str = "cnn"  # Add model type to response
    model: Dict[str, Any] = {
        "name": "vgg16_asl_final",
        "input_size": [224, 224],
        "preprocess": "vgg16"
    }
    latency_ms: float


class FeedbackRequest(BaseModel):
    upload_id: str
    is_correct: bool
    correct_label: Optional[str] = None
    notes: Optional[str] = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()  # Allow model_ prefixed fields
    )
    
    status: str
    ai_model_loaded: bool  # Renamed from model_loaded
    num_labels: int
    db: str


class HistoryItem(BaseModel):
    upload_id: str
    file_path: str
    filename: str
    created_at: datetime
    top1: PredictionItem
    top3: List[PredictionItem]


class HistoryResponse(BaseModel):
    items: List[HistoryItem]
    total: int
    limit: int
    offset: int

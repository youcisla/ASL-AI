from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List, Optional


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        protected_namespaces=('settings_',)
    )
    
    # Database
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "asl_db"
    
    # Models
    cnn_model_path: str = "./models/custom_cnn_asl_model.keras"
    cnn_labels_path: str = "./models/custom_cnn_labels.json"
    vgg_model_path: str = "./models/vgg16_transfer_asl_final.keras"
    vgg_labels_path: str = "./models/vgg16_transfer_labels.json"
    
    # Legacy model paths (for backward compatibility)
    asl_model_path: str = "./models/custom_cnn_asl_model.keras"
    labels_path: str = "./models/custom_cnn_labels.json"
    
    # API
    allowed_origins: str = "http://localhost:5173"
    max_file_size_mb: int = 10
    
    # Storage
    use_gridfs: bool = False
    uploads_dir: str = "./uploads"
    
    @property
    def allowed_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


settings = Settings()

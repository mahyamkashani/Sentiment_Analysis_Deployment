from pydantic import BaseModel, Field


class InferenceConfig(BaseModel):
    """
    Configuration for TinyBERT inference.

    This class uses Pydantic for validation and configuration management.
    """
    model_path: str = "./fine_tuned_model"
    max_length: int = Field(gt=0, le=512, default=128)
    device: str = "cpu"  # "cpu" or "cuda"
    batch_size: int = Field(gt=0, default=32)

    class Config:
        """Pydantic configuration"""
        frozen = False  # Allow modification after creation

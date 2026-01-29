from pydantic import BaseModel, Field
from typing import List, Dict, Any


class InferenceRequest(BaseModel):
    """
    Request schema for inference endpoints.

    Validates that the texts field contains at least one string.
    """
    texts: List[str] = Field(min_length=1, description="List of texts to analyze")

    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["I love this product!", "This is terrible"]
            }
        }


class PredictionResponse(BaseModel):
    """
    Response schema for full prediction (all probabilities).
    Note: In practice, this is wrapped by ResponseEnvelope.
    """
    predictions: List[Dict[str, float]]

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {"negative": 0.05, "positive": 0.95},
                    {"negative": 0.88, "positive": 0.12}
                ]
            }
        }


class TopPredictionResponse(BaseModel):
    """
    Response schema for top prediction only.
    Note: In practice, this is wrapped by ResponseEnvelope.
    """
    predictions: List[Dict[str, Any]]

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {"label": "positive", "confidence": 0.95}
                ]
            }
        }

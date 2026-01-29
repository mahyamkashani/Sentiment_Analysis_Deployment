from typing import List, Dict, Any
from app.api.base_controller import BaseController
from app.api.api_decorators import put_in_envelope
from app.managers.tinybert_model_manager import TinyBERTModelManager
from app.schemas import InferenceRequest


class TinyBERTApiController(BaseController):
    """
    Controller for TinyBERT inference endpoints.

    This class handles HTTP request/response for prediction operations.

    Responsibilities:
    - Extract and validate request data
    - Call manager methods
    - Return raw data (decorator wraps it in ResponseEnvelope)
    """

    def __init__(self, manager: TinyBERTModelManager):
        super().__init__()
        self.manager = manager

    @put_in_envelope
    def predict(self, request: InferenceRequest) -> Dict[str, List[Dict[str, float]]]:
        """
        Predict sentiment for input texts with full probability distribution.

        Args:
            request: InferenceRequest with list of texts

        Returns:
            Dictionary with predictions (wrapped by decorator in ResponseEnvelope)

        Example Response (after decorator wrapping):
        {
            "status": "success",
            "code": 200,
            "message": "OK",
            "data": {
                "predictions": [
                    {"negative": 0.05, "positive": 0.95}
                ]
            }
        }
        """
        # Validate input
        if not request.texts:
            raise ValueError("Input texts cannot be empty")

        # Call manager for business logic
        predictions = self.manager.predict_with_labels(request.texts)

        # Return raw data - decorator wraps it in ResponseEnvelope
        return {"predictions": predictions}

    @put_in_envelope
    def predict_top(self, request: InferenceRequest) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get top prediction for input texts.

        Args:
            request: InferenceRequest with list of texts

        Returns:
            Dictionary with top predictions (wrapped by decorator)

        Example Response (after decorator wrapping):
        {
            "status": "success",
            "code": 200,
            "message": "OK",
            "data": {
                "predictions": [
                    {"label": "positive", "confidence": 0.95}
                ]
            }
        }
        """
        if not request.texts:
            raise ValueError("Input texts cannot be empty")

        # Call manager
        top_predictions = self.manager.get_top_prediction(request.texts)

        # Return raw data
        return {"predictions": top_predictions}

    @put_in_envelope
    def predict_batch(self, request: InferenceRequest) -> Dict[str, List[Dict[str, float]]]:
        """
        Process large batches of texts efficiently.

        Args:
            request: InferenceRequest with list of texts

        Returns:
            Dictionary with predictions (wrapped by decorator)
        """
        if not request.texts:
            raise ValueError("Input texts cannot be empty")

        # Call manager with batch processing
        predictions = self.manager.predict_batch(request.texts, batch_size=32)

        return {"predictions": predictions}

    @put_in_envelope
    def health_check(self) -> Dict[str, Any]:
        """
        Health check endpoint.

        Returns:
            Status information (wrapped by decorator)

        Example Response:
        {
            "status": "success",
            "code": 200,
            "message": "OK",
            "data": {
                "status": "healthy",
                "service": "TinyBERT Inference API",
                "model_loaded": true
            }
        }
        """
        is_ready = self.manager.is_ready()

        return {
            "status": "healthy" if is_ready else "not ready",
            "service": "TinyBERT Inference API",
            "model_loaded": is_ready
        }

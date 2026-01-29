from fastapi import APIRouter
from app.schemas import InferenceRequest
from app.api.v1.inference.tinybert_api_controller import TinyBERTApiController
from app.managers.tinybert_model_manager import TinyBERTModelManager
from app.config import InferenceConfig

"""
API Dispatcher - Route Registration

This file is equivalent to dispatcher_api_controller_v0.py from softremedy_report.
It registers all API routes and connects them to controller methods.

Pattern:
1. Create FastAPI router (equivalent to Flask Blueprint)
2. Initialize manager as singleton
3. Register routes that instantiate controllers and call methods
"""

# Create router (equivalent to Flask Blueprint)
api_v1_router = APIRouter(prefix="/api/v1", tags=["inference"])

# Initialize manager as singleton
# In production, consider using dependency injection or FastAPI dependencies
config = InferenceConfig(max_length=128, device="cpu")
manager = TinyBERTModelManager(config)

# Initialize model on startup
# Note: In production, this might be done in a startup event handler
try:
    manager.initialize(
        bucket_name=None,  # Set to S3 bucket name if using S3
        prefix=None        # Set to S3 prefix if using S3
    )
    print("✓ Model initialized successfully")
except Exception as e:
    print(f"✗ Model initialization failed: {e}")
    print("  Model will need to be manually initialized before accepting requests")


# ==================== ROUTE DEFINITIONS ====================

@api_v1_router.get("/health")
def health_check():
    """
    Health check endpoint.

    Similar to softremedy_report pattern:
    - Instantiate controller
    - Call controller method
    - Return result (controller method is decorated with @put_in_envelope)
    """
    controller = TinyBERTApiController(manager)
    return controller.health_check()


@api_v1_router.post("/predict")
def predict(request: InferenceRequest):
    """
    Predict sentiment for input texts.
    Returns probability distribution for all labels.

    Request Body:
    {
        "texts": ["I love this!", "This is terrible"]
    }

    Response:
    {
        "status": "success",
        "code": 200,
        "message": "OK",
        "data": {
            "predictions": [
                {"negative": 0.05, "positive": 0.95},
                {"negative": 0.88, "positive": 0.12}
            ]
        }
    }
    """
    controller = TinyBERTApiController(manager)
    return controller.predict(request)


@api_v1_router.post("/predict/top")
def predict_top(request: InferenceRequest):
    """
    Predict sentiment for input texts.
    Returns only the top prediction with confidence.

    Request Body:
    {
        "texts": ["Amazing experience!"]
    }

    Response:
    {
        "status": "success",
        "code": 200,
        "message": "OK",
        "data": {
            "predictions": [
                {"label": "positive", "confidence": 0.97}
            ]
        }
    }
    """
    controller = TinyBERTApiController(manager)
    return controller.predict_top(request)


@api_v1_router.post("/predict/batch")
def predict_batch(request: InferenceRequest):
    """
    Batch inference endpoint.
    Optimized for processing large batches of texts.

    Request Body:
    {
        "texts": ["text1", "text2", ..., "text1000"]
    }

    Response: Same as /predict but processes in batches internally
    """
    controller = TinyBERTApiController(manager)
    return controller.predict_batch(request)


# ==================== ADDITIONAL ROUTES ====================
# Add more routes here following the same pattern:
# 1. Define route with decorator
# 2. Instantiate controller
# 3. Call controller method
# 4. Return result

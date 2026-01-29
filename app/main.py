from fastapi import FastAPI
from app.api.dispatcher import api_v1_router

"""
Main FastAPI Application

This file initializes the FastAPI app and registers routers.
Following the pattern from softremedy_report, all routes are defined
in the dispatcher, and this file only handles app-level configuration.
"""

# Create FastAPI app
app = FastAPI(
    title="TinyBERT Inference API",
    description="Sentiment analysis using fine-tuned TinyBERT model. "
                "Built with 3-layer architecture: Dispatcher → Controller → Manager",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Register API router (all routes are defined in dispatcher.py)
app.include_router(api_v1_router)


# Root endpoint
@app.get("/")
def root():
    """
    Root endpoint - API information
    """
    return {
        "message": "TinyBERT API is running",
        "version": "1.0.0",
        "architecture": "3-layer (Dispatcher → Controller → Manager)",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": {
            "health": "GET /api/v1/health",
            "predict": "POST /api/v1/predict",
            "predict_top": "POST /api/v1/predict/top",
            "predict_batch": "POST /api/v1/predict/batch"
        }
    }


# Optional: Startup event handler
@app.on_event("startup")
async def startup_event():
    """
    Runs when the application starts.
    You can add initialization logic here if needed.
    """
    print("=" * 60)
    print("TinyBERT Inference API Starting...")
    print("=" * 60)
    print("Architecture: Dispatcher → Controller → Manager")
    print("Similar to softremedy_report pattern")
    print("=" * 60)


# Optional: Shutdown event handler
@app.on_event("shutdown")
async def shutdown_event():
    """
    Runs when the application shuts down.
    You can add cleanup logic here if needed.
    """
    print("TinyBERT Inference API Shutting down...")

# :dolphin: TinyBERT Service - Based on Java Spring Architecture

A sentiment analysis API built with FastAPI following the 3-layer architecture pattern.

## Architecture Overview

This project implements the **Dispatcher → Controller → Manager** pattern:

```
┌─────────────────────────────────────┐
│  Presentation Layer                 │
│  - Dispatcher (Front Controller)    │  ← Routing
│  - Controller (Request Handler)     │  ← HTTP concerns
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  Business Logic Layer               │
│  - Manager (Service Layer)          │  ← Business rules
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  Data Access Layer                  │
│  - Service (Infrastructure)         │  ← Data/Model operations
└─────────────────────────────────────┘

```

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Dispatcher (app/api/dispatcher.py)           │
│  - Route registration                                   │
│  - Controller instantiation                             │
│  - Request delegation                                   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Layer 2: Controller (tinybert_api_controller.py)       │
│  - HTTP request/response handling                       │
│  - Data validation                                      │
│  - Decorated with @put_in_envelope()                    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Layer 3: Manager (tinybert_model_manager.py)           │
│  - Business logic (inference, label mapping)            │
│  - Calls Model Service                                  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Layer 4: Service (model_service.py)                    │
│  - Model loading from S3/disk                           │
│  - Low-level infrastructure                             │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
tinybert_service_refactored/
├── app/
│   ├── __init__.py
│   ├── main.py                              # FastAPI app initialization
│   ├── config.py                            # Configuration
│   ├── schemas.py                           # Pydantic models
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── base_controller.py               # Base controller class
│   │   ├── response_envelope.py             # Response wrapper
│   │   ├── api_decorators.py                # @put_in_envelope decorator
│   │   ├── dispatcher.py                    # Route registration
│   │   │
│   │   └── v1/
│   │       ├── __init__.py
│   │       └── inference/
│   │           ├── __init__.py
│   │           └── tinybert_api_controller.py    # TinyBERT controller
│   │
│   ├── managers/
│   │   ├── __init__.py
│   │   └── tinybert_model_manager.py        # Business logic & inference
│   │
│   └── services/
│       ├── __init__.py
│       └── model_service.py                 # Model loading & S3 operations
│
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

## Key Components

### 1. ResponseEnvelope (response_envelope.py)

Wraps all API responses in a consistent format:

```json
{
  "status": "success",
  "code": 200,
  "message": "OK",
  "data": {
    "predictions": [...]
  }
}
```

### 2. @put_in_envelope Decorator (api_decorators.py)

Automatically wraps controller method responses:

```python
@put_in_envelope
def predict(self, request: InferenceRequest):
    predictions = self.manager.predict_with_labels(request.texts)
    return {"predictions": predictions}  # Raw data - decorator wraps it
```

### 3. Dispatcher (dispatcher.py)

Registers routes and delegates to controllers (similar to Flask Blueprint):

```python
@api_v1_router.post("/predict")
def predict(request: InferenceRequest):
    controller = TinyBERTApiController(manager)
    return controller.predict(request)
```

### 4. Controller (tinybert_api_controller.py)

Handles HTTP concerns:

```python
class TinyBERTApiController(BaseController):
    @put_in_envelope
    def predict(self, request: InferenceRequest):
        # Validate input
        if not request.texts:
            raise ValueError("Input texts cannot be empty")

        # Call manager
        predictions = self.manager.predict_with_labels(request.texts)

        # Return raw data
        return {"predictions": predictions}
```

### 5. Manager (tinybert_model_manager.py)

Implements business logic:

```python
class TinyBERTModelManager:
    def predict_with_labels(self, texts: List[str]):
        probabilities = self.predict_probabilities(texts)
        return self.map_labels(probabilities)
```

### 6. Service (model_service.py)

Handles infrastructure:

```python
class ModelService:
    def load_model(self, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
```

## API Endpoints

### Health Check
```bash
GET /api/v1/health
```

**Response:**
```json
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
```

### Predict (Full Probabilities)
```bash
POST /api/v1/predict
Content-Type: application/json

{
  "texts": ["I love this product!", "This is terrible"]
}
```

**Response:**
```json
{
  "status": "success",
  "code": 200,
  "message": "OK",
  "data": {
    "predictions": [
      {"negative": 0.05, "positive": 0.95},
      {"negative": 0.92, "positive": 0.08}
    ]
  }
}
```

### Predict Top Label
```bash
POST /api/v1/predict/top
Content-Type: application/json

{
  "texts": ["Amazing experience!"]
}
```

**Response:**
```json
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
```

### Batch Prediction
```bash
POST /api/v1/predict/batch
Content-Type: application/json

{
  "texts": ["text1", "text2", ..., "text1000"]
}
```

## Installation & Setup

### Option 1: Local Development

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install fastapi uvicorn pydantic transformers boto3 pytest httpx

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Docker

```bash
# Build image
docker build -t tinybert-service .

# Run container
docker run -d -p 8000:8000 --name tinybert tinybert-service

# View logs
docker logs -f tinybert

# Stop container
docker stop tinybert
```

## Usage Examples

### Using curl

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Predict sentiment
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I love this!", "This is bad"]}'

# Get top prediction
curl -X POST http://localhost:8000/api/v1/predict/top \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Amazing product!"]}'
```

### Using Python

```python
import requests

# Predict sentiment
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={"texts": ["I love this!", "This is terrible"]}
)

data = response.json()
print(data["data"]["predictions"])
# Output: [{'negative': 0.05, 'positive': 0.95}, {'negative': 0.92, 'positive': 0.08}]
```

## Request/Response Flow

```
1. HTTP Request
   POST /api/v1/predict
   │
   ▼
2. FastAPI Router (dispatcher.py)
   @api_v1_router.post("/predict")
   │
   ▼
3. Dispatcher Function
   controller = TinyBERTApiController(manager)
   return controller.predict(request)
   │
   ▼
4. Controller Method (@put_in_envelope)
   ├─ Validate input
   ├─ Call manager.predict_with_labels()
   └─ Return raw data: {"predictions": [...]}
   │
   ▼
5. Manager (Business Logic)
   ├─ predict_probabilities()
   │   ├─ Get tokenizer & model from Service
   │   ├─ Tokenize texts
   │   ├─ Run inference
   │   └─ Return probabilities
   └─ map_labels(probabilities)
   │
   ▼
6. @put_in_envelope Decorator
   ├─ Catches returned data
   ├─ Wraps in ResponseEnvelope.success()
   └─ Returns JSONResponse
   │
   ▼
7. Response to Client
   {
     "status": "success",
     "code": 200,
     "message": "OK",
     "data": {"predictions": [...]}
   }
```

## Key Benefits

1. **Separation of Concerns**: HTTP, business logic, and infrastructure are separate
2. **Consistent Responses**: All endpoints return the same format via decorator
3. **Error Handling**: Centralized exception handling in decorator
4. **Testability**: Each layer can be tested independently
5. **Scalability**: Easy to add new endpoints or models
6. **Maintainability**: Clear structure makes code easier to understand
7. **Type Safety**: Pydantic models throughout
8. **Auto Documentation**: FastAPI generates OpenAPI docs at `/docs`


## Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI Schema: http://localhost:8000/openapi.json

## License

MIT License

## Author

Mahya Kashani

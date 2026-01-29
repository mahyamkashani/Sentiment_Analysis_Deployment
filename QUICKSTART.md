# Quick Start Guide

## Run the Service Locally

### Step 1: Install Dependencies

```bash
cd tinybert_service_refactored

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install fastapi uvicorn pydantic transformers boto3 pytest httpx
```

### Step 2: Prepare Model Files

Place your fine-tuned TinyBERT model files in `./fine_tuned_model/` directory:

```
fine_tuned_model/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
├── vocab.txt
└── special_tokens_map.json
```

Or modify `app/api/dispatcher.py` to download from S3:

```python
manager.initialize(
    bucket_name="your-s3-bucket",
    prefix="path/to/model/"
)
```

### Step 3: Run the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 4: Test the API

Open your browser to http://localhost:8000/docs for interactive API documentation.

Or use curl:

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Predict sentiment
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I love this product!", "This is terrible"]}'
```

## Run with Docker

```bash
# Build image
docker build -t tinybert-service .

# Run container
docker run -d -p 8000:8000 --name tinybert tinybert-service

# Check logs
docker logs -f tinybert

# Test
curl http://localhost:8000/api/v1/health
```

## Architecture at a Glance

```
Request → Dispatcher → Controller → Manager → Service
                          ↓
                  @put_in_envelope
                          ↓
                   ResponseEnvelope
                          ↓
                      Response
```

## File Responsibilities

- **dispatcher.py**: Registers routes (`@api_v1_router.post(...)`)
- **controller**: Handles HTTP, calls manager, decorated with `@put_in_envelope`
- **manager**: Business logic (predict, map labels)
- **service**: Infrastructure (load model, S3 operations)
- **response_envelope**: Wraps all responses in consistent format

## Adding a New Endpoint

1. Add method to controller:
```python
@put_in_envelope
def analyze_batch(self, request: BatchRequest):
    results = self.manager.batch_analyze(request.data)
    return {"results": results}
```

2. Register route in dispatcher:
```python
@api_v1_router.post("/analyze/batch")
def analyze_batch(request: BatchRequest):
    controller = TinyBERTApiController(manager)
    return controller.analyze_batch(request)
```

3. Implement logic in manager:
```python
def batch_analyze(self, data):
    # Your logic here
    return processed_results
```

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns API information"""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["version"] == "1.0.0"


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["code"] == 200
    assert "data" in data
    assert "service" in data["data"]


def test_predict_endpoint_valid():
    """Test predict endpoint with valid input"""
    response = client.post(
        "/api/v1/predict",
        json={"texts": ["I love this product!", "This is terrible"]}
    )

    # Should return 200 even if model not loaded (will be in error state)
    data = response.json()
    assert "status" in data
    assert "code" in data
    assert "message" in data
    assert "data" in data


def test_predict_endpoint_empty_texts():
    """Test predict endpoint with empty texts list"""
    response = client.post(
        "/api/v1/predict",
        json={"texts": []}
    )

    # Pydantic validation should fail
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_endpoint_invalid_json():
    """Test predict endpoint with invalid JSON"""
    response = client.post(
        "/api/v1/predict",
        json={"wrong_field": ["text"]}
    )

    # Pydantic validation should fail
    assert response.status_code == 422


def test_predict_top_endpoint():
    """Test predict top endpoint"""
    response = client.post(
        "/api/v1/predict/top",
        json={"texts": ["Amazing product!"]}
    )

    data = response.json()
    assert "status" in data
    assert "code" in data


def test_predict_batch_endpoint():
    """Test batch prediction endpoint"""
    response = client.post(
        "/api/v1/predict/batch",
        json={"texts": ["text1", "text2", "text3"]}
    )

    data = response.json()
    assert "status" in data
    assert "code" in data


def test_response_envelope_structure():
    """Test that all responses follow ResponseEnvelope structure"""
    response = client.get("/api/v1/health")
    data = response.json()

    # All responses should have these fields
    required_fields = ["status", "code", "message", "data"]
    for field in required_fields:
        assert field in data

    # Status should be either "success" or "error"
    assert data["status"] in ["success", "error"]

    # Code should be an integer
    assert isinstance(data["code"], int)


def test_docs_endpoint():
    """Test that OpenAPI docs are accessible"""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema():
    """Test that OpenAPI schema is accessible"""
    response = client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert schema["info"]["title"] == "TinyBERT Inference API"


# Integration tests (require model to be loaded)
# To run: pytest tests/test_api.py -m integration
@pytest.mark.integration
class TestIntegration:
    """Integration tests that require a loaded model"""

    def test_predict_returns_probabilities(self):
        """Test that predict returns actual probabilities"""
        response = client.post(
            "/api/v1/predict",
            json={"texts": ["I love this!"]}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        predictions = data["data"]["predictions"]
        assert len(predictions) == 1
        assert "negative" in predictions[0]
        assert "positive" in predictions[0]

        # Probabilities should sum to approximately 1
        prob_sum = sum(predictions[0].values())
        assert 0.99 <= prob_sum <= 1.01

    def test_predict_top_returns_label(self):
        """Test that predict/top returns label and confidence"""
        response = client.post(
            "/api/v1/predict/top",
            json={"texts": ["This is amazing!"]}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        predictions = data["data"]["predictions"]
        assert len(predictions) == 1
        assert "label" in predictions[0]
        assert "confidence" in predictions[0]
        assert predictions[0]["label"] in ["negative", "positive"]
        assert 0 <= predictions[0]["confidence"] <= 1


# To run integration tests:
# pytest tests/test_api.py -m integration
#
# To run only non-integration tests:
# pytest tests/test_api.py -m "not integration"

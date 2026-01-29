import torch
from typing import List, Dict, Optional, Any
from app.services.model_service import ModelService
from app.config import InferenceConfig


class TinyBERTModelManager:
    """
    Business logic layer for TinyBERT inference.

    This manager handles:
    - Prediction logic and inference
    - Label mapping
    - Batch processing
    - Error handling for ML operations

    This class contains all business logic separated from HTTP concerns.
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model_service = ModelService()
        self.labels = ["negative", "positive"]
        self.device = config.device

    def initialize(self, bucket_name: Optional[str] = None, prefix: Optional[str] = None):
        """
        Initialize the model manager.
        Downloads from S3 if bucket info provided, then loads model.

        Args:
            bucket_name: Optional S3 bucket name
            prefix: Optional S3 prefix path
        """
        if bucket_name and prefix:
            self.model_service.download_from_s3(bucket_name, prefix)

        self.model_service.load_model(device=self.device)

    def predict_probabilities(self, texts: List[str]) -> List[List[float]]:
        """
        Run inference and return probability distributions.

        Args:
            texts: List of input texts

        Returns:
            List of probability distributions (one per text)

        Raises:
            ValueError: If texts is empty
        """
        if not texts:
            raise ValueError("Input texts cannot be empty")

        tokenizer = self.model_service.get_tokenizer()
        model = self.model_service.get_model()

        # Tokenize inputs
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )

        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Run inference
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)

        return probs.cpu().tolist()

    def map_labels(self, probabilities: List[List[float]]) -> List[Dict[str, float]]:
        """
        Map probability distributions to label dictionaries.

        Args:
            probabilities: List of probability distributions

        Returns:
            List of dictionaries mapping label to probability

        Example:
            Input: [[0.1, 0.9], [0.8, 0.2]]
            Output: [
                {"negative": 0.1, "positive": 0.9},
                {"negative": 0.8, "positive": 0.2}
            ]
        """
        results = []
        for prob_dist in probabilities:
            label_dict = {
                label: float(prob)
                for label, prob in zip(self.labels, prob_dist)
            }
            results.append(label_dict)
        return results

    def predict_with_labels(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Run inference and return labeled predictions.
        This is the main method used by controllers.

        Args:
            texts: List of input texts

        Returns:
            List of dictionaries with label probabilities

        Example:
            Input: ["I love this!", "This is terrible"]
            Output: [
                {"negative": 0.05, "positive": 0.95},
                {"negative": 0.88, "positive": 0.12}
            ]
        """
        probabilities = self.predict_probabilities(texts)
        return self.map_labels(probabilities)

    def get_top_prediction(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Get the top prediction for each text.

        Args:
            texts: List of input texts

        Returns:
            List of dictionaries with top label and confidence

        Example:
            Input: ["Amazing product!"]
            Output: [
                {"label": "positive", "confidence": 0.97}
            ]
        """
        labeled_predictions = self.predict_with_labels(texts)

        results = []
        for pred in labeled_predictions:
            top_label = max(pred, key=pred.get)
            results.append({
                "label": top_label,
                "confidence": pred[top_label]
            })

        return results

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
        """
        Process large batches of texts in chunks.

        Args:
            texts: List of input texts
            batch_size: Size of each batch

        Returns:
            List of dictionaries with label probabilities
        """
        all_results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.predict_with_labels(batch)
            all_results.extend(batch_results)

        return all_results

    def is_ready(self) -> bool:
        """Check if the manager is initialized and ready for inference"""
        return self.model_service.is_loaded()

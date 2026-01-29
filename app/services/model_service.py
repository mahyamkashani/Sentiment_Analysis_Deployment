import torch
import os
import boto3
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional

MODEL_DIR = "./fine_tuned_model"


class ModelService:
    """
    Low-level service for model loading and S3 operations.
    Separated from business logic following the service layer pattern.

    This class handles:
    - Downloading models from S3
    - Loading tokenizer and model from disk
    - Managing model lifecycle
    - Device management
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.device: str = "cpu"

    def download_from_s3(self, bucket_name: str, prefix: str):
        """
        Download model files from S3 bucket.

        Args:
            bucket_name: S3 bucket name
            prefix: S3 prefix path to model files
        """
        s3 = boto3.client("s3")
        os.makedirs(self.model_dir, exist_ok=True)

        print(f"Downloading model from s3://{bucket_name}/{prefix}...")

        objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        for obj in objects.get("Contents", []):
            key = obj["Key"]
            filename = key.split("/")[-1]
            if filename:
                local_path = os.path.join(self.model_dir, filename)
                print(f"  Downloading {filename}...")
                s3.download_file(bucket_name, key, local_path)

        print("Model download completed!")

    def load_model(self, device: str = "cpu"):
        """
        Load tokenizer and model from local directory.

        Args:
            device: Device to load model on ("cpu" or "cuda")

        Raises:
            FileNotFoundError: If model directory doesn't exist
        """
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        print(f"Loading model from {self.model_dir}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)

        self.device = device
        self.model.to(device)
        self.model.eval()

        print(f"Model loaded successfully on device: {device}")

    def get_tokenizer(self) -> AutoTokenizer:
        """
        Get loaded tokenizer.

        Returns:
            Loaded tokenizer instance

        Raises:
            RuntimeError: If tokenizer not loaded
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")
        return self.tokenizer

    def get_model(self) -> AutoModelForSequenceClassification:
        """
        Get loaded model.

        Returns:
            Loaded model instance

        Raises:
            RuntimeError: If model not loaded
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model

    def is_loaded(self) -> bool:
        """Check if model and tokenizer are loaded"""
        return self.model is not None and self.tokenizer is not None

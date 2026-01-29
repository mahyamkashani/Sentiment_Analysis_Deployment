"""
TinyBERT Model Training Script

This module provides a class-based approach to training TinyBERT models
for sentiment classification tasks.

Architecture:
    TrainingConfig - Configuration dataclass
    DatasetManager - Dataset loading and preprocessing
    ModelTrainer - Training orchestration

Constants are organized in training_constants.py following the pattern
from softremedy_report's shaya_general_configs.py
"""

import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizer,
    PreTrainedModel
)

# Import constants from organized config file
from training_constants import (
    ModelConfig,
    DatasetConfig,
    TrainingHyperparameters,
    TrainingBehavior,
    OutputConfig,
    HardwareConfig,
    ReportingConfig,
    TokenizationConfig,
    DataProcessingConfig
)


@dataclass
class TrainingConfig:
    """
    Configuration for model training.

    Uses constants from training_constants.py organized by category
    following the pattern from softremedy_report.

    Attributes:
        model_name: Pretrained model name from HuggingFace
        num_labels: Number of classification labels
        output_dir: Directory to save trained model
        max_length: Maximum sequence length for tokenization
        train_samples: Number of training samples to use
        eval_samples: Number of evaluation samples to use
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        logging_steps: Steps between logging
        seed: Random seed for reproducibility
    """
    # Model configuration (from ModelConfig)
    model_name: str = ModelConfig.MODEL_NAME
    num_labels: int = ModelConfig.NUM_LABELS
    max_length: int = ModelConfig.MAX_LENGTH

    # Dataset configuration (from DatasetConfig)
    train_samples: int = DatasetConfig.TRAIN_SAMPLES
    eval_samples: int = DatasetConfig.EVAL_SAMPLES
    seed: int = DatasetConfig.SEED

    # Training hyperparameters (from TrainingHyperparameters)
    batch_size: int = TrainingHyperparameters.BATCH_SIZE
    eval_batch_size: int = TrainingHyperparameters.EVAL_BATCH_SIZE
    num_epochs: int = TrainingHyperparameters.NUM_EPOCHS
    learning_rate: float = TrainingHyperparameters.LEARNING_RATE

    # Output configuration (from OutputConfig)
    output_dir: str = OutputConfig.OUTPUT_DIR
    logging_dir: str = OutputConfig.LOGGING_DIR

    # Training behavior (from TrainingBehavior)
    logging_steps: int = TrainingBehavior.LOGGING_STEPS
    evaluation_strategy: str = TrainingBehavior.EVALUATION_STRATEGY
    save_strategy: str = TrainingBehavior.SAVE_STRATEGY
    load_best_model_at_end: bool = TrainingBehavior.LOAD_BEST_MODEL_AT_END


class DatasetManager:
    """
    Manages dataset loading, preprocessing, and tokenization.

    This class handles all data-related operations including:
    - Loading datasets from HuggingFace
    - Tokenizing text data
    - Preparing data for training
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize DatasetManager.

        Args:
            config: Training configuration
        """
        self.config = config
        self.tokenizer: Optional[PreTrainedTokenizer] = None

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load tokenizer from pretrained model.

        Returns:
            Loaded tokenizer
        """
        print(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        return self.tokenizer

    def tokenize_function(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize a batch of text data.

        Uses constants from TokenizationConfig for consistent behavior.

        Args:
            batch: Batch containing "text" field

        Returns:
            Tokenized batch with input_ids, attention_mask
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")

        return self.tokenizer(
            batch[DataProcessingConfig.TEXT_COLUMN_NAME],
            padding=TokenizationConfig.PADDING,
            truncation=TokenizationConfig.TRUNCATION,
            max_length=self.config.max_length
        )

    def load_and_prepare_dataset(
        self,
        dataset_name: str = DatasetConfig.DATASET_NAME
    ) -> tuple[Dataset, Dataset]:
        """
        Load dataset and prepare for training.

        Args:
            dataset_name: Name of dataset to load from HuggingFace

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        print(f"Loading dataset: {dataset_name}")

        # Load dataset
        dataset = load_dataset(dataset_name)

        # Tokenize dataset
        print("Tokenizing dataset...")
        dataset = dataset.map(self.tokenize_function, batched=True)

        # Rename label column to match trainer expectations
        dataset = dataset.rename_column("label", DataProcessingConfig.LABEL_COLUMN_NAME)

        # Set format to PyTorch tensors
        dataset.set_format("torch", columns=["input_ids", "attention_mask", DataProcessingConfig.LABEL_COLUMN_NAME])

        # Create train and eval splits with shuffling
        print(f"Creating train split ({self.config.train_samples} samples)...")
        train_dataset = dataset["train"]
        if DataProcessingConfig.SHUFFLE_TRAIN:
            train_dataset = train_dataset.shuffle(seed=self.config.seed)
        train_dataset = train_dataset.select(range(self.config.train_samples))

        print(f"Creating eval split ({self.config.eval_samples} samples)...")
        eval_dataset = dataset["test"]
        if DataProcessingConfig.SHUFFLE_EVAL:
            eval_dataset = eval_dataset.shuffle(seed=self.config.seed)
        eval_dataset = eval_dataset.select(range(self.config.eval_samples))

        return train_dataset, eval_dataset


class ModelTrainer:
    """
    Orchestrates the model training process.

    This class handles:
    - Model initialization
    - Training configuration
    - Training execution
    - Model saving
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize ModelTrainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.trainer: Optional[Trainer] = None

    def load_model(self) -> PreTrainedModel:
        """
        Load pretrained model for fine-tuning.

        Returns:
            Loaded model
        """
        print(f"Loading model: {self.config.model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels
        )
        return self.model

    def create_training_arguments(self) -> TrainingArguments:
        """
        Create training arguments for Trainer.

        Uses constants from training_constants.py for consistent behavior.

        Returns:
            Training arguments
        """
        return TrainingArguments(
            output_dir=self.config.output_dir,
            logging_dir=self.config.logging_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            logging_steps=self.config.logging_steps,
            load_best_model_at_end=self.config.load_best_model_at_end,
            fp16=torch.cuda.is_available() if HardwareConfig.FP16 is None else HardwareConfig.FP16,
            seed=self.config.seed,
            report_to=ReportingConfig.REPORT_TO,
            disable_tqdm=ReportingConfig.DISABLE_TQDM,
            logging_first_step=ReportingConfig.LOGGING_FIRST_STEP,
            # Additional constants from config classes
            dataloader_num_workers=HardwareConfig.DATALOADER_NUM_WORKERS,
            dataloader_pin_memory=HardwareConfig.DATALOADER_PIN_MEMORY,
            no_cuda=HardwareConfig.NO_CUDA
        )

    def setup_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        tokenizer: PreTrainedTokenizer
    ):
        """
        Setup Hugging Face Trainer.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer for model
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        self.tokenizer = tokenizer
        training_args = self.create_training_arguments()

        print("Setting up Trainer...")
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer
        )

    def train(self):
        """
        Execute training.
        """
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_trainer() first.")

        print("=" * 60)
        print("Starting training...")
        print("=" * 60)

        self.trainer.train()

        print("=" * 60)
        print("Training complete!")
        print("=" * 60)

    def save_model(self):
        """
        Save trained model and tokenizer.
        """
        if self.trainer is None or self.tokenizer is None:
            raise ValueError("Trainer and tokenizer must be initialized before saving.")

        print(f"Saving model to: {self.config.output_dir}")
        self.trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        print("Model saved successfully!")

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate trained model on eval dataset.

        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_trainer() first.")

        print("Evaluating model...")
        metrics = self.trainer.evaluate()

        print("Evaluation results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        return metrics


class TinyBERTTrainingPipeline:
    """
    Complete training pipeline orchestrating all components.

    This is the main class that coordinates:
    - Configuration
    - Dataset preparation
    - Model training
    - Model saving
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize training pipeline.

        Args:
            config: Training configuration (uses default if None)
        """
        self.config = config or TrainingConfig()
        self.dataset_manager = DatasetManager(self.config)
        self.model_trainer = ModelTrainer(self.config)

    def run(self, dataset_name: str = DatasetConfig.DATASET_NAME):
        """
        Run complete training pipeline.

        Uses constants from training_constants.py for default values.

        Args:
            dataset_name: Dataset to use for training (default from DatasetConfig)
        """
        print("=" * 60)
        print("TinyBERT Training Pipeline")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Model: {self.config.model_name}")
        print(f"  Dataset: {dataset_name}")
        print(f"  Train samples: {self.config.train_samples}")
        print(f"  Eval samples: {self.config.eval_samples}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Output: {self.config.output_dir}")
        print("=" * 60)
        print()

        # Step 1: Load tokenizer
        tokenizer = self.dataset_manager.load_tokenizer()

        # Step 2: Prepare dataset
        train_dataset, eval_dataset = self.dataset_manager.load_and_prepare_dataset(
            dataset_name
        )

        # Step 3: Load model
        self.model_trainer.load_model()

        # Step 4: Setup trainer
        self.model_trainer.setup_trainer(train_dataset, eval_dataset, tokenizer)

        # Step 5: Train
        self.model_trainer.train()

        # Step 6: Evaluate
        self.model_trainer.evaluate()

        # Step 7: Save
        self.model_trainer.save_model()

        print()
        print("=" * 60)
        print("Pipeline complete!")
        print("=" * 60)


def main():
    """
    Main entry point for training script.

    This function can be called directly or used as CLI entry point.
    """
    # Create custom configuration if needed
    config = TrainingConfig(
        model_name="huawei-noah/TinyBERT_General_4L_312D",
        num_labels=2,
        output_dir="./fine_tuned_model",
        max_length=128,
        train_samples=2000,
        eval_samples=500,
        batch_size=16,
        num_epochs=2,
        seed=42
    )

    # Create and run pipeline
    pipeline = TinyBERTTrainingPipeline(config)
    pipeline.run(dataset_name="imdb")


if __name__ == "__main__":
    main()

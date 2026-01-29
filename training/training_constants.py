"""
Training Constants Configuration

This module organizes all training-related constants into clustered classes
following the pattern from softremedy_report's shaya_general_configs.py

Each class groups related constants by domain/category.
"""


class ModelConfig:
    """Model-related constants"""
    MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
    NUM_LABELS = 2
    MAX_LENGTH = 128


class DatasetConfig:
    """Dataset-related constants"""
    DATASET_NAME = "imdb"
    TRAIN_SAMPLES = 2000
    EVAL_SAMPLES = 500
    SEED = 42


class TrainingHyperparameters:
    """Training hyperparameters"""
    BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 16
    NUM_EPOCHS = 8
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.0
    WARMUP_STEPS = 0


class TrainingBehavior:
    """Training behavior and strategy constants"""
    EVALUATION_STRATEGY = "epoch"
    SAVE_STRATEGY = "epoch"
    LOGGING_STEPS = 50
    LOAD_BEST_MODEL_AT_END = False
    SAVE_TOTAL_LIMIT = 2
    METRIC_FOR_BEST_MODEL = None
    GREATER_IS_BETTER = None


class OutputConfig:
    """Output and paths configuration"""
    OUTPUT_DIR = "./fine_tuned_model"
    LOGGING_DIR = "./logs"
    OVERWRITE_OUTPUT_DIR = True


class HardwareConfig:
    """Hardware and performance configuration"""
    FP16 = None  # Will be set to torch.cuda.is_available() at runtime
    DATALOADER_NUM_WORKERS = 0
    DATALOADER_PIN_MEMORY = True
    NO_CUDA = False


class ReportingConfig:
    """Reporting and tracking configuration"""
    REPORT_TO = "tensorboard"  # Options: "tensorboard", "wandb", "none"
    # Note: To use "tensorboard", install: pip install tensorboard
    # Note: To use "wandb", install: pip install wandb
    DISABLE_TQDM = False
    LOGGING_FIRST_STEP = True


class OptimizationConfig:
    """Optimization-related constants"""
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1.0
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.999
    ADAM_EPSILON = 1e-8


class TokenizationConfig:
    """Tokenization-related constants"""
    PADDING = "max_length"
    TRUNCATION = True
    RETURN_TENSORS = "pt"


class DataProcessingConfig:
    """Data processing constants"""
    SHUFFLE_TRAIN = True
    SHUFFLE_EVAL = True
    LABEL_COLUMN_NAME = "labels"
    TEXT_COLUMN_NAME = "text"


# Quick access to commonly used constants
class DefaultTrainingConfig:
    """
    Default training configuration aggregating all constant classes.
    Use this for quick access to all defaults.
    """

    # Model
    MODEL_NAME = ModelConfig.MODEL_NAME
    NUM_LABELS = ModelConfig.NUM_LABELS
    MAX_LENGTH = ModelConfig.MAX_LENGTH

    # Dataset
    DATASET_NAME = DatasetConfig.DATASET_NAME
    TRAIN_SAMPLES = DatasetConfig.TRAIN_SAMPLES
    EVAL_SAMPLES = DatasetConfig.EVAL_SAMPLES
    SEED = DatasetConfig.SEED

    # Training
    BATCH_SIZE = TrainingHyperparameters.BATCH_SIZE
    EVAL_BATCH_SIZE = TrainingHyperparameters.EVAL_BATCH_SIZE
    NUM_EPOCHS = TrainingHyperparameters.NUM_EPOCHS
    LEARNING_RATE = TrainingHyperparameters.LEARNING_RATE

    # Output
    OUTPUT_DIR = OutputConfig.OUTPUT_DIR
    LOGGING_DIR = OutputConfig.LOGGING_DIR


    # Behavior
    LOGGING_STEPS = TrainingBehavior.LOGGING_STEPS
    EVALUATION_STRATEGY = TrainingBehavior.EVALUATION_STRATEGY
    SAVE_STRATEGY = TrainingBehavior.SAVE_STRATEGY

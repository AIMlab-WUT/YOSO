import yaml
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from utils import mse_metric, ssim_metric

def _assert(condition, message):
    if not condition:
        raise ValueError(f"[CONFIG ERROR] {message}")

def load_config(config_filepath):
    """
    Load and validate experiment configuration from a YAML file.

    Args:
        config_filepath (str): Path to YAML configuration file.

    Returns:
        dict: Fully validated and processed configuration dictionary.
    """
    # Load YAML configuration
    with open(config_filepath, "r") as f:
        config = yaml.safe_load(f)

    # Check required sections
    required_sections = ["data", "model", "training"]
    for section in required_sections:
        _assert(section in config, f"Missing section: '{section}'")

    # DATA CONFIGURATION
    data_cfg = config["data"]
    _assert(0 < data_cfg["validation_split"] < 1, "validation_split must be in (0,1)")
    _assert(0 < data_cfg["test_split"] < 1, "test_split must be in (0,1)")
    _assert(data_cfg["batch_size"] > 0, "batch_size must be > 0")

    _assert(
        data_cfg["validation_split"] + data_cfg["test_split"] < 1,
        "validation_split + test_split must be < 1"
    )

    # MODEL CONFIGURATION
    model_cfg = config["model"]

    _assert(model_cfg["num_resnet_blocks"] > 0, "num_resnet_blocks must be > 0")
    _assert(model_cfg["init_num_feature_maps"] > 0, "init_num_feature_maps must be > 0")

    # initializer
    if model_cfg["initializer"] == "he_normal":
        model_cfg["initializer"] = tf.keras.initializers.HeNormal()
    else:
        raise ValueError(f"Unsupported initializer: {model_cfg['initializer']}")

    # input shape
    if "input_shape" in model_cfg:
        shape = model_cfg["input_shape"]
        _assert(isinstance(shape, list), "input_shape must be a list")
        _assert(len(shape) == 3, "input_shape must have 3 dimensions")

    # TRAINING CONFIGURATION
    train_cfg = config["training"]

    _assert(train_cfg["epochs"] > 0, "epochs must be > 0")

    # optimizer
    _assert("optimizer" in train_cfg, "Missing 'optimizer' in training")

    opt_cfg = train_cfg["optimizer"]
    _assert(opt_cfg["type"] == "adam", "Only 'adam' optimizer supported")

    lr = opt_cfg["learning_rate"]
    _assert(lr > 0, "learning_rate must be > 0")

    # scheduler
    if "decay" in opt_cfg:
        decay_cfg = opt_cfg["decay"]

        if decay_cfg["type"] == "exponential":
            _assert(decay_cfg["decay_steps"] > 0, "decay_steps must be > 0")
            _assert(0 < decay_cfg["decay_rate"] <= 1, "decay_rate must be in (0,1]")

            lr_schedule = ExponentialDecay(
                lr,
                decay_steps=decay_cfg["decay_steps"],
                decay_rate=decay_cfg["decay_rate"]
            )
        else:
            raise ValueError(f"Unsupported decay type: {decay_cfg['type']}")
    else:
        lr_schedule = lr

    train_cfg["optimizer"] = Adam(learning_rate=lr_schedule)

    # loss
    _assert("loss" in train_cfg, "Missing 'loss' in training")

    if train_cfg["loss"]["type"] == "mse":
        train_cfg["loss"] = tf.keras.losses.MeanSquaredError()
    else:
        raise ValueError(f"Unsupported loss: {train_cfg['loss']['type']}")

    # metrics
    _assert("metrics" in train_cfg, "Missing 'metrics' in training")

    allowed_metrics = {"mse", "ssim"}
    metrics_map = {
        "mse": mse_metric,
        "ssim": ssim_metric,
    }

    metrics = []
    for m in train_cfg["metrics"]:
        _assert(m in allowed_metrics, f"Unsupported metric: {m}")
        metrics.append(metrics_map[m])

    train_cfg["metrics"] = metrics

    return config
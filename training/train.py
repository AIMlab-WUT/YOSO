from config import load_config
from data import load_dataset, preprocess_dataset, SaveTrainingData, load_training_data
from model import build_resnet_model

import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import gc
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np

def setup_environment():
    """
    Configure TensorFlow execution environment for reproducible and stable training.
    """
    # Reduce TensorFlow logging verbosity for cleaner output
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    # Use asynchronous CUDA memory allocator for improved GPU performance
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    # Clear previous TensorFlow graph/session (prevents memory accumulation)
    tf.keras.backend.clear_session()
    # Force Python garbage collection to free unused memory
    gc.collect()

    # Detect available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print("GPU name:", tf.config.experimental.get_device_details(gpus[0])['device_name'])
         # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
        tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        print("No GPU detected.")

def init_model(model, training_config, metrics=False):
    """
    Compile a Keras model using training configuration.

    This function applies the loss function, optimizer, and optionally
    evaluation metrics defined in the configuration dictionary.

    Args:
        model (tf.keras.Model): Uncompiled Keras model.
        training_config (dict): Dictionary containing training setup:
            - loss: loss function
            - optimizer: optimizer instance
            - metrics: list of evaluation metrics
        metrics (bool): If True, metrics are logging during training.

    Returns:
        tf.keras.Model: Compiled Keras model ready for training.
    """
    loss = training_config["loss"]
    optimizer = training_config["optimizer"]
    metrics = training_config["metrics"]

    if metrics:
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    else:
        model.compile(loss=loss, optimizer=optimizer)

    return model

def execute_training(model, train_dataset, val_dataset, training_config, save_dir):
    """
    Execute the full training pipeline for a Keras model.

    Args:
        model (tf.keras.Model): Uncompiled Keras model.
        train_dataset (tf.data.Dataset): Training dataset.
        val_dataset (tf.data.Dataset): Validation dataset.
        training_config (dict): Training configuration (epochs, loss, optimizer, etc.).
        save_dir (str): Directory where outputs (models, logs, plots) are saved.

    Returns:
        tf.keras.callbacks.History: Training history object returned by model.fit().
    """
    # Compile model with optimizer, loss, and metrics
    model = init_model(model, training_config)

    # Save best model based on validation loss
    best_model_checkpoint = ModelCheckpoint(
        f"{save_dir}/best_model.keras",
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    callbacks = [
        best_model_checkpoint,
        SaveTrainingData(dir_name=save_dir),
        TensorBoard("logs", histogram_freq=1),
    ]

    # Determine number of steps per epoch from dataset cardinality
    steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
    val_steps_per_epoch = tf.data.experimental.cardinality(val_dataset).numpy()

    # Model training
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=training_config["epochs"],
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps_per_epoch,
    )
    
    # Save model after all epochs
    model.save(f"{save_dir}/final_model.keras")

    # Generate trainin plot and summary
    show_training_progress(f"{save_dir}/training_data.pkl", save_dir, history, steps_per_epoch)

def show_training_progress(training_data_file, save_dir, history, steps_per_epoch):
    """
    Visualize training progress using batch-level and epoch-level loss curves.
    The function loads saved training statistics and generates a combined plot
    showing:
    - Batch-level loss evolution
    - Epoch-level training loss
    - Epoch-level validation loss

    Args:
        training_data_file (str): Path to saved training statistics (pickle file).
        save_dir (str): Directory where the plot will be saved.
        history (tf.keras.callbacks.History): Keras training history.
        steps_per_epoch (int): Number of training steps per epoch.
    """
    # Load previously saved training statistics
    training_data = load_training_data(training_data_file)

    loss = training_data['epoch_history']['loss']
    val_loss = training_data['epoch_history']['val_loss']
    batch_losses = training_data['batch_losses']

    epochs = range(1, len(loss) + 1)

    fig, ax1 = plt.subplots(figsize=(15, 8))

    # Batch-level loss curve
    if batch_losses:
        # Downsample batch losses for readability
        display_every = max(1, len(batch_losses) // 1000)
        batch_indices = range(0, len(batch_losses), display_every)
        displayed_batches = [batch_losses[i] for i in batch_indices]

        ax1.plot(batch_indices, np.log(displayed_batches), 'b-',
                alpha=0.5, linewidth=0.5, label='Batch Loss')

        print(f"Displaying {len(displayed_batches)} batch points out of {len(batch_losses)} total")

    # Epoch-level training loss
    if epochs and loss:
        iteration_points = np.array(epochs) * steps_per_epoch
        ax1.plot(iteration_points, np.log(loss), 'ro--',
                label='Train Loss', markersize=6, linewidth=2, alpha=0.5)

    # Epoch-level validation loss
    if epochs and val_loss:
        iteration_points = np.array(epochs) * steps_per_epoch
        ax1.plot(iteration_points, np.log(val_loss), 'go--',
                label='Val Loss', markersize=6, linewidth=2, alpha=0.5)

    ax1.set_xlabel('Batch Number')
    ax1.set_ylabel('Log Loss')

    # Secondary x-axis showing epoch numbers
    ax2 = ax1.twiny()
    if epochs:
        ax2.set_xlim(ax1.get_xlim())
        epoch_ticks = np.array(epochs) * steps_per_epoch
        ax2.set_xticks(epoch_ticks)
        ax2.set_xticklabels(epochs)
        ax2.set_xlabel('Epoch Number')

    ax1.set_title('Training progress - batch losses & epoch metrics')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Save visualization to file
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_plot.png", dpi=150, bbox_inches='tight')

    # Console summary
    print(f"Training summary:")
    if batch_losses:
        print(f"   - Batch loss: {len(batch_losses)} batches recorded")
        print(f"   - Final batch loss: {batch_losses[-1]:.4f}")
    if loss:
        print(f"   - Epoch loss: {len(loss)} epochs recorded")
        print(f"   - Final train loss: {loss[-1]:.4f}, val loss: {val_loss[-1]:.4f}")

if __name__ == "__main__":
    """
    Main entry point for training a ResNet-based model.

    This script:
    1. Parses command-line arguments
    2. Configures the execution environment (GPU, TensorFlow settings)
    3. Loads and validates configuration file
    4. Loads dataset from disk
    5. Preprocesses dataset into train/val/test splits
    6. Builds the model architecture
    7. Executes full training pipeline
    """
    parser = argparse.ArgumentParser()
    # Path to YAML configuration file defining experiment setup
    parser.add_argument("--config_file", default="configs/config.yaml")
    # Path to dataset file (e.g., HDF5)
    parser.add_argument("--data_file")
    # Directory where all outputs (models, logs, plots) will be saved
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    # Setup environment
    setup_environment()
    # Load experiment configuration
    config = load_config(args.config_file)
    # Load dataset
    dataset = load_dataset(args.data_file)
    # Preprocess data, splitting on training, validation and test sets
    train_ds, val_ds, test_ds = preprocess_dataset(dataset, config["data"])
    
    # Generate Keras model
    model = build_resnet_model(config["model"])
    # Execute model training
    execute_training(model, train_ds, val_ds, config["training"], args.output_dir)
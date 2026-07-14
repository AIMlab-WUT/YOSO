import tensorflow as tf
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
import pickle

def _create_dataset(x_data, y_data, batch_size, shuffle=False):
    """
    Create a TensorFlow dataset with batching and optional shuffling.

    Args:
        x_data (np.ndarray): Input data.
        y_data (np.ndarray): Target data.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        tf.data.Dataset: Prepared dataset with batching and prefetching.
    """
    # Create dataset from input and output data
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    # Group samples into batches
    dataset = dataset.batch(batch_size)
    # Prefetch batches to overlap data loading and model execution
    return dataset.prefetch(tf.data.AUTOTUNE)

def load_dataset(dataset_file):
    """
    Load dataset from an HDF5 file and validate its structure.
    The function ensures that the required datasets ("inputs" and "targets")
    are present in the file and returns them in a dictionary-like format.

    Args:
        dataset_file (str): Path to the HDF5 dataset file.

    Returns:
        dict: Dictionary containing:
            - "inputs": HDF5 dataset with input samples
            - "targets": HDF5 dataset with corresponding targets

    Raises:
        RuntimeError: If the file cannot be loaded or required datasets are missing.
    """
    try:
        h5f = h5py.File(dataset_file, "r")

        # Validate required dataset keys
        if 'inputs' not in h5f:
            raise KeyError("Missing 'inputs' dataset in the file.")
        if 'targets' not in h5f:
            raise KeyError("Missing 'targets' dataset in the file.")
        
        dataset = {
            "inputs": h5f['inputs'],
            "targets": h5f['targets']
        }

        print(f"Loaded dataset with {dataset['inputs'].shape[0]} samples.")
        return dataset
    except Exception as e:
        raise RuntimeError(f"Dataset loading error: {e}")

def preprocess_dataset(dataset, data_config, seed=42):
    """
    Split dataset into training, validation, and test sets and convert them
    into TensorFlow datasets.

    Args:
        dataset (h5py.File): Dataset containing "inputs" and "targets".
        data_config (dict): Configuration dictionary with keys:
            - batch_size (int): Number of samples per batch
            - validation_split (float): Fraction of data for validation
            - test_split (float): Fraction of data for testing
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple:
            train_dataset (tf.data.Dataset): Training dataset
            val_dataset (tf.data.Dataset): Validation dataset
            test_dataset (tf.data.Dataset): Test dataset
    """
    # Load data from dataset object into NumPy arrays
    inputs = np.array(dataset["inputs"][:])
    targets = np.array(dataset["targets"][:])
    batch_size = data_config["batch_size"]
    validation_split = data_config["validation_split"]
    test_split = data_config["test_split"]

    # First split: training vs (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        inputs, targets, test_size=(validation_split + test_split), random_state=seed
    )
    # Second split: validation vs test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_split/(validation_split + test_split), random_state=seed
    )

    # Convert NumPy arrays into tf.data pipelines
    train_dataset = _create_dataset(X_train, y_train, batch_size, True)
    val_dataset = _create_dataset(X_val, y_val, batch_size)
    test_dataset = _create_dataset(X_test, y_test, batch_size)

    print(f"Dataset sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return train_dataset, val_dataset, test_dataset

class SaveTrainingData(Callback):
    """
    Keras callback for tracking and saving training statistics in file training_data.pkl.
    This callback collects:
    - Batch-level losses during training
    - Epoch-level training and validation losses
    - Metadata such as total number of epochs and batches
    The collected data is periodically saved to a pickle file after each epoch,
    enabling post-training analysis and visualization.

    Args:
        dir_name (str): Directory where training data will be stored.
    """
    def __init__(self, dir_name):
        super().__init__()
        self.dir_name = dir_name
        self.data_file = f"{dir_name}/training_data.pkl"

    def on_train_begin(self, logs=None):
        """
        Initialize storage structures at the beginning of training.
        """
        self.training_data = {
            'epoch_history': {'loss': [], 'val_loss': []},
            'batch_losses': [],
            'metadata': {'total_epochs': 0, 'total_batches': 0}
        }
        print("Starting new training data")

    def on_batch_end(self, batch, logs=None):
        """
        Record loss after each batch.

        Args:
            batch (int): Batch index
            logs (dict): Dictionary containing batch metrics
        """
        if logs and 'loss' in logs:
            self.training_data['batch_losses'].append(logs['loss'])
            # Update batch counter
            self.training_data['metadata']['total_batches'] = len(self.training_data['batch_losses'])

    def on_epoch_end(self, epoch, logs=None):
        """
        Record epoch-level metrics and persist data to disk.

        Args:
            epoch (int): Current epoch index
            logs (dict): Dictionary containing epoch metrics
        """
        self.training_data['epoch_history']['loss'].append(logs.get('loss'))
        self.training_data['epoch_history']['val_loss'].append(logs.get('val_loss'))
        self.training_data['metadata']['total_epochs'] = len(self.training_data['epoch_history']['loss'])

        # Save data to file after each epoch
        try:
            with open(self.data_file, "wb") as f:
                pickle.dump(self.training_data, f)
            print(f"Saved training data after epoch {epoch+1} "
                  f"(batches: {len(self.training_data['batch_losses'])}, "
                  f"epochs: {len(self.training_data['epoch_history']['loss'])})")
        except Exception as e:
            print(f"Error saving training data: {e}")

def load_training_data(data_file):
    """
    Load previously saved training statistics from a pickle file.
    If the file does not exist, an empty training structure is returned.

    Args:
        data_file (str): Path to the serialized training data file.

    Returns:
        dict: Dictionary containing:
            - epoch_history (dict): Lists of epoch-level metrics ("loss", "val_loss")
            - batch_losses (list): List of batch-level loss values
            - metadata (dict): Training metadata (total epochs and batches)
    """
    try:
        with open(data_file, "rb") as f:
            training_data = pickle.load(f)
        print(f"Loaded training data: {len(training_data['epoch_history']['loss'])} epochs, {len(training_data.get('batch_losses', []))} batches")
        return training_data
    except FileNotFoundError:
        print("No training data found")
        return {
            'epoch_history': {'loss': [], 'val_loss': []},
            'batch_losses': [],
            'metadata': {'total_epochs': 0, 'total_batches': 0}
        }
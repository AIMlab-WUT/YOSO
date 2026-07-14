import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Conv2D, Input, BatchNormalization, Activation, MaxPool2D, UpSampling2D

def _residual_block(inputs, filters, initializer):
    """
    Define a custom residual block with multiple skip connections.

    This block extends the standard ResNet idea by introducing several
    intermediate skip connections (skip_1, skip_2, skip_3), which help
    improve gradient flow and stabilize training in deeper architectures.

    Args:
        inputs (tf.Tensor): Input tensor.
        filters (int): Number of convolutional filters.
        initializer (tf.keras.initializers.Initializer): Kernel initializer.

    Returns:
        tf.Tensor: Output tensor after applying the residual block.
    """
    skip_1 = inputs
    x = Conv2D(filters, (3, 3), padding="same", kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)

    skip_2 = x
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)

    skip_3 = x
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)

    x = Add()([x, skip_1])
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)

    x = Add()([x, skip_2])
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)

    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)

    x = Add()([x, skip_3])
    x = Activation("relu")(x)
    
    return x

def _resnet_path(inputs, scale, initializer, num_blocks, num_filters):
    """
    Define a single ResNet processing path with optional spatial scaling.

    This function builds one branch of a multi-scale architecture. Depending
    on the scale factor, the input is optionally downsampled, processed through
    a series of residual blocks, and then upsampled back to the original resolution.

    Args:
        inputs (tf.Tensor): Input tensor.
        scale (int): Downsampling/upsampling factor (e.g., 1, 2, 4, 8).
        initializer (tf.keras.initializers.Initializer): Kernel initializer.
        num_blocks (int): Number of residual blocks in this path.
        num_filters (int): Number of convolutional filters.

    Returns:
        tf.Tensor: Output tensor after multi-scale processing.
    """
    x = inputs
    
    # Downsample input if scale > 1 to process lower-resolution features
    if scale > 1:
        x = MaxPool2D(pool_size=(3, 3), strides=scale, padding="same")(x)
    
    # Apply a sequence of residual blocks
    for _ in range(num_blocks):
        x = _residual_block(x, num_filters, initializer)

    # Upsample back to original resolution if downsampling was applied
    if scale > 1:
        x = UpSampling2D(size=scale)(x)
    
    x = Conv2D(num_filters, (3, 3), padding="same", kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    
    return Activation("relu")(x)

def build_resnet_model(model_config):
    """
    Build a multi-scale residual neural network.
    The model is based on parallel residual paths operating at different
    spatial scales (1, 2, 4, 8), whose outputs are merged via summation.

    Args:
        model_config (dict): Configuration dictionary containing:
            - initializer (tf.keras.initializers.Initializer): weight initializer
            - num_resnet_blocks (int): number of residual blocks per path
            - init_num_feature_maps (int): number of base convolutional filters
            - input_shape (tuple): shape of model input

    Returns:
        tf.keras.Model: Compiled computational graph (model architecture)
    """
    initializer = model_config["initializer"]
    num_blocks = model_config["num_resnet_blocks"]
    num_features = model_config["init_num_feature_maps"]

    inputs = Input(shape=model_config["input_shape"])

    # Initial feature extraction layer
    x = Conv2D(num_features, (3, 3), padding="same", kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Multi-scale residual feature extraction paths
    x1 = _resnet_path(x, 1, initializer, num_blocks, num_features)
    x2 = _resnet_path(x, 2, initializer, num_blocks, num_features)
    x3 = _resnet_path(x, 4, initializer, num_blocks, num_features)
    x4 = _resnet_path(x, 8, initializer, num_blocks, num_features)

    # Fuse multi-scale features via element-wise summation
    x = Add()([x1, x2, x3, x4])
    
    # Final projection to output channel space
    x = Conv2D(1, (1, 1), padding="same", kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    
    outputs = Activation("relu")(x)

    model = Model(inputs=inputs, outputs=outputs, name="ResNetModel")
    return model
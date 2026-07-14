import tensorflow as tf

def mse_metric(y_true, y_pred):
    """
    Compute Mean Squared Error (MSE) between ground truth and predictions.

    Args:
        y_true (tf.Tensor): Ground truth images.
        y_pred (tf.Tensor): Predicted images.

    Returns:
        tf.Tensor: Scalar tensor representing MSE.
    """
    mse_loss = tf.keras.losses.MeanSquaredError()
    return mse_loss(y_true, y_pred)

def ssim_metric(y_true, y_pred):
    """
    Compute Structural Similarity Index (SSIM) between ground truth and predictions.

    Args:
        y_true (tf.Tensor): Ground truth images.
        y_pred (tf.Tensor): Predicted images.

    Returns:
        tf.Tensor: Scalar tensor representing 1 - SSIM score (range: 0 = perfect match, higher = worse similarity).
    """
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    return 1.0 - ssim
import tensorflow as tf
import numpy as np
from typing import Optional


class DNN:
    """
    Wrapper for TensorFlow CNN inference on holograms
    with automatic padding to multiples of 8.
    """

    def __init__(self, model_path: str) -> None:
        self.model = tf.keras.models.load_model(model_path)

        self.pad_h: Optional[int] = None
        self.pad_w: Optional[int] = None

    # -----------------------------
    # PADDING
    # -----------------------------
    def pad_to_multiple_of_8(self, img: np.ndarray) -> np.ndarray:
        """
        Pad image so spatial dimensions are multiples of 8 (edge padding).
        """

        h, w = img.shape[:2]

        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8

        padding = [(0, pad_h), (0, pad_w)]

        if img.ndim == 3:
            padding.append((0, 0))

        padded = np.pad(img, padding, mode="edge")

        self.pad_h = pad_h
        self.pad_w = pad_w

        return padded

    def remove_padding(self, img: np.ndarray) -> np.ndarray:
        """
        Remove padding added by `pad_to_multiple_of_8`.
        """

        if self.pad_h is None or self.pad_w is None:
            raise ValueError("Padding not set. Call pad_to_multiple_of_8 first.")

        h_end = img.shape[0] - self.pad_h if self.pad_h > 0 else img.shape[0]
        w_end = img.shape[1] - self.pad_w if self.pad_w > 0 else img.shape[1]

        return img[:h_end, :w_end]


    # -----------------------------
    # INFERENCE
    # -----------------------------
    def infer(self, holo: np.ndarray) -> np.ndarray:
        """
        Run CNN inference on a hologram.
        """

        holo = holo / (np.median(holo) + 1e-12)

        padded = self.pad_to_multiple_of_8(holo)

        pred = self.model.predict(
            padded[np.newaxis, :, :, np.newaxis],
            verbose=0
        )[0, :, :, 0]

        # fill zeros safely
        # mask = pred != 0
        # fill_value = pred[mask].mean() if np.any(mask) else 0.0
        # pred[~mask] = fill_value

        # fill zeros safely
        non_zero_mean = pred[pred != 0].mean()
        pred[pred == 0] = non_zero_mean

        return self.remove_padding(pred)

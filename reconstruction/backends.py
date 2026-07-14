import tensorflow as tf
import numpy as np


class Backend:
    """Base backend class"""
    def fft2(self, x): raise NotImplementedError
    def ifft2(self, x): raise NotImplementedError
    def exp(self, x): raise NotImplementedError
    def sqrt(self, x): raise NotImplementedError
    def angle(self, x): raise NotImplementedError
    def abs(self, x): raise NotImplementedError
    def conj(self, x): raise NotImplementedError
    def expj(self, phase): raise NotImplementedError
    def to_complex(self, x): raise NotImplementedError
    def zeros_like(self, x): raise NotImplementedError


class NumpyBackend:
    """
    Backend using NumPy
    """
    def __init__(self):
        self.dtype = np.complex64
    def fft2(self, x): return np.fft.fft2(x)
    def ifft2(self, x): return np.fft.ifft2(x)
    def exp(self, x): return np.exp(x)
    def sqrt(self, x): return np.sqrt(x)
    def angle(self, x): return np.angle(x)
    def abs(self, x): return np.abs(x)
    def conj(self, x): return np.conj(x)
    def expj(self, phase): return np.exp(-1j * phase)
    def to_complex(self, x): return x.astype(self.dtype)
    def zeros_like(self, x): return np.zeros_like(x, dtype=self.dtype)


class TFBackend:
    """
    Backend using TesnorFlow (approx. 10fold speedup)
    """
    def __init__(self):
        self.dtype = tf.complex64
    def fft2(self, x): return tf.signal.fft2d(x)
    def ifft2(self, x): return tf.signal.ifft2d(x)
    def exp(self, x): return tf.exp(x)
    def sqrt(self, x): return tf.sqrt(x)
    def angle(self, x): return tf.math.angle(x)
    def abs(self, x): return tf.abs(x)
    def conj(self, x): return tf.math.conj(x)
    def expj(self, phase):
        phase = tf.cast(phase, self.dtype)
        return tf.exp(-1j * phase)
    def to_complex(self, x): return tf.cast(x, self.dtype)
    def zeros_like(self, x): return tf.zeros_like(x, self.dtype)

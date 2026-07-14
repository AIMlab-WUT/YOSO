import tensorflow as tf
import numpy as np
from typing import Optional, Tuple, Any


class FieldRetriever:
    """
    Holographic field reconstruction using Gerchberg–Saxton and Gabor methods
    with interchangeable NumPy / TensorFlow backends.
    """

    def __init__(
        self,
        holos_shape: Tuple[int, int, int],
        distances: Tuple[float, float],
        wavelength: float,
        sampling: float,
        backend: Any,
        ri: float = 1.0,
    ) -> None:
        self.backend = backend
        self.amps: Optional[Any] = None

        ny, nx, n_holos = holos_shape
        assert n_holos == 2
        assert len(distances) == 2

        self.shape: Tuple[int, int] = (ny, nx)
        self.dist: Tuple[float, float] = distances
        self.wavelength: float = wavelength
        self.sampling: float = sampling
        self.ri: float = ri

        self.tf_gs = self.free_space_transfer_function(self.dist[1] - self.dist[0])
        self.tf_z1 = self.free_space_transfer_function(self.dist[0])

    def set(self, holos: np.ndarray) -> None:
        """Set hologram amplitudes from intensity data."""
        b = self.backend
        self.amps = b.to_complex(b.sqrt(holos))

    def gerchberg_saxton(self, max_iter: int = 25) -> Any:
        """Reconstruct field using Gerchberg–Saxton algorithm."""
        self._require_amps()
        b = self.backend

        field = b.to_complex(self.amps[:, :, 0])

        for _ in range(max_iter):
            field = b.ifft2(b.fft2(field) * self.tf_gs)

            field = self.amps[:, :, 1] * b.expj(b.angle(field))

            field = b.ifft2(b.fft2(field) * self.tf_gs)

            field = self.amps[:, :, 0] * b.expj(b.angle(field))

        field = b.conj(b.ifft2(b.fft2(b.conj(field)) * self.tf_z1))
        return field

    def gabor(self) -> Any:
        """Reconstruct field using Gabor back-propagation."""
        self._require_amps()
        b = self.backend

        field = b.fft2(self.amps[:, :, 0]) * self.tf_z1
        field = b.conj(b.ifft2(field))
        return field

    def free_space_transfer_function(self, prop_dist: float) -> Any:
        """Compute free-space propagation transfer function."""
        if self.backend.__class__.__name__ == "NumpyBackend":
            return self._tf_numpy(prop_dist)
        else:
            return self._tf_tensorflow(prop_dist)

    def _tf_numpy(self, prop_dist: float) -> np.ndarray:
        """NumPy implementation of transfer function."""
        k = 2 * np.pi / self.wavelength

        fx = np.fft.fftfreq(self.shape[1], d=self.sampling)
        fy = np.fft.fftfreq(self.shape[0], d=self.sampling)

        fx2 = fx[None, :]
        fy2 = fy[:, None]

        freq_sq = fx2**2 + fy2**2
        arg = self.ri**2 - (self.wavelength**2) * freq_sq

        mask = arg >= 0
        trans_fun = np.zeros_like(freq_sq, dtype=np.complex64)

        phase = np.sqrt(arg[mask])
        trans_fun[mask] = np.exp(1j * (k * prop_dist * phase - k * prop_dist * self.ri))

        return trans_fun

    def _tf_tensorflow(self, prop_dist: float) -> tf.Tensor:
        """TensorFlow wrapper around NumPy transfer function."""
        tf_arr = self._tf_numpy(prop_dist)
        return tf.convert_to_tensor(tf_arr, dtype=self.backend.dtype)

    def _require_amps(self) -> None:
        """Ensure amplitudes are initialized before reconstruction."""
        if self.amps is None:
            raise ValueError("'amps' must be initialized before reconstruction.")
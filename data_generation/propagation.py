import numpy as np

_kernel_cache = {}


def _get_kernel(nx, ny, pixel_size, wavelength, n0, z):
    key = (nx, ny, float(pixel_size), float(wavelength), float(n0), float(z))
    if key in _kernel_cache:
        return _kernel_cache[key]

    k = 2 * np.pi / wavelength

    fx = np.fft.fftfreq(nx, d=pixel_size)
    fy = np.fft.fftfreq(ny, d=pixel_size)
    Fx, Fy = np.meshgrid(fx, fy)

    spatial = Fx**2 + Fy**2
    cutoff = n0**2 / wavelength**2
    mask = spatial <= cutoff

    kz = np.zeros_like(Fx)
    kz[mask] = np.sqrt(n0**2 - wavelength**2 * spatial[mask])

    kernel = np.zeros_like(Fx, dtype=np.complex128)
    kernel[mask] = np.exp(1j * k * z * kz[mask])

    _kernel_cache[key] = kernel
    return kernel


def propagate(uin, z, n0, wavelength, pixel_size):
    """
    Propagate a complex optical field using the angular spectrum method.

    Parameters
    ----------
    uin : ndarray
        Input complex field.

    z : float
        Propagation distance [um].

    n0 : float
        Refractive index of propagation medium.

    wavelength : float
        Wavelength [um].

    pixel_size : float
        Sampling interval [um].

    Returns
    -------
    ndarray
        Propagated complex field.
    """

    ny, nx = uin.shape  # array shape: rows, columns

    # Backward propagation implemented using complex conjugation
    if z < 0:
        uin = np.conj(uin)

    U = np.fft.fft2(uin)
    kernel = _get_kernel(nx, ny, pixel_size, wavelength, n0, abs(z))

    U *= kernel
    uout = np.fft.ifft2(U)

    return np.conj(uout) if z < 0 else uout
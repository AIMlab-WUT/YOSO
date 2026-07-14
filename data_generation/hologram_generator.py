import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from propagation import propagate


def generate_hologram(phase,
                      amp,
                      phase_range,
                      amplitude_range,
                      px_trim,
                      gaussian_sigma,
                      z1,
                      z2,
                      wavelength,
                      pixel_size,
                      img_size,
                      pad_img_size
                      ):
                
    """
    Generate a pair of synthetic holograms at propagation distances
    z1 and z2 from natural images representing object phase and amplitude
    for YOSO model training.
    """

    # trim
    phase = phase[px_trim:-px_trim, px_trim:-px_trim]
    amp = amp[px_trim:-px_trim, px_trim:-px_trim]

    # grayscale
    if phase.ndim == 3:
        phase = phase.mean(axis=-1)
    if amp.ndim == 3:
        amp = amp.mean(axis=-1)

    # resize
    phase = zoom(phase, (img_size/phase.shape[0], img_size/phase.shape[1]), order=1)
    amp = zoom(amp, (img_size/amp.shape[0], img_size/amp.shape[1]), order=1)

    # filter + normalize
    phase = phase - gaussian_filter(phase, gaussian_sigma)

    phase = (phase - phase.min()) / (phase.max() - phase.min() + 1e-8)
    amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8)

    # random scaling
    phase *= np.random.uniform(*phase_range)
    amp_scale = np.random.uniform(*amplitude_range)

    # padding
    pad_val = (pad_img_size - img_size) // 2

    phase = np.pad(
        phase,
        ((pad_val, pad_val), (pad_val, pad_val)),
        mode="edge"
    )

    amp = np.pad(
        amp,
        ((pad_val, pad_val), (pad_val, pad_val)),
        mode="edge"
    )
    # field
    object_phase = phase - np.median(phase)
    object_amplitude = 1 + amp_scale * (amp - 0.5)
    object_field = object_amplitude * np.exp(1j * object_phase)

    # propagate
    n0 = 1  # Refractive index of air
    field_z1 = propagate(object_field, z1, n0, wavelength, pixel_size)
    field_z2 = propagate(object_field, z2, n0, wavelength, pixel_size)

    hologram_z1 = np.abs(field_z1)**2
    hologram_z2 = np.abs(field_z2)**2

    # crop
    crop_start = (pad_img_size - img_size) // 2
    crop_end = crop_start + img_size

    return (hologram_z1[crop_start:crop_end, crop_start:crop_end],
            hologram_z2[crop_start:crop_end, crop_start:crop_end]
            )

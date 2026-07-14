import matplotlib.pyplot as plt
import numpy as np


def display_generated_sample(
        phase_image,
        amplitude_image,
        input_holo,
        target_holo
):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Original RGB phase image
    axes[0].imshow(phase_image.astype(np.uint8))
    axes[0].set_title("Phase source image")
    axes[0].axis("off")

    # Original RGB amplitude image
    axes[1].imshow(amplitude_image.astype(np.uint8))
    axes[1].set_title("Amplitude source image")
    axes[1].axis("off")

    # Generated hologram z1
    axes[2].imshow(input_holo, cmap="gray")
    axes[2].set_title("Hologram z1")
    axes[2].axis("off")

    # Generated hologram z2
    axes[3].imshow(target_holo, cmap="gray")
    axes[3].set_title("Hologram z2")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()
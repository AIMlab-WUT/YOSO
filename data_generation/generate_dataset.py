import os
import time

import h5py
import imageio.v2 as imageio

from pathlib import Path

from hologram_generator import generate_hologram
from display import display_generated_sample
from config import load_config


def main():

    config = load_config("configs/config.yaml")

    data_path = Path(config["dataset"]["data_path"])
    img_count = config["dataset"]["img_count"]
    pixel_size = config["simulation"]["pixel_size"]
    img_size = config["simulation"]["img_size"]
    padded_img_size = config["simulation"]["padded_img_size"]
    wavelength = config["simulation"]["wavelength"]
    z1 = config["simulation"]["z1"]
    z2 = config["simulation"]["z2"]
    phase_range = config["object"]["phase_range"]
    amplitude_range = config["object"]["amplitude_range"]
    px_trim = config["preprocessing"]["px_trim"]
    gaussian_sigma = config["preprocessing"]["gaussian_sigma"]

    output_file = f"dataset_{img_count}_{img_size}x{img_size}.h5"


    # -----------------------
    # Dataset preparation
    # -----------------------

    files = sorted(os.listdir(data_path))[:img_count]

    with h5py.File(output_file, "w") as f:

        datasets = {
            name: f.create_dataset(
                name,
                shape=(0, img_size, img_size, 1),
                maxshape=(None, img_size, img_size, 1),
                dtype="float32"
            )
            for name in ["inputs", "targets"]
        }

        start = time.time()

        for i in range(len(files)):

            # Load natural image used as object phase/amplitude information
            ph_img = imageio.imread(data_path / files[i]).astype(float)
            amp_img = imageio.imread(
                data_path / files[(i + 1) % len(files)]
            ).astype(float)

            # Generate holographic training pair
            input_holo, target_holo = generate_hologram(
                ph_img,
                amp_img,
                phase_range,
                amplitude_range,
                px_trim,
                gaussian_sigma,
                z1,
                z2,
                wavelength,
                pixel_size,
                img_size,
                padded_img_size
            )

            if i == 0:
                display_generated_sample(
                    ph_img,
                    amp_img,
                    input_holo,
                    target_holo
                )

            # Save sample
            for name, data in zip(
                ["inputs", "targets"],
                [input_holo, target_holo]
            ):
                datasets[name].resize(
                    datasets[name].shape[0] + 1,
                    axis=0
                )
                datasets[name][-1] = data[..., None]

            print(f"Generated {i+1}/{img_count}")

    print(f"Dataset generation completed in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()

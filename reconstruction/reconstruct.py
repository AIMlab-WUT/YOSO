import numpy as np

from backend_config import get_backend
from FieldRetriever import FieldRetriever
from DNN import DNN
from loader import load_hologram, load_config, print_config
from utils import (
    remove_padding,
    generate_saving_dir,
    save_result,
    save_metadata,
    visualize
)

CONFIG_FILE = "config/stinkbug.yaml"
#CONFIG_FILE = "config/phase_test.yaml"
#CONFIG_FILE = "config/cheek_cells.yaml"
#CONFIG_FILE = "config/brain.yaml"
#CONFIG_FILE = "config/sperm_cells.yaml"

if __name__ == "__main__":
    # load configs
    yoso_config, opt_config = load_config(CONFIG_FILE)
    print_config(yoso_config, opt_config)
    holo = load_hologram(yoso_config["file_path"])

    # load and prepare hologram
    # select roi
    if "roi" in yoso_config:
        roi = yoso_config["roi"]
        holo = holo[roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3]]
    result_path = generate_saving_dir()

    # Prepare hologram
    holo = holo / np.median(holo)
    holo = np.pad(holo, (yoso_config["pad"], yoso_config["pad"]), mode='edge')

    print("YOSO inference step...")
    dnn = DNN(yoso_config["model_path"])
    holo2 = dnn.infer(holo)
    holos = np.dstack((holo, holo2))

    print("Hologram reconstruction has started...")
    backend = get_backend()
    field_retriever = FieldRetriever(np.shape(holos), **opt_config, backend=backend)
    field_retriever.set(holos)
    field_yoso = field_retriever.gerchberg_saxton(yoso_config["max_iter"])
    field_gabor = field_retriever.gabor()

    print("Visualization and saving...")
    field_yoso = remove_padding(field_yoso, yoso_config["pad"])
    field_gabor = remove_padding(field_gabor, yoso_config["pad"])
    amp_yoso, ph_yoso = np.abs(field_yoso), np.angle(field_yoso)
    amp_gabor, ph_gabor = np.abs(field_gabor), np.angle(field_gabor)

    # visualization
    visualize([holo, holo2],
                   ['I1', 'Estimated I2'],
                   cbar_label='intensity [a.u.]',
                   suptitle='Holograms',
                   save_dir=result_path,
                   cmap='gray')

    visualize([amp_gabor, amp_yoso],
                   ["Gabor", "YOSO"],
                   suptitle='Reconstructed amplitude',
                   cbar_label='[a.u.]',
                   save_dir=result_path,
                   cmap='viridis')

    visualize([ph_gabor, ph_yoso],
                   ["Gabor", "YOSO"],
                   suptitle='Reconstructed phase',
                   cbar_label='[rad]',
                   save_dir=result_path,
                   cmap='viridis')

    # saving
    save_result(result_path, "results", field_gabor=field_gabor, field_yoso=field_yoso)
    save_metadata(opt_config, yoso_config, output_dir=result_path / "metadata")

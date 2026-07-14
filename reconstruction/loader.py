import h5py
from pathlib import Path
import yaml


def load_config(path: str):
    config_path = Path(path).resolve()
    base_dir = Path.cwd()

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    yoso_config = cfg["general"]
    opt_config = cfg["opt"]

    yoso_config["file_path"] = (base_dir / yoso_config["file_path"]).resolve()
    yoso_config["model_path"] = (base_dir / yoso_config["model_path"]).resolve()

    opt_config = {
        "distances": (float(opt_config["z1"]), float(opt_config["z2"])),
        "wavelength": float(opt_config["wavelength"]),
        "sampling": float(opt_config["sampling"]),
    }

    return yoso_config, opt_config


def print_config(general: dict, opt: dict):

    print("=== OPT CONFIG ===")
    print(f"z1         : {opt.get('distances')[0]}")
    print(f"z2         : {opt.get('distances')[1]}")
    print(f"wavelength : {opt.get('wavelength')}")
    print(f"sampling   : {opt.get('sampling')}")

    print("\n=== GENERAL CONFIG ===")
    print(f"file_path  : {general.get('file_path')}")
    print(f"model_path : {general.get('model_path')}")
    print(f"pad        : {general.get('pad')}")
    print(f"max_iter   : {general.get('max_iter')}")

    roi = general.get("roi", None)
    if roi is not None:
        print(f"roi        : {roi}")
    else:
        print("roi        : None")

    print("=====================")


def load_hologram(filepath: str):
    """
    Load inline holograms.

    Expected datasets:
    - i_ccd1 : (height, width)
    """
    with h5py.File(filepath, "r") as f:
        i1 = f["i_ccd1"][()]
    return i1

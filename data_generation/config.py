import yaml
from pathlib import Path


def load_config(config_path):
    """
    Load YAML configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file.

    Returns
    -------
    dict
        Configuration parameters.
    """

    config_path = Path(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
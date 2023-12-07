import json


def load_config(config_path):
    """
    Load a config file from a path.

    Parameters
    ----------
    config_path : str
        Path to the config file.

    Returns
    -------
    config : dict
        Dictionary containing the config.
    """
    with open(config_path) as f:
        config = json.load(f)
    return config

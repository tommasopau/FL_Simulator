import yaml
import logging

def save_config(config: dict, config_path: str = "config.yaml"):
    """Saves the configuration dictionary to a file."""
    try:
        with open(config_path, "w") as file:
            yaml.safe_dump(config, file)
    except Exception as e:
        raise
from omegaconf import OmegaConf


def load_config(config_path):
    try:
        config = OmegaConf.load(config_path)
        OmegaConf.resolve(config)
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        exit(1)

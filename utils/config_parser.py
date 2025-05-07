import yaml

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration loaded successfully:")
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file {config_path}: {e}")
        exit(1)
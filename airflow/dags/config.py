"""
Configuration Module for Reviews Processing System

This module provides centralized configuration settings for the reviews scraping,
processing, and analysis pipeline. It includes file paths, HTTP settings, text
processing settings, and system parameters.
"""

import os
import yaml
from pathlib import Path

def load_config(config_path: str = "dags/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def resolve_paths(config: dict) -> dict:
    base_dir = Path(config["paths"]["base_dir"])
    resolved_paths = {}
    for key, value in config["paths"].items():
        resolved_paths[key] = Path(value.format(**{"base_dir": base_dir, **resolved_paths}))
    config["paths"] = resolved_paths
    config["files"] = {key: Path(value.format(**config["paths"])) for key, value in config["files"].items()}
    return config

def init_from_env(config: dict) -> dict:
    """Override configuration values from environment variables."""
    env_map = {
        "company_name": "REVIEWS_COMPANY_NAME",
        "request_settings.max_pages_to_scrape": "REVIEWS_MAX_PAGES",
        "request_settings.timeout": "REVIEWS_REQUEST_TIMEOUT",
    }
    for config_key, env_var in env_map.items():
        env_value = os.environ.get(env_var)
        if env_value is not None:
            keys = config_key.split(".")
            current = config
            for key in keys[:-1]:
                current = current[key]
            current[keys[-1]] = type(current[keys[-1]])(env_value)
    return config

# Main function to initialize configuration
def get_config() -> dict:
    config = load_config()
    config = resolve_paths(config)
    config = init_from_env(config)
    return config

if __name__ == "__main__":
    config = load_config()
    config = resolve_paths(config)
    print(config)
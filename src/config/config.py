import os
import yaml


class Config:
    def __init__(self) -> None:
        self.configs = None

    def set_configs(self, path: str) -> dict:
        try:
            with open(path, 'r') as file:
                self.configs = yaml.safe_load(file) or {}
        except FileNotFoundError:
            print(f"Warning: Config file {path} not found.")
            self.configs = {}
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")
            self.configs = {}

    def get_config(self, key, default=None):
        keys = key.split('.')
        value = self.configs
        for k in keys:
            if value is not None and isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

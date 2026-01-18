import os
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


class ConfigLoader:
    """Handles configuration loading from YAML files and environment variables.

    Attributes:
        config_path (str): Path to the YAML configuration file.
        _config (Dict[str, Any]): Internal dictionary storage for configuration.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initializes the ConfigLoader.

        Args:
            config_path (str): Path to the YAML configuration file. Defaults to "config.yaml".
        """
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._load_config()
        self._load_env()

    def _load_config(self) -> None:
        """Loads configuration from the YAML file into _config."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._config = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                print(f"Error parsing YAML configuration: {e}")
        else:
            print(f"Warning: Configuration file {self.config_path} not found.")

    def _load_env(self) -> None:
        """Loads environment variables using dotenv."""
        load_dotenv()

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value using dot-notation keys.

        Args:
            key (str): Dot-separated key path (e.g., 'tmdb.base_url').
            default (Any, optional): Default value if key is not found. Defaults to None.

        Returns:
            Any: The configuration value or the default.
        """
        keys = key.split(".")
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    @property
    def tmdb_api_key(self) -> Optional[str]:
        """Retrieves the TMDB API key from environment variables.

        Returns:
            Optional[str]: The API key or None if not set.
        """
        return os.getenv("TMDB_API_KEY")

    @property
    def tmdb_auth_token(self) -> Optional[str]:
        """Retrieves the TMDB Auth Token from environment variables.

        Returns:
            Optional[str]: The Auth Token or None if not set.
        """
        return os.getenv("TMDB_AUTH_TOKEN")

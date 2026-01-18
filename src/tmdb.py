import json
import os
import time
from typing import Any, Callable, Dict, List

import requests

from src.config import ConfigLoader


class TMDBClient:
    """Client for interacting with the The Movie Database (TMDB) API.

    Attributes:
        config (ConfigLoader): Configuration loader instance.
        base_url (str): Base URL for the TMDB API.
        headers (Dict[str, str]): HTTP headers for requests.
        api_key (Optional[str]): TMDB API Key.
    """

    def __init__(self, config: ConfigLoader) -> None:
        """Initializes the TMDBClient.

        Args:
            config (ConfigLoader): ConfigLoader instance.
        """
        self.config = config
        self.base_url = self.config.get("tmdb.base_url", "https://api.themoviedb.org/3")
        self.headers = {
            "accept": "application/json",
            "Authorization": self.config.tmdb_auth_token or "",
        }
        self.api_key = self.config.tmdb_api_key

    def check_authentication(self) -> str:
        """Checks if the authentication token is valid.

        Returns:
            str: Response text from the authentication endpoint.
        """
        endpoint = self.config.get("tmdb.authentication_endpoint", "/authentication")
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, headers=self.headers)
        return response.text

    def _download_data(
        self, file_path: str, items: List[Any], url_generator: Callable, label: str
    ) -> None:
        """Generic method to download data for a list of items.

        Args:
            file_path (str): Path to save the downloaded JSON data.
            items (List[Any]): List of items (titles, IDs) to process.
            url_generator (callable): Function that takes an item and returns the API URL.
            label (str): Label for logging (e.g., "title", "ID").
        """
        if os.path.exists(file_path):
            print(f"File {file_path} already exists. Skipping download.")
            return

        all_data: List[Dict[str, Any]] = []
        wait_time = float(self.config.get("tmdb.wait_time", 1.0))

        # Deduplicate items
        unique_items = list(set(items))

        from tqdm import tqdm

        for item in tqdm(unique_items, desc=f"Downloading by {label}"):
            url = url_generator(item)
            try:
                # Add small delay to avoid hitting rate limits instantly
                # time.sleep(0.01)
                response = requests.get(url, headers=self.headers)
                if response.status_code == 200:
                    all_data.append(response.json())
                else:
                    print(f"Error fetching {label} '{item}': {response.status_code}")
            except Exception as e:
                print(f"Exception fetching {label} '{item}': {e}")

            time.sleep(wait_time)

        with open(file_path, "w", encoding="utf-8") as f_out:
            json.dump(all_data, f_out, indent=4)
        print(f"Done downloading by {label}.")

    def download_data_by_movie_title(
        self, file_path: str, movie_titles: List[str]
    ) -> None:
        """Downloads movie data by list of titles.

        Args:
            file_path (str): Output file path.
            movie_titles (List[str]): List of movie titles.
        """
        endpoint = self.config.get("tmdb.search_movie_endpoint", "/search/movie")

        def url_gen(title):
            return f"{self.base_url}{endpoint}?query={title}"

        self._download_data(file_path, movie_titles, url_gen, "title")

    def download_data_by_movie_id(self, file_path: str, movie_ids: List[int]) -> None:
        """Downloads movie data by list of IDs.

        Args:
            file_path (str): Output file path.
            movie_ids (List[int]): List of movie IDs.
        """

        def url_gen(mid):
            return f"{self.base_url}/movie/{mid}?api_key={self.api_key}"

        self._download_data(file_path, movie_ids, url_gen, "ID")

    def download_credits(self, file_path: str, movie_ids: List[int]) -> None:
        """Downloads movie credits by list of IDs.

        Args:
            file_path (str): Output file path.
            movie_ids (List[int]): List of movie IDs.
        """

        def url_gen(mid):
            return f"{self.base_url}/movie/{mid}/credits?api_key={self.api_key}"

        self._download_data(file_path, movie_ids, url_gen, "credits")

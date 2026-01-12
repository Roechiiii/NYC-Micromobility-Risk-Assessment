import os
import requests
import xml.etree.ElementTree as ET
from src.config import Config

class CitibikeFetcher:
    BASE_URL =  Config.CITYBIKE_URL
    NS = {"s3": Config.CITYBIKE_NS}

    def __init__(self, data_dir):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def get_available_urls(self):
        print("Fetching file index from S3...")
        urls = []
        params = {}

        while True:
            resp = requests.get(self.BASE_URL, params=params)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)

            # collect keys
            for key in root.findall(".//s3:Key", self.NS):
                if key.text.endswith((".zip", ".csv")):
                    urls.append(self.BASE_URL + key.text)

            # check pagination
            is_truncated = root.find("s3:IsTruncated", self.NS)
            if is_truncated is None or is_truncated.text != "true":
                break

            next_token = root.find("s3:NextContinuationToken", self.NS)
            params["continuation-token"] = next_token.text

        return urls

    def download_file(self, url, max_retries=3):
        """Downloads a file with a basic retry mechanism."""
        filename = os.path.basename(url)
        local_path = os.path.join(self.data_dir, filename)

        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            print(f"  ↪ {filename} already exists. Skipping download.")
            return local_path

        for attempt in range(max_retries):
            try:
                print(f"Downloading {filename} (Attempt {attempt+1})...")
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            f.write(chunk)
                return local_path
            except Exception as e:
                print(f"Download failed: {e}")
                if attempt == max_retries - 1: raise
        return None
    

class CollisionFetcher:
    # Direct CSV export link for the NYC Open Data portal
    COLLISION_URL = Config.NYC_COLLISION_URL

    def __init__(self, data_dir):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def download_collisions(self):
        local_path = os.path.join(self.data_dir, "nyc_collisions_raw.csv")
        
        if os.path.exists(local_path):
            print("  ↪ Collision data already exists locally.")
            return local_path

        print("Downloading NYC Collision Data (this may take a while)...")
        with requests.get(self.COLLISION_URL, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
        return local_path
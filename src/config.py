import os
from dotenv import load_dotenv
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
env_path = root_dir / ".env"

load_dotenv(env_path)

class Config:
    DB_PATH = os.getenv("CITIBIKE_DB_PATH")
    
    CITYBIKE_URL = os.getenv("CITYBIKE_URL")
    CITYBIKE_NS = os.getenv("CITYBIKE_NS")
    NYC_COLLISION_URL = os.getenv("NYC_COLLISION_URL")

    OUTPUT_DIR_COLLISION = Path(os.getenv("OUTPUT_DIR_COLLISION", "./outputs/NYC_Collision"))
    OUTPUT_DIR_CITIBIKE = Path(os.getenv("OUTPUT_DIR_CITIBIKE", "./outputs/citibike"))
    OUTPUT_DIR_RISK = Path(os.getenv("OUTPUT_DIR_RISK", "./outputs/RiskIntegration"))

    OUTPUT_CACHE = os.getenv("CACHE_PATH")
    
    @classmethod
    def initialize_folders(cls):
        cls.OUTPUT_DIR_COLLISION.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR_CITIBIKE.mkdir(parents=True, exist_ok=True)

# Initialize when the module is imported
Config.initialize_folders()
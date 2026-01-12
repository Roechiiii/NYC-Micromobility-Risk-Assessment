import os
from dotenv import load_dotenv
from src.utils.db import DatabaseManager
from src.ingest.fetcher import CitibikeFetcher, CollisionFetcher
from src.ingest.processor import CitibikeProcessor, CollisionProcessor

def main():
    load_dotenv()
    
    # Configuration
    DATA_DIR = os.getenv("CITIBIKE_DATA_DIR", "citibike_data")
    DB_PATH = os.getenv("CITIBIKE_DB_PATH", "citibike.duckdb")

    # 1. Setup Resources
    db_mgr = DatabaseManager(DB_PATH)
    conn = db_mgr.connect()
    cb_fetcher = CitibikeFetcher(DATA_DIR)
    cb_processor = CitibikeProcessor(conn)

    # 2. Fetch and Download
    urls = cb_fetcher.get_available_urls()
    for url in urls:
        local_path = cb_fetcher.download_file(url)
        
        # 3. Process
        if local_path and local_path.endswith(".zip"):
            cb_processor.process_zip(local_path)
        elif local_path and local_path.endswith(".csv"):
            # processor.process_csv(local_path) 
            pass

    # 4. Final Aggregation
    cb_processor.consolidate_masters()
    
    # --- PHASE 2: COLLISIONS ---
    col_fetcher = CollisionFetcher(DATA_DIR)
    col_processor = CollisionProcessor(conn)
    
    col_path = col_fetcher.download_collisions()
    col_processor.process_collisions(col_path)

    # --- PHASE 3: ANALYSIS ---
    # Now you can run the spatial joins we discussed earlier!
    print("Pipeline Complete. Ready for Risk Analysis.")

    db_mgr.close()
    print("Full Pipeline Complete.")

if __name__ == "__main__":
    main()
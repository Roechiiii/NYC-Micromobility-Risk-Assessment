import os
import re
import shutil
import io
from zipfile import ZipFile
from datetime import datetime

class CitibikeProcessor:
    def __init__(self, db_conn):
        self.db = db_conn
        self._setup_manifest()

    def _setup_manifest(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS ingest_manifest (
                source_file TEXT, csv_path TEXT, table_name TEXT, 
                row_count BIGINT, ingested_at TIMESTAMP
            )
        """)

    def _is_mac_junk(self, path):
        return "__MACOSX" in path or "/._" in path or ".DS_Store" in path

    def _extract_yyyymm(self, name):
        # Look for 6 digits (YYYYMM)
        match = re.search(r"(20\d{2})(\d{2})", name)
        return f"{match.group(1)}{match.group(2)}" if match else "unknown"

    def process_zip(self, local_path):
        filename = os.path.basename(local_path)
        print(f"Processing Master ZIP: {filename}")
        
        with ZipFile(local_path) as z:
            self._handle_zip_contents(z, filename)

    def _handle_zip_contents(self, z_obj, original_filename):
        prefix = "JC_" if original_filename.startswith("JC") else "NYC_"
        
        for member in z_obj.namelist():
            if self._is_mac_junk(member):
                continue
            
            # 1. Handle Nested ZIPs (2020/2021 style)
            if member.lower().endswith(".zip"):
                print(f"  → Opening nested ZIP: {member}")
                with z_obj.open(member) as nested_data:
                    nested_bytes = io.BytesIO(nested_data.read())
                    with ZipFile(nested_bytes) as nz:
                        self._handle_zip_contents(nz, original_filename)
            
            # 2. Handle CSVs (including parts _1, _2, etc.)
            elif member.lower().endswith(".csv"):
                self._process_csv_member(z_obj, member, original_filename, prefix)

    def _process_csv_member(self, z_obj, member, original_filename, prefix):
        # Check manifest using both zip name and internal csv path
        exists = self.db.execute(
            "SELECT 1 FROM ingest_manifest WHERE source_file=? AND csv_path=?", 
            (original_filename, member)
        ).fetchone()
        
        if exists:
            return

        yyyymm = self._extract_yyyymm(member)
        table_name = f"{prefix}{yyyymm}"
        
        print(f"    → Loading {member} into {table_name}")
        
        # Extract to a flat temp file to avoid path issues
        os.makedirs("temp_data", exist_ok=True)
        temp_path = os.path.join("temp_data", f"load_{os.path.basename(member)}")
        
        with z_obj.open(member) as src, open(temp_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

        try:
            # Create table if it's the first time we see this month
            self.db.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM read_csv_auto('{temp_path}') WHERE 1=0")
            
            # Use union_by_name=True to handle schema variations within the same year
            self.db.execute(f"""
                INSERT INTO {table_name} 
                SELECT * FROM read_csv_auto('{temp_path}', union_by_name=True, ignore_errors=True)
            """)
            
            # Verify row count for manifest
            cnt = self.db.execute(f"SELECT count(*) FROM read_csv_auto('{temp_path}', ignore_errors=True)").fetchone()[0]
            
            self.db.execute("INSERT INTO ingest_manifest VALUES (?, ?, ?, ?, ?)", 
                            (original_filename, member, table_name, cnt, datetime.now()))
        except Exception as e:
            print(f"Failed to load {member}: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def consolidate_masters(self):
        for prefix in ["NYC", "JC"]:
            master_table = f"{prefix}_MASTER"
            self.db.execute(f"DROP TABLE IF EXISTS {master_table}")
            
            # Get list of tables
            tables = [t[0] for t in self.db.execute(
                f"SELECT table_name FROM ingest_manifest WHERE table_name LIKE '{prefix}_20%'"
            ).fetchall()]
            
            if not tables: continue
            
            print(f"Consolidating {len(tables)} tables for {master_table}...")

            # 1. Create the table structure using the first table
            first_sql = self._get_normalization_sql(tables[0])
            self.db.execute(f"CREATE TABLE {master_table} AS {first_sql} LIMIT 0")
            
            # 2. Insert tables one by one (avoids one giant crashing query)
            for table in tables:
                print(f"Stacking {table}...")
                insert_sql = self._get_normalization_sql(table)
                try:
                    # No 'SET' command needed here
                    self.db.execute(f"INSERT INTO {master_table} {insert_sql}")
                except Exception as e:
                    print(f"Serious error in {table}: {e}")

            print(f"{master_table} is ready.")

    def _get_normalization_sql(self, table_name):
        cols = [c[1] for c in self.db.execute(f"PRAGMA table_info('{table_name}')").fetchall()]
        
        # Define our "Target" names and their "Possible" source names
        mapping = {
            "started_at": ["starttime", "Start Time", "started_at"],
            "ended_at": ["stoptime", "Stop Time", "ended_at"],
            "start_station_name": ["start station name", "Start Station Name", "start_station_name"],
            "start_station_id": ["start station id", "Start Station ID", "start_station_id"],
            "end_station_name": ["end station name", "End Station Name", "end_station_name"],
            "end_station_id": ["end station id", "End Station ID", "end_station_id"],
            "start_lat": ["start station latitude", "Start Station Latitude", "start_lat"],
            "start_lng": ["start station longitude", "Start Station Longitude", "start_lng"],
            "end_lat": ["end station latitude", "End Station Latitude", "end_lat"],
            "end_lng": ["end station longitude", "End Station Longitude", "end_lng"],
            "member_casual": ["usertype", "User Type", "member_casual"]
        }

        # Helper for the date parsing mess
        def date_parse(col):
            return f"""
                COALESCE(
                    try_cast("{col}" AS TIMESTAMP),
                    try_strptime("{col}"::VARCHAR, '%m/%d/%Y %H:%M:%S'),
                    try_strptime("{col}"::VARCHAR, '%m/%d/%Y %H:%M'),
                    try_strptime("{col}"::VARCHAR, '%-m/%-d/%Y %H:%M:%S'),
                    try_strptime("{col}"::VARCHAR, '%-m/%-d/%Y %H:%M')
                )"""

        select_parts = []
        for target, sources in mapping.items():
            found = next((s for s in sources if s in cols), None)
            if found:
                if "started_at" in target or "ended_at" in target:
                    select_parts.append(f"{date_parse(found)} AS {target}")
                elif "id" in target:
                    # Use TRY_CAST to VARCHAR and regex to clean IDs
                    select_parts.append(f'regexp_replace(try_cast("{found}" AS VARCHAR), \'\.0$\', \'\') AS {target}')
                else:
                    # This is the critical change for the Unicode error:
                    # We cast to VARCHAR. If DuckDB hits an invalid byte, 
                    # try_cast often helps it fail gracefully to NULL.
                    select_parts.append(f'try_cast("{found}" AS VARCHAR) AS {target}')
            else:
                select_parts.append(f"NULL AS {target}")

        select_parts.append("uuid() AS ride_id")
        # Clean duration calculation:
        start_col = next((s for s in mapping["started_at"] if s in cols), "started_at")
        end_col = next((s for s in mapping["ended_at"] if s in cols), "ended_at")
        select_parts.append(f"epoch({date_parse(end_col)} - {date_parse(start_col)}) AS tripduration_sq")

        return f"SELECT {', '.join(select_parts)} FROM {table_name}"
            

class CollisionProcessor:
    def __init__(self, db_conn):
        self.db = db_conn

    def process_collisions(self, csv_path):
        print("Ingesting collisions into DuckDB...")
        
        # Using DuckDB's high-performance CSV reader
        # We handle column names with spaces by quoting them
        self.db.execute(f"""
            CREATE TABLE IF NOT EXISTS collisions_raw AS 
            SELECT * FROM read_csv_auto('{csv_path}', types={{'ZIP CODE': 'VARCHAR'}})
        """)
        
        # Clean up the table names to be SQL-friendly
        self.db.execute("""
            CREATE OR REPLACE TABLE collisions AS 
            SELECT 
                "CRASH DATE"::DATE as crash_date,
                "CRASH TIME"::TIME as crash_time,
                BOROUGH,
                "ZIP CODE" as zip_code,
                LATITUDE::DOUBLE as latitude,
                LONGITUDE::DOUBLE as longitude,
                "ON STREET NAME" as on_street,
                "NUMBER OF CYCLIST INJURED"::INT as cyclist_injured,
                "NUMBER OF CYCLIST KILLED"::INT as cyclist_killed,
                "CONTRIBUTING FACTOR VEHICLE 1" as factor_1,
                "VEHICLE TYPE CODE 1" as vehicle_1,
                COLLISION_ID
            FROM collisions_raw
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        """)
        print(f"Collisions table ready: {self.db.execute('SELECT count(*) FROM collisions').fetchone()[0]} rows.")
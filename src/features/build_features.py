"""
MISSION: The Feature Layer.
Aggregates raw DuckDB data into station-level risk metrics.
Performs SQL-based spatial joins between collisions and stations.
"""
import pandas as pd
import numpy as np

class FeatureBuilder:
    def __init__(self):
        """Initialize the Feature Builder for station-level aggregation."""
        pass

    def build_station_safety_index(self, con):
        """
        Creates the 'station_safety_index' table in DuckDB for station-level risk analysis.
        Aggregates trip counts and maps collisions to stations via spatial bounding box.
        """
        print("    Building Station Safety Index (SQL Aggregation)...")
        query = """
        CREATE OR REPLACE TABLE station_safety_index AS
        WITH station_stats AS (
            SELECT 
                start_station_name as name, 
                AVG(TRY_CAST(start_lat AS DOUBLE)) as lat, 
                AVG(TRY_CAST(start_lng AS DOUBLE)) as lng, 
                COUNT(*) as trip_count
            FROM NYC_MASTER
            WHERE start_station_name IS NOT NULL
            GROUP BY 1
        ),
        accident_data AS (
            SELECT 
                s.name,
                ANY_VALUE(c.borough) as borough,
                ANY_VALUE(c.zip_code) as zip_code,
                COUNT(c.collision_id) as crash_count,
                SUM(c.cyclist_injured) as injuries,
                SUM(c.cyclist_killed) as deaths
            FROM station_stats s
            JOIN collisions c ON 
                (TRY_CAST(c.latitude AS DOUBLE) BETWEEN s.lat - 0.0015 AND s.lat + 0.0015) AND
                (TRY_CAST(c.longitude AS DOUBLE) BETWEEN s.lng - 0.0015 AND s.lng + 0.0015)
            GROUP BY 1
        )
        SELECT 
            s.*,
            a.borough, a.zip_code,
            COALESCE(a.crash_count, 0) as crash_count,
            COALESCE(a.injuries, 0) as injuries,
            COALESCE(a.deaths, 0) as deaths,
            -- Risk Index: Weighted severity per 10k trips
            ((COALESCE(a.injuries, 0) * 3) + (COALESCE(a.deaths, 0) * 10)) * 10000.0 / NULLIF(s.trip_count, 0) as risk_index
        FROM station_stats s
        LEFT JOIN accident_data a ON s.name = a.name
        """
        try:
            con.execute(query)
            # count rows
            count = con.execute("SELECT COUNT(*) FROM station_safety_index").fetchone()[0]
            print(f"Station Safety Index created with {count} stations.")
        except Exception as e:
            print(f"Error creating station safety index: {e}")

"""
MISSION: The Flow Layer.
Analyzes abstract station-to-station connections (Spider Maps) and 
implements the Distance-Normalized Risk Score.
"""
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pickle
import os
import duckdb
import geopandas as gpd
from shapely.geometry import LineString
import contextily as ctx
from src.config import Config  

class RiskNetworkAnalyzer:
    """
    Analyzes the 'abstract' station-to-station network.
    Layer B of the Risk Analysis Pipeline.
    """
    def __init__(self, db_conn, output_dir="outputs", cache_dir=None):
        self.con = db_conn
        self.output_dir = output_dir
        self.cache_dir = Config.OUTPUT_CACHE
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def build_risk_graph(self, min_volume=500):
        """Vectorized SQL to build a graph where edges contain Risk-per-Rider data."""
        cache_path = os.path.join(self.cache_dir, "graph_data.pkl")

        # QUICK LOAD: If we already did the work, just load it
        if os.path.exists(cache_path):
            print("Loading station graph data from cache...", flush=True)
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"    ⚠️ Cache corrupt, rebuilding graph. ({e})", flush=True)
            
        print(f"Building Path-Risk Graph (Top 500 High-Volume Routes)...", flush=True)
        
        # Find the busiest routes first!!!
        # This reduces the working set for the complex join significantly.
        try:
            print("    Step A: Identifying Top 500 Busiest Routes...", flush=True)
            path_query = """
            SELECT 
                start_station_name as source, 
                end_station_name as target,
                COUNT(*) as volume
            FROM (SELECT * FROM NYC_MASTER LIMIT 1000000)
            WHERE start_station_name IS NOT NULL 
              AND end_station_name IS NOT NULL
              AND start_lat != 'NULL' AND end_lat != 'NULL'
            GROUP BY source, target
            ORDER BY volume DESC
            LIMIT 500
            """
            top_routes = self.con.execute(path_query).df()
            
            if top_routes.empty:
                print("⚠️ No routes found.")
                return None, None, None
                
            print(f"    Found {len(top_routes)} routes. Fetching coordinates...", flush=True)
            self.con.register('top_routes_list', top_routes)
            
            # Enrich with Coordinates 
            # and join back to get coords
            coord_query = """
            WITH route_coords AS (
                SELECT 
                    start_station_name as source, 
                    end_station_name as target,
                    TRY_CAST(start_lat AS DOUBLE) as s_lat, 
                    TRY_CAST(start_lng AS DOUBLE) as s_lng,
                    TRY_CAST(end_lat AS DOUBLE) as e_lat, 
                    TRY_CAST(end_lng AS DOUBLE) as e_lng
                FROM NYC_MASTER
                WHERE start_station_name IN (SELECT source FROM top_routes_list)
                  AND end_station_name IN (SELECT target FROM top_routes_list)
            )
            SELECT 
                source, target,
                AVG(s_lat) as s_lat, AVG(s_lng) as s_lng,
                AVG(e_lat) as e_lat, AVG(e_lng) as e_lng
            FROM route_coords
            GROUP BY source, target
            """
            coords_df = self.con.execute(coord_query).df()
            
            # Combine
            paths_df = pd.merge(top_routes, coords_df, on=['source', 'target'])
            self.con.register('paths_temp', paths_df)
            
        except Exception as e:
            print(f"⚠️ Error identifying paths: {e}")
            return None, None, None

        # Step 2: Spatial Join with Collisions
        join_query = """
        WITH path_bounds AS (
            SELECT 
                *,
                CASE WHEN s_lat < e_lat THEN s_lat ELSE e_lat END as min_lat,
                CASE WHEN s_lat > e_lat THEN s_lat ELSE e_lat END as max_lat,
                CASE WHEN s_lng < e_lng THEN s_lng ELSE e_lng END as min_lng,
                CASE WHEN s_lng > e_lng THEN s_lng ELSE e_lng END as max_lng
            FROM paths_temp
        ),
        clean_collisions AS (
            SELECT 
                TRY_CAST(latitude AS DOUBLE) as lat,
                TRY_CAST(longitude AS DOUBLE) as lng,
                collision_id
            FROM collisions
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        )
        SELECT 
            p.*,
            COUNT(c.collision_id) as path_accidents
        FROM path_bounds p
        LEFT JOIN clean_collisions c ON 
             c.lat BETWEEN p.min_lat AND p.max_lat AND
             c.lng BETWEEN p.min_lng AND p.max_lng
        GROUP BY 
            p.source, p.target, p.s_lat, p.s_lng, p.e_lat, p.e_lng, p.volume,
            p.min_lat, p.max_lat, p.min_lng, p.max_lng
        """
        
        try:
            print("Step B: Mapping accidents to Top 500 routes...", flush=True)
            edges_df = self.con.execute(join_query).df()
        except Exception as e:
            print(f"⚠️ Error in spatial join: {e}")
            return None, None, None

        # Calculate Risk Ratio (Accidents per 1000 riders)
        edges_df['dist_deg'] = ((edges_df['s_lat'] - edges_df['e_lat'])**2 + (edges_df['s_lng'] - edges_df['e_lng'])**2)**0.5
        edges_df['dist_deg'] = edges_df['dist_deg'].replace(0, 0.0001)
        
        # Risk Density
        edges_df['risk_ratio'] = (edges_df['path_accidents'] * 1000.0) / (edges_df['volume'] * edges_df['dist_deg'])
        
        G = nx.DiGraph()
        for _, row in edges_df.iterrows():
            G.add_edge(row['source'], row['target'], volume=row['volume'], risk_ratio=row['risk_ratio'])
        
        centrality = {}
        
        results = (G, edges_df, centrality)
        with open(cache_path, 'wb') as f: 
            pickle.dump(results, f)
            
        print(f"    Graph loaded/built. Edges: {len(edges_df) if edges_df is not None else 0}")
        return results

    def generate_risk_report(self, edges_df, top_n=25):
        """
        Generates a tabular report of graph statistics and the 'Top N' risky paths.
        Alternatives to complex visualization.
        """
        print(f"Generating Risk Analysis Report (Top {top_n} Paths)...", flush=True)
        if edges_df is None or edges_df.empty:
            print("⚠️ No data to report.", flush=True)
            return

        # Handle column naming inconsistency (cache vs fresh)
        if 'path_accidents' not in edges_df.columns and 'path_risk' in edges_df.columns:
            edges_df.rename(columns={'path_risk': 'path_accidents'}, inplace=True)
            
        # 1. Global Network Statistics
        total_volume = edges_df['volume'].sum()
        total_accidents = edges_df['path_accidents'].sum() if 'path_accidents' in edges_df.columns else 0
        avg_risk = edges_df['risk_ratio'].mean()
        
        # Weighted Risk (Risk weighted by volume)
        weighted_risk = (edges_df['risk_ratio'] * edges_df['volume']).sum() / total_volume
        
        print("\n---Network Risk Statistics ---")
        print(f"  • Total Modeled Trips: {total_volume:,}")
        print(f"  • Total Approx. Accidents on Paths: {total_accidents:,}")
        print(f"  • Average Path Risk Score: {avg_risk:.4f}")
        print(f"  • Volume-Weighted Risk Score: {weighted_risk:.4f}")
        print(f"  • Connectivity (Edges): {len(edges_df):,}")
        print("-----------------------------------\n")

        # 2. Top N Risky Paths
        # Columns to display
        cols = ['source', 'target', 'volume', 'path_accidents', 'dist_deg', 'risk_ratio']
        
        # Sort by Risk Ratio
        top_risky = edges_df.sort_values(by='risk_ratio', ascending=False).head(top_n).copy()
        
        # Ensure dist_deg exists for reporting
        if 'dist_deg' not in top_risky.columns:
             top_risky['dist_deg'] = ((top_risky['s_lat'] - top_risky['e_lat'])**2 + (top_risky['s_lng'] - top_risky['e_lng'])**2)**0.5

        # Formatting for display
        display_df = top_risky[cols].copy()
        display_df['risk_ratio'] = display_df['risk_ratio'].round(2)
        display_df['dist_deg'] = display_df['dist_deg'].round(4)
        
        print(f"Top {top_n} Most Dangerous Routes:")
        print(display_df.to_string(index=False))
        
        # 3. Export to CSV
        out_path = os.path.join(self.output_dir, "top_25_risky_routes.csv")
        top_risky.to_csv(out_path, index=False)
        print(f"Detailed report saved to: {out_path}", flush=True)

    def plot_final_clean_network(self, G, edges_df, centrality_scores):
        print("Cleaning geographic outliers and plotting...", flush=True)
        
        # 1. THE GEOGRAPHIC CLAMP
        # NYC Bounds: Longitude [-74.05, -73.85], Latitude [40.65, 40.85]
        mask = (edges_df['s_lat'] > 40.6) & (edges_df['s_lat'] < 40.9) & \
               (edges_df['s_lng'] > -74.1) & (edges_df['s_lng'] < -73.8) & \
               (edges_df['e_lat'] > 40.6) & (edges_df['e_lat'] < 40.9) & \
               (edges_df['e_lng'] > -74.1) & (edges_df['e_lng'] < -73.8)
        
        df_plot = edges_df[mask].copy()

        # Clip risk at 95th percentile
        vmax_limit = df_plot['risk_ratio'].quantile(0.95)

        fig, ax = plt.subplots(figsize=(15, 15), facecolor='#0B0B0B')
        
        lines = [[(row['s_lng'], row['s_lat']), (row['e_lng'], row['e_lat'])] for _, row in df_plot.iterrows()]
        lc = LineCollection(lines, cmap='YlOrRd', norm=plt.Normalize(0, vmax_limit))
        lc.set_array(df_plot['risk_ratio'].values)
        lc.set_linewidth(0.6)
        lc.set_alpha(0.4)
        ax.add_collection(lc)

        unique_stations = df_plot[['source', 's_lat', 's_lng']].drop_duplicates()

        sizes = [centrality_scores.get(name, 0.001) * 10000 for name in unique_stations['source']]
        
        ax.scatter(unique_stations['s_lng'], unique_stations['s_lat'], 
                   s=sizes, c='#00F5FF', alpha=0.7, edgecolors='white', linewidth=0.3, zorder=3)

        ax.set_xlim(-74.05, -73.88) # Force zoom onto Manhattan/Brooklyn core
        ax.set_ylim(40.66, 40.83)
        ax.set_aspect('equal')
        ax.axis('off')

        cbar = fig.colorbar(lc, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label('Liability Exposure (Path Risk)', color='white', size=12)
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

        plt.title("NYC CITIBIKE: SYSTEMIC RISK GRAPH", color='white', fontsize=20, fontweight='bold')
        
        try:
            ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.DarkMatter, alpha=0.6)
        except Exception as e:
            print(f"⚠️ Could not add basemap: {e}")

        save_path = os.path.join(self.output_dir, "nyc_graph_final.png")
        plt.savefig(save_path, dpi=300, facecolor='#0B0B0B', bbox_inches='tight')
        print(f"Graph saved to: {save_path}", flush=True)
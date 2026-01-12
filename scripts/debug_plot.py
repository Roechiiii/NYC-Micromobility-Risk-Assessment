import pickle
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import contextily as ctx
import pandas as pd
import numpy as np
import os
from src.config import Config

# Load data
cache_path = Config.OUTPUT_CACHE+"/graph_data.pkl"
try:
    with open(cache_path, 'rb') as f:
        print(f"Loading {cache_path}...")
        results = pickle.load(f)
        G, edges_df, centrality = results
        print("Data loaded.")
except Exception as e:
    print(f"Failed to load cache: {e}")
    # Try local path if the big data path fails (fallback)
    cache_path = "data/graph_data.pkl" 
    # Note: user code had hardcoded /media/... path default but check logic
    exit(1)

def plot_final_clean_network(G, edges_df, centrality_scores):
    print("Cleaning geographic outliers and plotting...")
    
    # 1. THE GEOGRAPHIC CLAMP
    mask = (edges_df['s_lat'] > 40.6) & (edges_df['s_lat'] < 40.9) & \
           (edges_df['s_lng'] > -74.1) & (edges_df['s_lng'] < -73.8) & \
           (edges_df['e_lat'] > 40.6) & (edges_df['e_lat'] < 40.9) & \
           (edges_df['e_lng'] > -74.1) & (edges_df['e_lng'] < -73.8)
    
    df_plot = edges_df[mask].copy()
    print(f"Plotting {len(df_plot)} edges.")

    # 2. THE COLOR CLAMP
    vmax_limit = df_plot['risk_ratio'].quantile(0.95)

    fig, ax = plt.subplots(figsize=(15, 15), facecolor='#0B0B0B') 
    
    # 3. DRAW THE EDGES
    lines = [[(row['s_lng'], row['s_lat']), (row['e_lng'], row['e_lat'])] for _, row in df_plot.iterrows()]
    lc = LineCollection(lines, cmap='YlOrRd', norm=plt.Normalize(0, vmax_limit))
    lc.set_array(df_plot['risk_ratio'].values)
    lc.set_linewidth(0.6)
    lc.set_alpha(0.4)
    ax.add_collection(lc)

    # 4. DRAW THE NODES
    unique_stations = df_plot[['source', 's_lat', 's_lng']].drop_duplicates()
    sizes = [centrality_scores.get(name, 0.001) * 10000 for name in unique_stations['source']]
    
    ax.scatter(unique_stations['s_lng'], unique_stations['s_lat'], 
               s=sizes, c='#00F5FF', alpha=0.7, edgecolors='white', linewidth=0.3, zorder=3)

    # 5. FINAL TOUCHES
    ax.set_xlim(-74.05, -73.88) 
    ax.set_ylim(40.66, 40.83)
    ax.set_aspect('equal')
    ax.axis('off')

    cbar = fig.colorbar(lc, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Liability Exposure (Path Risk)', color='white', size=12)
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

    plt.title("NYC CITIBIKE: SYSTEMIC RISK GRAPH", color='white', fontsize=20, fontweight='bold')
    
    # 6. MAP BACKGROUND
    try:
        print("Adding basemap...")
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.DarkMatter, alpha=0.6)
        print("Basemap added.")
    except Exception as e:
        print(f"Could not add basemap: {e}")

    save_path = "debug_graph.png"
    plt.savefig(save_path, dpi=300, facecolor='#0B0B0B', bbox_inches='tight')
    print(f"Graph saved to: {save_path}")

plot_final_clean_network(G, edges_df, centrality)

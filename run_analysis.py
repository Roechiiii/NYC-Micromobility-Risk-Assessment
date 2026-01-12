import os
from dotenv import load_dotenv
from src.utils.db import DatabaseManager

# Placeholder imports for future modules
# from src.graph.network import StreetNetwork
# from src.models.risk_model import RiskModel

def main():
    load_dotenv()
    
    # Configuration
    DB_PATH = os.getenv("CITIBIKE_DB_PATH", "citibike.duckdb")
    
    print("=" * 80)
    print("  MICROMOBILITY RISK ASSESSMENT")
    print("=" * 80)
    
    # 1. Connect to DB
    db_mgr = DatabaseManager(DB_PATH)
    conn = db_mgr.connect()
    
    # PERFORMANCE TUNING: Enable disk spilling to prevent OOM on large aggregations
    try:
        conn.execute("PRAGMA memory_limit='2GB'") 
        conn.execute("PRAGMA temp_directory='/tmp/duckdb_temp'")
    except:
        pass # In case of permission error or older version Analysis
    
    
    # --- LAYER 1: AGGREGATION ---
    print("\n" + "=" * 80)
    print("LAYER 1: STATION AGGREGATION")
    print("Purpose: Build station-level risk metrics via SQL spatial joins")
    print("=" * 80)
    
    print("\n→ Aggregating trip volumes by station...")
    print("→ Mapping collisions to stations (±150m radius)...")
    print("→ Calculating Risk Index: (Deaths×10 + Injuries×3) × 10k / Trips")
    
    from src.features.build_features import FeatureBuilder
    fb = FeatureBuilder()
    fb.build_station_safety_index(conn)
    
    
    # --- LAYER 2: INTELLIGENCE ---
    print("\n" + "=" * 80)
    print("LAYER 2: STRATEGIC INTELLIGENCE")
    print("Purpose: Segment stations into actuarial tiers & predict high-risk zones")
    print("=" * 80)
    
    print("\n→ Step 2.1: K-Means Clustering (4 Actuarial Tiers)")
    print("   Grouping stations by risk profile: Safe → Moderate → High → Critical")
    
    from src.models.safety_clustering import InsuranceRiskML
    risk_ml = InsuranceRiskML(conn)
    
    # Run Clustering
    clustered_stations = risk_ml.run_clustering_pipeline()
    
    print("\n→ Step 2.2: Random Forest Risk Classifier")
    print("   Training predictive model: Can we predict high-risk stations?")
    
    # Train Classifier
    rf_model = risk_ml.train_risk_classifier()
    
    print("\n→ Step 2.3: Financial Liability Watchlist")
    print("   Identifying top 10 stations with highest estimated liability")
    
    # Generate Watchlist
    risk_ml.generate_financial_watchlist(clustered_stations)
    
    
    # --- LAYER 3: FLOW ANALYSIS ---
    print("\n" + "=" * 80)
    print("LAYER 3: FLOW ANALYSIS")
    print("Purpose: Analyze station-to-station network & identify systemic risk corridors")
    print("=" * 80)
    
    print("\n→ Step 3.1: Building Station-to-Station Graph")
    print("   Analyzing Top 500 highest-volume routes")
    print("   Calculating Distance-Normalized Risk Score")
    
    from src.graph.trip_network import RiskNetworkAnalyzer
    net_analyzer = RiskNetworkAnalyzer(conn)
    
    # Build Abstract Graph
    graph_results = net_analyzer.build_risk_graph(min_volume=500)
    
    # Plot / Report
    if graph_results[1] is not None:
        print("\n→ Step 3.2: Generating Risk Reports & Visualizations")
        
        # Generate Report
        net_analyzer.generate_risk_report(graph_results[1], top_n=25)
        
        # Generate Plot
        G, edges_df, centrality = graph_results
        net_analyzer.plot_final_clean_network(G, edges_df, centrality)

    # Final Summary
    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("  All outputs saved to: outputs/")
    print("=" * 80)
    
    db_mgr.close()

if __name__ == "__main__":
    main()
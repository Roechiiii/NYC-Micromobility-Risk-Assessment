"""
MISSION: The Strategic Layer.
Segments stations into actuarial "Insurance Tiers" using K-Means clustering.
Predicts high-liability zones for financial exposure management.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import duckdb
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class InsuranceRiskML:
    """
    Performs station-level risk assessment using clustering and classification.
    Layer B of the Risk Analysis Pipeline.
    """
    def __init__(self, db_conn, output_dir="outputs"):
        self.con = db_conn
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def run_clustering_pipeline(self, n_clusters=4):
        """Discovers Risk Archetypes via K-Means."""
        print(f"Discovering {n_clusters} Risk Archetypes...")
        
        # Ensure table exists (handled externally or checking here?)
        # Ideally, station_safety_index is created before calling this.
        try:
            df = self.con.execute("""
                SELECT name, borough, zip_code, trip_count, risk_index, injuries, deaths 
                FROM station_safety_index
            """).df()
        except duckdb.CatalogException:
            print("Table 'station_safety_index' not found. Please run aggregation first.")
            return None

        if df.empty:
            print("No data in station_safety_index.")
            return df

        scaler = StandardScaler()
        # Handle potential NaNs
        df_feats = df[['trip_count', 'risk_index']].fillna(0)
        scaled_features = scaler.fit_transform(df_feats)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(scaled_features)

        # Map Tiers: Tier 1 (Safe) -> Tier 4 (Dangerous)
        # We sort by mean risk_index to ensure Tier 1 is lowest risk
        cluster_grp = df.groupby('cluster')['risk_index'].mean().sort_values()
        cluster_order = cluster_grp.index
        # Create mapping: cluster_id -> "Tier X"
        tier_mapping = {cluster_id: f"Tier {i+1}" for i, cluster_id in enumerate(cluster_order)}
        
        df['insurance_tier'] = df['cluster'].map(tier_mapping)

        self._plot_clusters(df)
        return df

    def _plot_clusters(self, df):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='trip_count', y='risk_index', hue='insurance_tier', palette='RdYlGn_r')
        plt.title("Actuarial Segmentation: Risk Clusters (K-Means)")
        plt.xlabel("Trip Volume")
        plt.ylabel("Risk Index")
        
        out_path = os.path.join(self.output_dir, 'insurance_clusters.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Cluster plot saved to {out_path}")

    def train_risk_classifier(self):
        """Predicts if a station will become 'High Risk' based on volume and borough."""
        print("\nTraining Random Forest Risk Predictor (Station Level)...")
        try:
            df = self.con.execute("""
                SELECT trip_count, risk_index, borough
                FROM station_safety_index WHERE borough IS NOT NULL
            """).df()
        except duckdb.CatalogException:
            return None
            
        if df.empty:
            return None

        # Feature Engineering for Model
        median_risk = df['risk_index'].median()
        df['is_high_risk'] = (df['risk_index'] > median_risk).astype(int)
        
        # Simple encoding
        df['b_code'] = df['borough'].map({
            'MANHATTAN': 1, 'BROOKLYN': 2, 'QUEENS': 3, 'BRONX': 4
        }).fillna(0)

        X = df[['trip_count', 'b_code']]
        y = df['is_high_risk']
        
        if len(y.unique()) < 2:
            print("Not enough class diversity for classification.")
            return None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
        
        print("\nModel Performance (Station Risk):")
        print(classification_report(y_test, rf.predict(X_test)))
        return rf

    def generate_financial_watchlist(self, df_clustered, avg_injury_cost=35000):
        """Generates the executive summary for liability exposure."""
        if df_clustered is None or df_clustered.empty:
            return None
            
        print("Generating High-Liability Watchlist...")
        # Fill NaNs for safety
        df_clustered['injuries'] = df_clustered['injuries'].fillna(0)
        df_clustered['deaths'] = df_clustered['deaths'].fillna(0)
        
        df_clustered['est_liability'] = (df_clustered['injuries'] * avg_injury_cost) + (df_clustered['deaths'] * 500000)
        
        watchlist = df_clustered.sort_values(by='est_liability', ascending=False).head(10)
        
        out_path = os.path.join(self.output_dir, "critical_failure_watchlist.csv")
        watchlist.to_csv(out_path, index=False)
        print(f"Watchlist saved to {out_path}")
        return watchlist

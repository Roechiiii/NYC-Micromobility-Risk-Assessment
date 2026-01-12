import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import duckdb
import matplotlib.dates as mdates
from src.config import Config  
import os

class RiskIntegrator:
    def __init__(self, db_path=None, output_dir=None):
        self.db_path = Config.DB_PATH        
        self.output_dir = Config.OUTPUT_DIR_RISK
        os.makedirs(self.output_dir, exist_ok=True)
        self.con = None

    def __enter__(self):
        self.con = duckdb.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.con: self.con.close()

    def _save_plot(self, filename: str):
        """Internal helper to standardize how plots are saved."""
        if filename:
            if not filename.endswith(('.png', '.jpg', '.pdf')):
                filename += '.png'
            
            save_path = self.output_dir / filename
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print("Plot saved!")

    def q_to_df(self, sql):
        """Standardizes DuckDB output to lowercase for Seaborn/Pandas compatibility."""
        df = self.con.execute(sql).df()
        df.columns = [c.lower() for c in df.columns]
        return df
    
    # 1. RISK PER EXPOSURE: Crashes per 100k Trips by Hour
    def plot_hourly_risk_rate(self, filename="plot_hourly_risk_rate_combined"):
        print("Integrating Plot 1: Hourly Risk Rate...")
        query = """
            WITH bike_vol AS (
                SELECT hour(started_at) as hr, count(*) as trip_count 
                FROM NYC_MASTER GROUP BY 1
            ),
            crash_vol AS (
                SELECT hour(crash_time) as hr, count(*) as crash_count 
                FROM collisions GROUP BY 1
            )
            SELECT b.hr, 
                   (CAST(c.crash_count AS FLOAT) / b.trip_count) * 100000 as risk_index
            FROM bike_vol b
            JOIN crash_vol c ON b.hr = c.hr
            ORDER BY b.hr
        """
        df = self.con.execute(query).df()
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(df['hr'], df['risk_index'], color="red", alpha=0.1)
        plt.plot(df['hr'], df['risk_index'], marker='o', color='red', linewidth=3)
        
        plt.title("The 'Danger' Index: Crashes per 100,000 CitiBike Trips", fontsize=15, fontweight='bold')
        plt.xlabel("Hour of Day")
        plt.ylabel("Risk Rate (Normalized)")
        plt.xticks(range(0, 24))
        plt.grid(axis='y', alpha=0.3)
        self._save_plot(filename)

        plt.show()

    def plot_severity_correlation(self, filename="plot_severity_correlation"):
        print("Integrating Plot 4: Severity Correlation...")
        
        query = """
            WITH monthly_trips AS (
                SELECT date_trunc('month', started_at) as mo, count(*) as trip_count 
                FROM NYC_MASTER GROUP BY 1
            ),
            monthly_injuries AS (
                SELECT date_trunc('month', crash_date) as mo, sum(cyclist_injured) as injuries
                FROM collisions GROUP BY 1
            )
            SELECT 
                t.mo, 
                (t.trip_count / 1000000.0) as trips_millions, 
                i.injuries
            FROM monthly_trips t
            JOIN monthly_injuries i ON t.mo = i.mo
            ORDER BY t.mo
        """
        df = self.con.execute(query).df()
        
        # 1. Create a readable Date String for the X-axis labels
        df['mo_str'] = df['mo'].dt.strftime('%b %Y')

        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax2 = ax1.twinx()

        # 2. Plot Bars (Volume)
        # We use df.index to ensure the bar and line share the exact same X-coordinates
        sns.barplot(data=df, x='mo_str', y='trips_millions', ax=ax1, 
                    color='skyblue', alpha=0.4, label='Trip Volume')
        
        # 3. Plot Line (Injuries)
        # Important: Lineplot must use the same categorical X values (mo_str) as the barplot
        sns.lineplot(data=df, x='mo_str', y='injuries', ax=ax2, 
                    color='darkred', marker='o', linewidth=3, markersize=8, label='Cyclist Injuries')

        # --- X-AXIS READABILITY IMPROVEMENTS ---
        # Rotate labels and only show every 2nd or 3rd month if the timeline is long
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Limit the number of ticks if the list is too crowded
        n = 2  # Show every 2nd label
        [l.set_visible(False) for (i, l) in enumerate(ax1.get_xticklabels()) if i % n != 0]

        # --- AXIS DECORATION ---
        ax1.set_ylabel("CitiBike Trips (Millions)", color='steelblue', fontsize=12, fontweight='bold')
        ax2.set_ylabel("Total Cyclist Injuries", color='darkred', fontsize=12, fontweight='bold')
        
        ax1.set_xlabel("Timeline", fontsize=12)
        plt.title("System Exposure vs. Risk: Monthly Volume & Injury Correlation", 
                fontsize=16, fontweight='bold', pad=20)

        # Clean formatting for Y-axis (adding commas to injuries)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

        # Synchronize legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True)

        sns.despine(right=False)
        plt.tight_layout()
        self._save_plot(filename)

        plt.show()

    def plot_risk_quadrant(self, filename="plot_risk_quadrant"):
        # ... (Same Data Retrieval as before) ...
        """
        Final Synthesis: Maps Frequency (Crashes/Trips) vs Severity (Injuries/Crash).
        This identifies the specific time-slots that are most 'expensive' for insurers.
        """
        # 1. Get Crash Data
        df_crashes = self.q_to_df("""
            SELECT 
                dayname(crash_date) as dow_name,
                hour(crash_time) as hr,
                COUNT(*) as crash_count,
                AVG(cyclist_injured) as severity_rate
            FROM collisions 
            GROUP BY 1, 2
        """)
        
        # 2. Get CitiBike Data (Exposure)
        df_trips = self.q_to_df("""
            SELECT 
                dayname(started_at) as dow_name,
                hour(started_at) as hr,
                COUNT(*) as trip_count
            FROM NYC_MASTER
            GROUP BY 1, 2
        """)

        # 3. Merge and Calculate Frequency
        df_combined = pd.merge(df_crashes, df_trips, on=['dow_name', 'hr'])
        
        # Normalizing: Crashes per 1,000 trips
        df_combined['frequency_rate'] = (df_combined['crash_count'] / df_combined['trip_count']) * 1000

        # 4. Professional Visualization Improvements
        plt.figure(figsize=(14, 9), facecolor='#f8f9fa') # Light gray background for contrast
        #sns.set_style("whitegrid", {'grid.linestyle': '--'})
        
        # Each dot is an Hour of a specific Day
        ax = sns.scatterplot(
            data=df_combined, 
            x='frequency_rate', 
            y='severity_rate', 
            hue='hr', 
            palette='coolwarm', 
            size='trip_count', 
            sizes=(50, 600), # Increased bubble size for visibility
            alpha=0.7,
            edgecolor='white',
            linewidth=0.5
        )

        # Professional Median Lines
        freq_median = df_combined['frequency_rate'].median()
        sev_median = df_combined['severity_rate'].median()
        plt.axvline(freq_median, color='#34495e', linestyle='--', linewidth=2, alpha=0.6)
        plt.axhline(sev_median, color='#34495e', linestyle='--', linewidth=2, alpha=0.6)

        # 5. IMPROVED LABEL VISIBILITY (Using bounding boxes for readability)
        label_style = dict(fontweight='bold', fontsize=13, va='center')
        
        # High Risk Label (Top Right)
        plt.text(df_combined['frequency_rate'].max() * 0.75, df_combined['severity_rate'].max() * 0.95, 
                "HIGH RISK\n(Frequent & Severe)", color='#c0392b', ha='center', **label_style,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5))
        
        # Hidden Cost Label (Top Left)
        plt.text(df_combined['frequency_rate'].min() + 0.5, df_combined['severity_rate'].max() * 0.95, 
                "HIDDEN COST\n(Rare but Severe)", color='#d35400', ha='left', **label_style,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5))

        plt.title("NYC Cyclist Risk: Frequency vs. Severity Strategic Matrix", fontsize=18, fontweight='bold', pad=25)
        plt.xlabel("Frequency (Crashes per 1,000 Trips)", fontsize=13, labelpad=12)
        plt.ylabel("Severity (Average Injuries per Crash)", fontsize=13, labelpad=12)
        
        # Enhanced Legend
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Hour / Bubble = Volume", 
                title_fontsize='11', fontsize='10', frameon=True)
        
        plt.tight_layout()
        self._save_plot(filename)

        plt.show()

    def plot_segment_exposure_risk(self, filename="plot_segment_exposure_risk"):
        # 1. Get Vehicle Lethality (Severity)
        lethality_query = """
            SELECT 
                CASE 
                    WHEN vehicle_1 IN ('SPORT UTILITY / STATION WAGON', 'SUV') THEN 'SUV / Wagon'
                    WHEN vehicle_1 IN ('PICK-UP TRUCK', 'VAN', 'Box Truck') THEN 'Trucks & Vans'
                    WHEN vehicle_1 IN ('PASSENGER VEHICLE', 'Sedan') THEN 'Passenger Car'
                    WHEN vehicle_1 IN ('TAXI', 'Livery Vehicle') THEN 'Taxi / Uber'
                    WHEN vehicle_1 IN ('Bus', 'BUS') THEN 'Bus / Heavy'
                    ELSE 'Other' 
                END AS vehicle_class,
                SUM(cyclist_killed) * 10000.0 / NULLIF(COUNT(*), 0) as lethality_index
            FROM collisions
            WHERE vehicle_1 IS NOT NULL
            GROUP BY 1
        """
        
        # 2. Get Hourly Volume by Segment (Exposure)
        exposure_query = """
            SELECT 
                CASE 
                    WHEN LOWER(member_casual) IN ('member', 'subscriber') THEN 'Member'
                    ELSE 'Casual'
                END as user_segment,
                COUNT(*) as trip_count
            FROM NYC_MASTER
            GROUP BY 1
        """

        df_lethality = self.q_to_df(lethality_query)
        df_exposure = self.q_to_df(exposure_query)

        # 3. Probabilistic Risk Calculation
        # We weight the vehicle lethality by the segment's share of total trips
        total_trips = df_exposure['trip_count'].sum()
        df_exposure['share'] = df_exposure['trip_count'] / total_trips
        
        matrix_data = []
        for _, v_row in df_lethality.iterrows():
            for _, e_row in df_exposure.iterrows():
                # Risk = (Vehicle Lethality) * (User Segment's Exposure Share)
                risk_score = v_row['lethality_index'] * e_row['share']
                matrix_data.append({
                    'vehicle_class': v_row['vehicle_class'],
                    'user_segment': e_row['user_segment'],
                    'risk_score': risk_score
                })

        df_final = pd.DataFrame(matrix_data).pivot(index='vehicle_class', columns='user_segment', values='risk_score')

        # 4. Corrected Visualization (Removed hue/legend)
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_final, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={'label': 'Calculated Risk Score'})
        
        plt.title("Segmented Threat Matrix: Actuarial Risk by User Type", fontsize=15, fontweight='bold')
        plt.xlabel("CitiBike User Segment")
        plt.ylabel("Involved Vehicle Type")
        plt.tight_layout()
        self._save_plot(filename)

        plt.show()

    def plot_borough_exposure_gap(self, filename="plot_borough_exposure_gap"):
        print("Integrating Plot 2: Borough Exposure (Spatial Intersection)...")
        # Use the conjunction table which already has borough mapping
        query = """
            SELECT 
                borough,
                SUM(trip_count) / 1000.0 as trips_k,
                SUM(crash_count) as total_crashes
            FROM station_safety_index
            WHERE borough IS NOT NULL
            GROUP BY 1
            ORDER BY trips_k DESC
        """
        df = self.q_to_df(query)
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()

        # Bar plot for volume
        sns.barplot(data=df, x='borough', y='trips_k', ax=ax1, color='skyblue', alpha=0.6, label='Trips (k)')
        # Line plot for crashes
        sns.lineplot(data=df, x='borough', y='total_crashes', ax=ax2, color='darkred', marker='o', linewidth=3, label='Crashes')

        ax1.set_ylabel("CitiBike Trips (Thousands)", color='steelblue', fontsize=12, fontweight='bold')
        ax2.set_ylabel("Total Recorded Crashes", color='darkred', fontsize=12, fontweight='bold')
        plt.title("Risk Gap Analysis: Borough Volume vs. Safety Index", fontsize=15, fontweight='bold')
        
        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')
        self._save_plot(filename)

        plt.show()
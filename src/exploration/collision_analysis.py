import duckdb
from src.config import Config  
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib as mpl
from scipy import stats

# Set global formatting: No scientific notation
mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['axes.formatter.limits'] = [-20, 20] 


class CollisionAnalyzer:
    def __init__(self, db_path=None, output_dir=None):
        self.db_path = Config.DB_PATH        
        self.output_dir = Config.OUTPUT_DIR_COLLISION
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

    def check_data_quality(self):
        """Audits missing values across critical insurance dimensions."""
        return self.q_to_df("""
            SELECT 
                count(*) as total_records,
                sum(CASE WHEN latitude IS NULL THEN 1 ELSE 0 END) * 100.0 / count(*) as pct_missing_geo,
                sum(CASE WHEN borough IS NULL THEN 1 ELSE 0 END) * 100.0 / count(*) as pct_missing_borough,
                sum(CASE WHEN factor_1 = 'Unspecified' THEN 1 ELSE 0 END) * 100.0 / count(*) as pct_unspecified_cause
            FROM collisions
        """)

    def perform_actuarial_statistics(self):
        """
        Final Synthesis: Maps Frequency (Crashes/Trips) vs Severity (Injuries/Crash).
        This identifies the specific time-slots that are most 'expensive' for insurers.
        """

        df_crashes = self.q_to_df("""
            SELECT 
                dayname(crash_date) as dow_name,
                hour(crash_time) as hr,
                COUNT(*) as crash_count,
                AVG(cyclist_injured) as severity_rate
            FROM collisions 
            GROUP BY 1, 2
        """)
        
        df_trips = self.q_to_df("""
            SELECT 
                dayname(started_at) as dow_name,
                hour(started_at) as hr,
                COUNT(*) as trip_count
            FROM NYC_MASTER
            GROUP BY 1, 2
        """)

        df_combined = pd.merge(df_crashes, df_trips, on=['dow_name', 'hr'])
        
        # TEST A: Correlation (Exposure vs. Frequency)
        # Goal: Does more traffic = more crashes?
        corr, p_val_corr = stats.pearsonr(df_combined['trip_count'], df_combined['crash_count'])

        # TEST B: Welch’s T-test (Day vs. Night Severity)
        # Goal: Is Night (19:00-05:00) significantly more dangerous than Day?
        df_combined['is_night'] = df_combined['hr'].apply(lambda x: 1 if (x >= 19 or x <= 5) else 0)
        night_sev = df_combined[df_combined['is_night'] == 1]['severity_rate']
        day_sev = df_combined[df_combined['is_night'] == 0]['severity_rate']
        t_stat, p_val_t = stats.ttest_ind(night_sev, day_sev, equal_var=False)

        # TEST C: R-Squared (Predictive Power)
        # Goal: How much of the crash frequency is explained ONLY by volume?
        slope, intercept, r_value, _, _ = stats.linregress(df_combined['trip_count'], df_combined['crash_count'])
        r_sq = r_value**2

        print(f"--- ACTUARIAL STATISTICAL SUMMARY ---")
        print(f"1. Volume-Frequency Correlation: {corr:.3f} (p={p_val_corr:.4f})")
        print(f"2. Night/Day Severity T-Stat: {t_stat:.3f} (p={p_val_t:.4f})")
        print(f"3. R-Squared (Exposure impact): {r_sq:.3f}")

    def plot_vehicle_lethality(self, filename="plot_vehicle_lethality"):
        query = """
            SELECT 
                CASE 
                    WHEN vehicle_1 IN ('SPORT UTILITY / STATION WAGON', 'SUV') THEN 'SUV / Wagon'
                    WHEN vehicle_1 IN ('PICK-UP TRUCK', 'VAN', 'Box Truck') THEN 'Trucks & Vans'
                    WHEN vehicle_1 IN ('PASSENGER VEHICLE', 'Sedan') THEN 'Passenger Car'
                    WHEN vehicle_1 IN ('TAXI', 'Livery Vehicle') THEN 'Taxi / Uber'
                    WHEN vehicle_1 IN ('Bus', 'BUS') THEN 'Bus / Heavy'
                    ELSE 'Other' 
                END AS vehicle_group, 
                COUNT(*) as total_crashes,
                SUM(cyclist_killed) as total_deaths,
                (SUM(cyclist_killed) * 10000.0 / NULLIF(COUNT(*), 0)) as lethality_index
            FROM collisions 
            WHERE vehicle_1 IS NOT NULL
            GROUP BY 1 
            HAVING total_crashes > 100
            ORDER BY 4 DESC
        """
        df = self.q_to_df(query)

        plt.figure(figsize=(12, 8))
        sns.set_style("white") 
        
        # Create the horizontal bar chart
        ax = sns.barplot(data=df, x='lethality_index', y='vehicle_group', palette='RdYlBu_r')

        formula_text = (
            "CALCULATION METHODOLOGY:\n"
            "Index = (Total Deaths / Total Crashes) * 10,000\n"
            "This represents the statistical likelihood of a fatality\n"
            "per 10,000 collision events for each vehicle category."
        )

        plt.text(0.95, 0.05, formula_text, transform=ax.transAxes, fontsize=6,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        max_width = df['lethality_index'].max()
        
        for i, p in enumerate(ax.patches):
            val = p.get_width()
            deaths = int(df.iloc[i]['total_deaths'])
            crashes = int(df.iloc[i]['total_crashes'])
            
            # Create a clean label
            label_text = f" {val:.2f} ({deaths} deaths / {crashes:,} crashes)"
            
            if val < (max_width * 0.60):
                # Place text to the right of the bar
                ax.text(val + (max_width * 0.01), p.get_y() + p.get_height()/2, 
                        label_text, va='center', ha='left', fontsize=10, color='#333333', fontweight='bold')
            else:
                # Place text inside the end of the bar in white
                ax.text(val - (max_width * 0.01), p.get_y() + p.get_height()/2, 
                        label_text, va='center', ha='right', fontsize=10, color='white', fontweight='bold')

        plt.title("NYC Collision Lethality by Vehicle Category", fontsize=16, fontweight='bold', loc='left', pad=20)
        plt.xlabel("Lethality Index (Deaths per 10k Events)", fontsize=12, labelpad=10)
        plt.ylabel("") 
        
        sns.despine(left=True, bottom=False)
        plt.tight_layout()
        self._save_plot(filename)
        plt.show()

    def plot_weekend_risk_distribution(self, filename="plot_weekend_risk_distribution"):
        # Daily averages to see the SPREAD of risk
        query = """
            SELECT 
                crash_date,
                CASE WHEN dayofweek(crash_date) IN (0, 6) THEN 'Weekend' ELSE 'Weekday' END as day_type,
                SUM(cyclist_injured) * 1.0 / COUNT(*) as daily_injury_rate
            FROM collisions 
            GROUP BY 1, 2
            HAVING COUNT(*) > 5
        """
        df = self.q_to_df(query)

        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        sns.violinplot(data=df, x='day_type', y='daily_injury_rate', 
                    palette="Set2", inner="quartile", bw_adjust=.5)
        
        plt.title("The Weekend Volatility: Injury Probability Distribution", fontsize=14, pad=15)
        plt.ylabel("Injuries per Collision (Daily Avg)")
        plt.xlabel("Day Classification")
        
        plt.annotate('Wider "Bulge" or "Tail" indicates\nhigher unpredictability (Risk)', 
                    xy=(0.5, 0.2), xytext=(0.6, 0.3),
                    arrowprops=dict(facecolor='black', arrowstyle='->'))
        
        plt.tight_layout()
        self._save_plot(filename) 
        plt.show()

    def plot_risk_heatmap(self, filename="plot_risk_heatmap"):
        query = """
            SELECT 
                dayname(crash_date) as dow_name,
                dayofweek(crash_date) as dow,
                hour(crash_time) as hr,
                AVG(cyclist_injured) as risk_score
            FROM collisions 
            GROUP BY 1, 2, 3
            ORDER BY dow, hr
        """

        query2 = """
            SELECT hour(crash_time) as hr, dayname(crash_date) as day, count(*) as c FROM collisions GROUP BY 1, 2
        """

        df = self.q_to_df(query)
        df2 = self.q_to_df(query2)

        # Pivot for Heatmap format
        pivot_df = df.pivot(index="hr", columns="dow_name", values="risk_score")

        # Ensure days are in correct order
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_df = pivot_df[days_order]

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, cmap="YlOrRd", annot=False, cbar_kws={'label': 'Avg Injury Severity'})
        
        plt.title("Temporal Risk Matrix: When are Cyclists most at Risk?", fontsize=16)
        plt.xlabel("Day of Week")
        plt.ylabel("Hour of Day (24h)")
        self._save_plot(filename+"_severity")
        plt.show()

        pivot = df2.pivot(index="hr", columns="day", values="c")
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, cmap="YlOrRd").set_title("Crash Intensity by Hour/Day")
        self._save_plot(filename+"_intensity")
        plt.show()

    
    def plot_danger_streets(self, filename="plot_danger_streets"):
        # Cleaning Broadway and other duplicates using TRIM and UPPER
        query = """
            SELECT 
                TRIM(UPPER(on_street)) as street_name, 
                SUM(cyclist_injured) as total_injured,
                COUNT(*) as total_crashes,
                (SUM(cyclist_injured) * 100.0 / NULLIF(COUNT(*), 0)) as injury_probability
            FROM collisions 
            WHERE on_street IS NOT NULL 
            AND TRIM(on_street) != ''
            GROUP BY 1 
            ORDER BY 2 DESC 
            LIMIT 10
        """
        df3 = self.q_to_df(query)

        plt.figure(figsize=(12, 7))
        sns.set_style("white")
        
        ax = sns.barplot(data=df3, x='total_injured', y='street_name', palette='flare')
        
        # Corrected Annotation Logic
        for i, p in enumerate(ax.patches):
            prob = df3.iloc[i]['injury_probability']
            count = int(p.get_width())
            
            ax.annotate(f"{count} injured ({prob:.1f}% risk/crash)", 
                    (p.get_width(), p.get_y() + p.get_height()/2),
                    xytext=(7, 0), 
                    textcoords='offset points', 
                    va='center', 
                    fontsize=10, 
                    fontweight='bold')

        plt.title("Top 10 High-Liability Corridors (Total Cyclist Injuries)", fontsize=16, fontweight='bold', pad=25)
        plt.xlabel("Cumulative Injuries (Historical)", fontsize=12)
        plt.ylabel("")
        
        sns.despine(left=True, bottom=False)
        plt.tight_layout()

        self._save_plot(filename) 

        plt.show()

    def plot_crash_intensity(self, filename="plot_crash_intensity"):
        # Filter for incidents where at least one person was hurt or killed
        df4 = self.q_to_df("""
            SELECT cyclist_injured, cyclist_killed, COUNT(*) as incident_count
            FROM collisions 
            WHERE cyclist_injured > 0 OR cyclist_killed > 0
            GROUP BY 1, 2
        """)        
        
        # Pivot for a heatmap
        intensity_matrix = df4.pivot(index='cyclist_killed', columns='cyclist_injured', values='incident_count').fillna(0)

        plt.figure(figsize=(10, 8))
        sns.heatmap(intensity_matrix, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={'label': 'Number of Accidents'})
        plt.title("Collision Severity Matrix: Injuries vs. Fatalities", fontsize=15, pad=20)
        plt.xlabel("Number of Cyclists Injured")
        plt.ylabel("Number of Cyclists Killed")
        plt.text(0.5, -0.1, "Note: Data filtered to show only incidents with casualties.\nHigh-value cells represent common claim scenarios.", 
                ha='center', transform=plt.gca().transAxes, color='gray', fontsize=10)
        
        self._save_plot(filename) 
        plt.show()

    def plot_hourly_rush(self, filename="plot_hourly_rush"):
        df5 = self.q_to_df("SELECT hour(crash_time) as hr, count(*) as count FROM collisions GROUP BY 1 ORDER BY 1")
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(df5['hr'], df5['count'], color="skyblue", alpha=0.3, label='Crash Volume')
        plt.plot(df5['hr'], df5['count'], color="Slateblue", marker='o', linewidth=2)
        plt.axvspan(7, 10, color='orange', alpha=0.1, label='Morning Rush')
        plt.axvspan(16, 19, color='red', alpha=0.1, label='Evening Rush')
        plt.title("When Do Crashes Happen? (24-Hour Risk Profile)", fontsize=15, fontweight='bold')
        plt.xlabel("Hour of Day (Military Time)")
        plt.ylabel("Total Number of Collisions")
        plt.xticks(range(0, 24))
        plt.legend(loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        self._save_plot(filename) 
        plt.show()

    def plot_contributing_factors(self, filename="plot_contributing_factors"):
        df6 = self.q_to_df("""
            SELECT factor_1, SUM(cyclist_injured) as injuries, 
                SUM(cyclist_injured) * 100.0 / (SELECT SUM(cyclist_injured) FROM collisions) as pct
            FROM collisions 
            WHERE factor_1 NOT IN ('Unspecified', 'Unknown')
            GROUP BY 1 ORDER BY 2 DESC LIMIT 8
        """)
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=df6, x='injuries', y='factor_1', hue='factor_1', palette='magma', legend=False)
        
        # Add percentage labels to show "Contribution to Total Risk"
        for i, p in enumerate(ax.patches):
            pct = df6.iloc[i]['pct']
            ax.annotate(f'{pct:.1f}% of all injuries', (p.get_width(), p.get_y() + p.get_height()/2),
                    xytext=(5, 0), textcoords='offset points', va='center', fontweight='bold')

        plt.title("Primary Causes of Cyclist Injuries", fontsize=15, fontweight='bold')
        plt.xlabel("Total Injuries (Aggregated)")
        plt.ylabel("")
        self._save_plot(filename) 

        plt.show()
        
    def plot_borough_severity(self, filename="plot_borough_severity"):
        # Mean injury rate
        df7 = self.q_to_df("""
            SELECT borough, 
                cyclist_injured,
                CASE WHEN cyclist_injured > 0 THEN 1 ELSE 0 END as injury_occurred
            FROM collisions 
            WHERE borough IS NOT NULL
        """)
        
        plt.figure(figsize=(10, 6))
        
        sns.pointplot(
            data=df7,
            x='borough',
            y='cyclist_injured',
            hue='borough',
            palette='Set1',
            linestyle='none',
            legend=False
        )
        
        plt.title("Insurance Risk Rating: Avg. Injury Severity by Borough", fontsize=15, fontweight='bold')
        plt.ylabel("Mean Injuries per Collision")
        plt.xlabel("Borough")
        
        # Add a horizontal line for the city-wide average for comparison
        city_avg = df7['cyclist_injured'].mean()
        plt.axhline(city_avg, color='gray', linestyle='--', label=f'City Average ({city_avg:.3f})')
        plt.legend()
        plt.text(0.5, -0.2, "Vertical lines represent the confidence interval.\nBoroughs above the dashed line are 'High Risk' zones.", 
                ha='center', transform=plt.gca().transAxes, fontsize=10, color='red')
        
        self._save_plot(filename) 
        plt.show()

    def plot_seasonal_trends(self, filename="plot_seasonal_trends"):
        df8 = self.q_to_df("SELECT month(crash_date) as mo, count(*) as count FROM collisions GROUP BY 1 ORDER BY 1")
        
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")
        
        # Draw the main trend line
        sns.lineplot(data=df8, x='mo', y='count', marker='o', color='crimson', linewidth=3, markersize=8)
        
        # Add Seasonal Shading
        plt.axvspan(6, 8, color='orange', alpha=0.1, label='Peak Summer Riding')
        plt.axvspan(11, 12, color='blue', alpha=0.05, label='Winter Risk')
        plt.axvspan(1, 2, color='blue', alpha=0.05)
        plt.title("NYC Seasonal Risk Profile: Monthly Collision Volume", fontsize=15, fontweight='bold')
        plt.xlabel("Month of the Year")
        plt.ylabel("Total Collisions")
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.legend(loc='upper left')
        self._save_plot(filename) 
        plt.show()

    def plot_incident_severity_comparison(self, filename="plot_incident_severity_comparison"):
        # Compare the probability of an injury occurring
        df10 = self.q_to_df("""
            SELECT 
                CASE WHEN vehicle_1 IS NOT NULL THEN 'Vehicle Identified' ELSE 'Unidentified/Hit-Run' END as status,
                AVG(cyclist_injured) * 100 as injury_pct
            FROM collisions 
            GROUP BY 1
        """)
        
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(data=df10, x='status', y='injury_pct', hue='status', palette=['#95a5a6', '#e74c3c'], legend=False)
        
        # Add labels
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}% Risk', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontweight='bold')

        plt.title("Claim Probability: Does Vehicle Identification Matter?", fontsize=14, fontweight='bold')
        plt.ylabel("Avg % Chance of Injury per Crash")
        plt.xlabel("")
        plt.ylim(0, max(df10['injury_pct']) * 1.2) 
        self._save_plot(filename) 

        plt.show()


    def plot_zip_hotspots_named(self, filename="plot_zip_hotspots_named"):

        ZIP_TO_NAME = {
        # Brooklyn
        "11201": "Dumbo/Downtown", "11203": "East Flatbush", "11205": "Bed-Stuy",
        "11206": "Williamsburg (S)", "11207": "East New York", "11208": "Cypress Hills",
        "11211": "Williamsburg (N)", "11212": "Brownsville", "11213": "Crown Heights",
        "11215": "Park Slope", "11216": "Bed-Stuy (W)", "11217": "Boerum Hill",
        "11220": "Sunset Park", "11221": "Bushwick", "11226": "Flatbush",
        "11233": "Stuy-Heights", "11235": "Sheepshead Bay", "11236": "Canarsie",
        "11237": "Bushwick (N)", "11238": "Clinton Hill",
        # Queens
        "11101": "Long Island City", "11354": "Flushing", "11368": "Corona", 
        "11372": "Jackson Heights", "11373": "Elmhurst", "11377": "Woodside",
        "11385": "Ridgewood", "11432": "Jamaica", "11434": "South Jamaica",
        # Manhattan
        "10002": "Lower East Side", "10003": "East Village", "10009": "Alphabet City",
        "10013": "Tribeca/Soho", "10019": "Midtown", "10025": "Upper West Side",
        "10027": "Manhattanville", "10029": "East Harlem", "10031": "Hamilton Heights",
        "10032": "Washington Heights",
        # Bronx
        "10451": "Concourse", "10452": "Highbridge", "10453": "Morris Heights",
        "10454": "Mott Haven", "10455": "Melrose", "10456": "Morrisania",
        "10457": "Tremont", "10458": "Belmont", "10467": "Williamsbridge",
        "10468": "University Heights",
        "11234": "Flatlands / Mill Basin",
        "10016": "Murray Hill / Kips Bay",
        "10036": "Hell's Kitchen / Midtown",
        "10022": "Midtown East / Sutton Pl"
    }
        
        df9 = self.q_to_df("""
            SELECT CAST(zip_code AS VARCHAR) as zip, count(*) as count 
            FROM collisions WHERE zip_code IS NOT NULL AND zip_code != '0'
            GROUP BY 1 ORDER BY 2 DESC LIMIT 15
        """)

        # Apply the mapping
        df9['neighborhood'] = df9['zip'].map(ZIP_TO_NAME).fillna("Unknown/Mixed")
        
        # Combine Zip and Name for the label: "11211 (Williamsburg)"
        df9['full_label'] = df9['zip'] + "\n(" + df9['neighborhood'] + ")"

        plt.figure(figsize=(15, 7))
        sns.set_style("white")
        
        ax = sns.barplot(data=df9, x='full_label', y='count', palette="Reds_r")

        plt.title("NYC Danger Zones: Top 15 Hotspots by Neighborhood", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Zip Code & Neighborhood", fontsize=12)
        plt.ylabel("Total Collisions", fontsize=12)
        plt.xticks(rotation=0, fontsize=7)
        
        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), 
                    textcoords='offset points', fontweight='bold', color='black')

        sns.despine()
        plt.tight_layout()
        self._save_plot(filename) 
        plt.show()
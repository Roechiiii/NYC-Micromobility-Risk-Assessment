import duckdb
from src.config import Config  
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import os
import contextily as cx
import geopandas as gpd
from shapely.geometry import Point

import matplotlib as mpl
# Set global formatting: No scientific notation, use commas
mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['axes.formatter.limits'] = [-20, 20] # Effectively disables scientific notation

class CitiBikeAnalyzer:

    def __init__(self, db_path=None, output_dir=None):
        self.db_path = Config.DB_PATH        
        self.output_dir = Config.OUTPUT_DIR_CITIBIKE
        os.makedirs(self.output_dir, exist_ok=True)
        self.con = None

    def __enter__(self):
        self.con = duckdb.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.con: self.con.close()

    def q_to_df(self, sql):
        """Standardizes DuckDB output to lowercase for Seaborn/Pandas compatibility."""
        df = self.con.execute(sql).df()
        df.columns = [c.lower() for c in df.columns]
        return df

    def _save_plot(self, filename: str):
        """Internal helper to standardize how plots are saved."""
        if filename:
            if not filename.endswith(('.png', '.jpg', '.pdf')):
                filename += '.png'
            
            save_path = self.output_dir / filename
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print("Plot saved!")

    def plot_monthly_volume(self, filename="plot_monthly_volume"):
        df1 = self.con.execute("""
            SELECT date_trunc('month', started_at) as month, count(*) as rides 
            FROM NYC_MASTER GROUP BY 1 ORDER BY 1
        """).df()
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(df1['month'], df1['rides'], color="skyblue", alpha=0.3)
        plt.plot(df1['month'], df1['rides'], color="navy", marker='o', linewidth=2, markersize=4)
        plt.ticklabel_format(style='plain', axis='y', useOffset=False)
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

        avg_rides = df1['rides'].mean()
        plt.axhline(avg_rides, color='red', linestyle='--', alpha=0.6, label=f'Avg: {int(avg_rides):,}')
        plt.title("CitiBike System Growth: Total Monthly Trips", fontsize=15, fontweight='bold', loc='left')
        plt.ylabel("Trip Count (Actual Numbers)")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        sns.despine()
        self._save_plot(filename)

        plt.show()

    def plot_monthly_volume_by_user(self, filename="plot_monthly_volume_by_user"):
        # SQL: Standardizing the user types during the pull
        query = """
            SELECT 
                date_trunc('month', started_at) as month,
                CASE 
                    WHEN LOWER(member_casual) IN ('member', 'subscriber') THEN 'Member'
                    ELSE 'Casual'
                END as user_type,
                count(*) as rides 
            FROM NYC_MASTER 
            GROUP BY 1, 2 
            ORDER BY 1
        """
        df1 = self.con.execute(query).df()
        
        # Pivot so we have separate columns for 'Member' and 'Casual'
        df_pivot = df1.pivot(index='month', columns='user_type', values='rides').fillna(0)

        plt.figure(figsize=(14, 7))
        sns.set_style("whitegrid", {'axes.grid': False})

        # Colors for our two classes
        colors = {'Member': '#2ecc71', 'Casual': '#3498db'}

        for user_type in df_pivot.columns:
            # 1. Fill the area under the line
            plt.fill_between(df_pivot.index, df_pivot[user_type], 
                            color=colors[user_type], alpha=0.15)
            
            # 2. Plot the line with dots (markers)
            plt.plot(df_pivot.index, df_pivot[user_type], 
                    marker='o', markersize=5, linewidth=2.5, 
                    color=colors[user_type], label=user_type)

        # 3. Add a "Total" baseline (Optional, but good for context)
        total_avg = df_pivot.sum(axis=1).mean()
        plt.axhline(total_avg, color='gray', linestyle=':', alpha=0.5, label='Combined Avg')

        # Formatting for professional look
        plt.title("Ride Volume Trends: Member vs. Casual Riders", fontsize=16, fontweight='bold', loc='left')
        plt.ylabel("Monthly Trip Count")
        plt.xlabel("Timeline")
        
        # Place legend outside or in a clear spot
        plt.legend(frameon=True, facecolor='white', edgecolor='none')
        
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        sns.despine()
        plt.tight_layout()
        self._save_plot(filename)

        plt.show()

    def plot_monthly_volume_with_group_avgs(self, filename="plot_monthly_volume_with_group_avgs"):
        query = """
            SELECT 
                date_trunc('month', started_at) as month,
                CASE 
                    WHEN LOWER(member_casual) IN ('member', 'subscriber') THEN 'Member'
                    ELSE 'Casual'
                END as user_type,
                count(*) as rides 
            FROM NYC_MASTER 
            GROUP BY 1, 2 
            ORDER BY 1
        """
        df1 = self.con.execute(query).df()
        df_pivot = df1.pivot(index='month', columns='user_type', values='rides').fillna(0)

        plt.figure(figsize=(14, 7))
        colors = {'Member': '#2ecc71', 'Casual': '#3498db'}

        for user_type in df_pivot.columns:
            # Plot the main data
            plt.fill_between(df_pivot.index, df_pivot[user_type], color=colors[user_type], alpha=0.1)
            plt.plot(df_pivot.index, df_pivot[user_type], marker='o', markersize=4, 
                    linewidth=2, color=colors[user_type], label=f'{user_type} Volume')
            
            # ADD GROUP-SPECIFIC AVERAGE
            group_avg = df_pivot[user_type].mean()
            plt.axhline(group_avg, color=colors[user_type], linestyle='--', alpha=0.5, 
                        label=f'{user_type} Avg: {int(group_avg):,}')

        plt.title("Growth Analysis: Monthly Volume with Group Baselines", fontsize=16, fontweight='bold')
        plt.ylabel("Trip Count")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        sns.despine()
        plt.tight_layout()
        self._save_plot(filename)

        plt.show()

    def plot_popular_corridors(self, filename="plot_popular_corridors"):
        # This finds the top 10 unique paths traveled
        query = """
            SELECT 
                start_station_name || ' TO ' || end_station_name as corridor,
                count(*) as trip_count
            FROM NYC_MASTER
            WHERE start_station_name IS NOT NULL 
            AND end_station_name IS NOT NULL
            AND start_station_name != end_station_name
            GROUP BY 1
            ORDER BY 2 DESC
            LIMIT 10
        """
        df9 = self.con.execute(query).df()

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=df9, x='trip_count', y='corridor', hue='corridor', palette='viridis', legend=False)
        
        plt.title("Top 10 High-Traffic Corridors (User Flow)", fontsize=15, fontweight='bold')
        plt.xlabel("Number of Trips")
        plt.ylabel("")
        sns.despine()
        self._save_plot(filename)

        plt.show()

    def plot_hourly_demand(self, filename="plot_hourly_demand"):
        df2 = self.con.execute("""
            SELECT hour(started_at) as hour, count(*) as rides 
            FROM NYC_MASTER GROUP BY 1 ORDER BY 1
        """).df()
        
        plt.figure(figsize=(12, 6))
        # Use hue to avoid the deprecation warning
        ax = sns.barplot(data=df2, x='hour', y='rides', hue='hour', palette='Blues_d', legend=False)
        
        # Highlight Rush Hours
        plt.axvspan(7.5, 9.5, color='orange', alpha=0.15, label='AM Rush')
        plt.axvspan(16.5, 18.5, color='orange', alpha=0.15, label='PM Rush')
        
        plt.title("Hourly Demand Profile: System Utilization Peaks", fontsize=15, fontweight='bold')
        plt.xlabel("Hour of Day (24h)")
        plt.ylabel("Total Trip Starts")
        plt.legend()
        sns.despine()
        self._save_plot(filename)

        plt.show()

    def plot_user_composition(self, filename="plot_user_composition"):
        # Standardize names in SQL to avoid 'Subscriber/Customer' confusion
        query = """
            SELECT 
                CASE 
                    WHEN LOWER(member_casual) IN ('member', 'subscriber') THEN 'Annual Member'
                    WHEN LOWER(member_casual) IN ('casual', 'customer') THEN 'Casual Rider'
                    ELSE 'Other'
                END as user_type,
                count(*) as count 
            FROM NYC_MASTER 
            GROUP BY 1
        """
        df3 = self.con.execute(query).df()
        
        # Use the new column name 'user_type'
        num_categories = len(df3)
        explode_values = [0.05] + [0] * (num_categories - 1)
        colors = ['#2ecc71', '#3498db', '#95a5a6'][:num_categories]
        
        plt.figure(figsize=(8, 8))
        plt.pie(
            df3['count'], 
            labels=df3['user_type'],
            autopct='%1.1f%%', 
            colors=colors, 
            startangle=140, 
            pctdistance=0.85, 
            explode=explode_values
        )
        
        # Donut hole
        plt.gca().add_artist(plt.Circle((0,0), 0.70, fc='white'))
        
        total_trips = df3['count'].sum()
        plt.text(0, 0, f'Total Trips\n{total_trips:,}', ha='center', va='center', fontsize=12, fontweight='bold')
        plt.title("User Segment Composition", fontsize=15, fontweight='bold')
        self._save_plot(filename)

        plt.show()

    def plot_duration_distribution(self, filename="plot_duration_distribution"):
        # Use date_diff to get the difference in seconds, then divide by 60
        # Or use epoch(ended_at - started_at) / 60
        df4 = self.con.execute("""
            SELECT 
                date_diff('second', started_at, ended_at) / 60.0 as minutes 
            FROM NYC_MASTER 
            WHERE started_at IS NOT NULL 
            AND ended_at IS NOT NULL
            AND minutes > 1 
            AND minutes < 60 
            USING SAMPLE 1%
        """).df()

        plt.figure(figsize=(12, 6))
        sns.histplot(df4['minutes'], bins=60, kde=True, color='orange', edgecolor='white')

        # Add pricing threshold lines
        plt.axvline(30, color='red', linestyle='--', label='Member Free Limit (30m)')
        #plt.axvline(45, color='darkred', linestyle=':', label='Casual/E-Bike Limit (45m)')

        # --- Formatting for "Real Numbers" ---
        plt.ticklabel_format(style='plain', axis='y')
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        
        plt.title("CitiBike Usage: Ride Duration Distribution", fontsize=15, fontweight='bold')
        plt.xlabel("Minutes per Trip")
        plt.ylabel("Frequency (Sampled 1%)")
        plt.legend()
        sns.despine()
        self._save_plot(filename)

        plt.show()

    def plot_day_of_week(self, filename="plot_day_of_week"):
        df5 = self.con.execute("""
            SELECT strftime(started_at, '%A') as day, count(*) as rides 
            FROM NYC_MASTER 
            GROUP BY 1
        """).df()
        
        # Ensure correct order for the plot
        cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df5['day'] = pd.Categorical(df5['day'], categories=cats, ordered=True)
        df5 = df5.sort_values('day')

        # Color Weekends (Sat/Sun) differently
        colors = ['#34495e' if d not in ['Saturday', 'Sunday'] else '#e74c3c' for d in df5['day']]

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=df5, x='day', y='rides', palette=colors, hue='day', legend=False)

        # Clean numbers (No scientific notation)
        plt.ticklabel_format(style='plain', axis='y')
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        
        plt.title("Weekly Utilization: Weekday vs. Weekend Demand", fontsize=15, fontweight='bold')
        plt.ylabel("Total Rides")
        plt.xlabel("")
        sns.despine()
        self._save_plot(filename)

        plt.show()
    
    def plot_top_start_stations(self, filename="plot_top_start_stations"):
        df6 = self.con.execute("""
            SELECT start_station_name, count(*) as count 
            FROM NYC_MASTER 
            WHERE start_station_name IS NOT NULL 
            AND start_station_name != ''
            GROUP BY 1 
            ORDER BY 2 DESC 
            LIMIT 10
        """).df()

        plt.figure(figsize=(12, 7))
        ax = sns.barplot(data=df6, y='start_station_name', x='count', hue='start_station_name', palette='Blues_r', legend=False)

        # Clean numbers for the X-axis
        plt.ticklabel_format(style='plain', axis='x')
        plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

        # Adding the labels to bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{int(df6.iloc[i]["count"]):,}', 
                    (p.get_width(), p.get_y() + p.get_height()/2),
                    xytext=(5, 0), textcoords='offset points', va='center', fontweight='bold')

        plt.title("High-Volume Nodes: Top 10 Start Stations", fontsize=16, fontweight='bold')
        plt.xlabel("Total Outbound Trips")
        sns.despine(left=True)
        self._save_plot(filename)

        plt.show()

    def plot_distance_by_user(self, filename="plot_distance_by_user"):
        # Calculate distance in km using a simplified Haversine logic in DuckDB
        query = """
            SELECT 
                CASE WHEN LOWER(member_casual) IN ('member', 'subscriber') THEN 'Member' ELSE 'Casual' END as user_type,
                haversine_distance(start_lat, start_lng, end_lat, end_lng) as distance_km
            FROM NYC_MASTER 
            WHERE start_lat IS NOT NULL AND end_lat IS NOT NULL
            AND distance_km > 0.1 AND distance_km < 15
            USING SAMPLE 5%
        """
        df7 = self.con.execute(query).df()
        
        plt.figure(figsize=(12, 6))
        sns.boxenplot(data=df7, x='user_type', y='distance_km', palette=['#2ecc71', '#3498db'])
        
        plt.title("Trip Exposure: Distance Traveled per User Type", fontsize=15, fontweight='bold')
        plt.ylabel("Distance (km)")
        plt.xlabel("")
        plt.grid(axis='y', alpha=0.3)
        sns.despine()
        self._save_plot(filename)

        plt.show()

    def plot_geo_density_with_map(self, filename="nyc_bike_density"):
            """
            Generates a high-fidelity hexbin density map of bike pickups.
            Uses a linear scale with outlier clipping for better visual variance.
            """            

            query = """
                SELECT 
                    start_lng AS lon, 
                    start_lat AS lat 
                FROM NYC_MASTER 
                WHERE start_lat IS NOT NULL 
                AND start_lng IS NOT NULL
                USING SAMPLE 100000 ROWS
            """
            df = self.con.execute(query).df()
            
            # 2. Convert to GeoDataFrame
            geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            
            # Project to Web Mercator
            gdf = gdf.to_crs(epsg=3857)

            # 3. Plotting
            fig, ax = plt.subplots(figsize=(14, 14))
            
            hb = ax.hexbin(
                gdf.geometry.x, 
                gdf.geometry.y, 
                gridsize=100,      
                cmap='inferno',    
                mincnt=1,
                alpha=0.6,
                edgecolors='none'
            )
            
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
            
            ax.set_title("NYC CitiBike: Spatial Exposure Density", fontsize=18, fontweight='bold', pad=20)
            ax.set_axis_off()
            
            cb = fig.colorbar(hb, ax=ax, shrink=0.5, label='Number of Pickups')
            
            self._save_plot(filename)
            plt.show()
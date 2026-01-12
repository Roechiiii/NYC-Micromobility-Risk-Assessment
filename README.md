# NYC Mobility Risk Assessment - CitiBike

Professional Risk Analysis of CitiBike usage and NYC collision data to identify high-liability route segments and station archetypes.

## 🚀 Overview

This repository provides a data-driven framework for pricing insurance liability in the micromobility sector. By synthesizing geospatial network analysis with probabilistic machine learning, we categorize risk across the NYC CitiBike ecosystem.

### Key Analytical Pillars
- **Network-Aware Risk**: Imputing bike routes on the NYC street graph to calculate distance-normalized risk scores.
- **Probabilistic Modeling**: Using Poisson Regression to predict collision events based on infrastructure and volume.
- **Strategic Segmentation**: K-Means clustering of stations into distinct "Insurance Tiers" for simplified actuarial pricing.

---

## 📂 Project Structure

```text
.
├── notebooks/          # Strategic report and EDA notebooks
├── outputs/            # Generated results (Plots, Risk Tables, CSVs)
├── scripts/            # Utility scripts (Ingestion, Debugging)
├── src/                # Modular Source Code
│   ├── features/       # Feature engineering & SQL aggregations
│   ├── graph/          # Network logic
│   ├── models/         # ML Model architectures (K-Means)
│   └── utils/          # Database & Configuration management
├── docs/               # Technical docs 
├── data/               # [Internal] DuckDB storage & Graph caches
├── run_analysis.py     # CORE ENTRY POINT: Main execution pipeline
└── README.md           # This document
```

---

## 🏛️ Architecture Deep Dive

The project follows a **Layered Micromobility Risk Architecture**, transitioning from physical street layouts to financial risk intelligence.

┌──────────────────────────────┐
│ Raw Data                     │
│ (CitiBike + Collisions)      │
└──────────────┬───────────────┘
               │ SQL Ingestion
               ▼
┌──────────────────────────────┐
│ Flow Layer                   │
│ (Station-to-Station Network) │
└──────────────┬───────────────┘
               │ Spatial Joins
               ▼
┌──────────────────────────────┐
│ Feature Layer                │
│ (Edge & Station Risk Tags)   │
└──────────────┬───────────────┘
               │ Probabilistic ML
               ▼
┌──────────────────────────────┐
│ Intelligence Layer           │
│ (Actuarial Tiers)            │
└──────────────────────────────┘


### Module Mission Statements
- **`src.graph`**: Manages abstract networks.
- **`src.features`**: Bridges raw data and graphs via spatial aggregation.
- **`src.models`**: Implements K-Means segmentation.
- **`src.utils`**: Provides unified configuration and DB management.

---

## 🛠️ Getting Started

### Prerequisites
- **Python**: 3.10 or higher
- **DuckDB**: Used as the primary high-performance data engine

### Installation
```bash
# Using uv (highly recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 📈 Running the Pipeline

To reproduce the full analysis from scratch:

1. **Data Ingestion**: Fetch raw CitiBike and Collision datasets.
   ```bash
   python scripts/ingest_data.py
   ```

2. **Full Analysis Pipeline**: Execute workflow.
   ```bash
   python run_analysis.py
   ```
   *This will generate all plots and risk tables in the `outputs/` directory.*

3. **Review Results**: View the generated executive report or explorative data analysis.
   ```bash
   # Open notebooks/Report.ipynb to see the final synthesized findings.
   # Open notebooks/EDA.ipynb to see the explorative data analysis.
   ```

---

## 📊 Key Outputs

- **Dynamic Spider Map**: Visualization of systemic risk flow across the NYC network.
- **High-Liability Watchlist**: Prioritized CSV of the most dangerous route segments.
- **Actuarial Tiers**: Segmentation of stations into pricing risk categories.

---
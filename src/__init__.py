"""
NYC Micromobility Risk Architecture
-----------------------------------
This package implements a multi-layer analytical framework for assessing 
insurance liability in the CitiBike ecosystem.

Module Hierarchy:
- `graph`: Handles the station-to-station trip network with risk metrics.
- `features`: Aggregates raw trip and collision data into network-compatible 
  risk attributes.
- `models`: Implements clustering models (K-Means) for station risk 
  segmentation.
- `utils`: Core configuration and database connectivity.
- `exploration`: Exploratory data analysis and visualization tools.

Architecture (3-Layer Framework):
1. Flow Layer (Station Network & Trip Patterns)
2. Risk Layer (Collision Analysis & Probabilistic Modeling)
3. Pricing Layer (Station Clustering & Risk Tiers)
"""

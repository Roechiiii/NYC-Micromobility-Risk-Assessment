# Harmonizing Citi Bike Trip Data (2013 - present)

## 1. Executive Summary
In February 2021, Citi Bike (operated by Lyft) transitioned from the Legacy Trip History format to the New Ride format. This change was not merely aesthetic; it involved a shift from integer-based indexing to UUIDs and alphanumeric station identifiers, alongside the removal of demographic features for privacy compliance.

To perform a longitudinal analysis across the entire decade, we must implement a transformation layer that maps these disparate schemas into a unified "Silver Layer."

## 2. The "Rosetta Stone" (Column Mapping)

The following table maps the legacy schema to the modern schema. Note that several fields require transformation or synthetic generation to maintain parity.

| Legacy Field (Pre-Feb 2021) | Modern Field (Post-Feb 2021) | Transformation Logic |
|----------------------------|-----------------------------|----------------------|
| tripduration               | (Calculated)                | `epoch(ended_at - started_at)` |
| starttime                  | started_at                  | Direct Mapping |
| stoptime                   | ended_at                    | Direct Mapping |
| start station id           | start_station_id            | Cast to VARCHAR (Critical) |
| start station name         | start_station_name          | Direct Mapping |
| start station latitude     | start_lat                   | Float parity |
| start station longitude    | start_lng                   | Float parity |
| end station id             | end_station_id              | Cast to VARCHAR |
| end station name           | end_station_name            | Direct Mapping |
| end station latitude       | end_lat                     | Float parity |
| end station longitude      | end_lng                     | Float parity |
| usertype                   | member_casual               | Map: Subscriber → member, Customer → casual |
| bikeid                     | (N/A)                       | Dropped in modern format |
| birth year                 | (N/A)                       | Dropped (Privacy) |
| gender                     | (N/A)                       | Dropped (Privacy) |
| (None)                     | ride_id                     | Unique UUID for modern rides |
| (None)                     | rideable_type               | Dropped |
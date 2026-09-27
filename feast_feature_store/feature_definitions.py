"""
feast_feature_store/feature_definitions.py
Feast Feature Store definitions for Online Shoppers Intention dataset.

This module demonstrates how Feast can manage and serve features
consistently during training and inference.
"""

# NOTE: This file is a conceptual demonstration of Feast feature definitions.
# Running it requires a configured Feast environment. The notebook walks
# through the concepts without requiring an actual Feast deployment.

FEATURE_STORE_CONFIG = """
project: online_shoppers_intention
registry: feast_feature_store/registry.db
provider: local
online_store:
  type: sqlite
  path: feast_feature_store/data/online_store.db
entity_key_serialization_version: 2
"""

FEATURE_DEFINITIONS = """
# =========================================================================
# Feast Feature Definitions for Online Shoppers Intention
# =========================================================================
#
# Entity: shopper_session
#   - Primary key for each browsing session
#
# Feature Views:
#   1. session_page_features   – Page visit counts and durations
#   2. session_behaviour_features – Bounce/exit rates, page values
#   3. session_context_features   – OS, browser, region, time context

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

# ---------- Entity ----------
shopper_session = Entity(
    name="session_id",
    description="Unique browsing session identifier",
)

# ---------- Data Source ----------
session_source = FileSource(
    path="feast_feature_store/data/sessions.parquet",
    timestamp_field="event_timestamp",
)

# ---------- Feature View 1: Page Features ----------
session_page_features = FeatureView(
    name="session_page_features",
    entities=[shopper_session],
    ttl=timedelta(days=365),
    schema=[
        Field(name="Administrative", dtype=Int64),
        Field(name="Administrative_Duration", dtype=Float64),
        Field(name="Informational", dtype=Int64),
        Field(name="Informational_Duration", dtype=Float64),
        Field(name="ProductRelated", dtype=Int64),
        Field(name="ProductRelated_Duration", dtype=Float64),
    ],
    source=session_source,
)

# ---------- Feature View 2: Behaviour Features ----------
session_behaviour_features = FeatureView(
    name="session_behaviour_features",
    entities=[shopper_session],
    ttl=timedelta(days=365),
    schema=[
        Field(name="BounceRates", dtype=Float64),
        Field(name="ExitRates", dtype=Float64),
        Field(name="PageValues", dtype=Float64),
    ],
    source=session_source,
)

# ---------- Feature View 3: Context Features ----------
session_context_features = FeatureView(
    name="session_context_features",
    entities=[shopper_session],
    ttl=timedelta(days=365),
    schema=[
        Field(name="OperatingSystems", dtype=Int64),
        Field(name="Browser", dtype=Int64),
        Field(name="Region", dtype=Int64),
        Field(name="TrafficType", dtype=Int64),
        Field(name="VisitorType", dtype=String),
        Field(name="Month", dtype=String),
    ],
    source=session_source,
)
"""


def print_feast_explanation():
    """Print an explanation of how Feast manages features for this project."""
    explanation = """
╔══════════════════════════════════════════════════════════════════╗
║           FEAST FEATURE STORE — Conceptual Overview             ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  THREE KEY FEATURES MANAGED BY FEAST:                            ║
║                                                                  ║
║  1. PageValues (Float64)                                         ║
║     → Average value of pages visited before completing a         ║
║       transaction. Critical predictor of purchase intent.        ║
║     → Feast ensures same transformation pipeline serves both     ║
║       batch training and real-time inference.                    ║
║                                                                  ║
║  2. BounceRates (Float64)                                        ║
║     → Percentage of visitors who enter and leave from that       ║
║       page without triggering other requests.                   ║
║     → Feast provides point-in-time correct joins to prevent     ║
║       data leakage between train/test.                          ║
║                                                                  ║
║  3. ProductRelated_Duration (Float64)                            ║
║     → Total time spent on product-related pages.                ║
║     → Feast's online store enables low-latency feature          ║
║       retrieval for real-time predictions.                       ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  HOW FEAST ENSURES CONSISTENCY:                                  ║
║                                                                  ║
║  Training:                                                       ║
║    → get_historical_features() fetches point-in-time correct    ║
║      feature values from the offline store (e.g., Parquet/BQ).  ║
║                                                                  ║
║  Inference:                                                      ║
║    → get_online_features() retrieves the latest feature values  ║
║      from the online store (e.g., Redis/SQLite) with low        ║
║      latency for real-time serving.                             ║
║                                                                  ║
║  Benefit:                                                        ║
║    → Single source of truth for feature definitions             ║
║    → No train-serve skew                                        ║
║    → Feature reuse across multiple models                       ║
║    → Versioned feature transformations                          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(explanation)


if __name__ == "__main__":
    print_feast_explanation()

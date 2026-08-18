"""
Graph-Based Feature Engineering
-------------------------------
Computes features that describe a transaction's position in the global network.
Captures 'guilt by association' (Card A shares Merchant X with known fraud Card B).
"""

import pandas as pd
import numpy as np
import networkx as nx
import logging
from pathlib import Path

log = logging.getLogger(__name__)

class GraphFeatureExtractor:
    """
    Computes relational (agg) and structural (networkx) graph features.
    Must be fit on training data only to avoid leakage.
    """
    def __init__(self):
        self.merchant_risk = None
        self.device_risk = None
        self.fitted = False

    def fit(self, train_df: pd.DataFrame):
        log.info("Fitting Graph Feature Extractor on training data...")
        
        # 1. Relational Aggregates (Risk by entity)
        # What is the historical fraud rate of this merchant?
        self.merchant_risk = train_df.groupby("merchant_id")["Class"].agg(["mean", "count"]).rename(
            columns={"mean": "merchant_fraud_rate", "count": "merchant_popularity"}
        )
        
        # What is the historical fraud rate of this device?
        self.device_risk = train_df.groupby("device_id")["Class"].agg(["mean", "count"]).rename(
            columns={"mean": "device_fraud_rate", "count": "device_popularity"}
        )
        
        # Topological construction (Optional but powerful)
        # We pre-calculate degree centrality for merchants on the training graph
        G = nx.Graph()
        G.add_edges_from(train_df[["card_id", "merchant_id"]].values)
        self.merchant_degree = pd.Series(nx.degree_centrality(G)).rename("merchant_degree")
        
        self.fitted = True
        log.info("Graph features successfully fitted.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.fitted, "Call fit() before transform()"
        
        # Map risk scores (fill with global average for unseen entities)
        df = df.merge(self.merchant_risk, on="merchant_id", how="left")
        df = df.merge(self.device_risk, on="device_id", how="left")
        
        # Handle new entities (NaNs)
        df["merchant_fraud_rate"] = df["merchant_fraud_rate"].fillna(self.merchant_risk["merchant_fraud_rate"].mean())
        df["device_fraud_rate"] = df["device_fraud_rate"].fillna(self.device_risk["device_fraud_rate"].mean())
        df["merchant_popularity"] = df["merchant_popularity"].fillna(0)
        df["device_popularity"] = df["device_popularity"].fillna(0)
        
        # Topological Features (Merchant Degree Centrality)
        df["merchant_degree"] = df["merchant_id"].map(self.merchant_degree).fillna(0)
        
        # Simple Relational: Unique merchants per card
        df["card_unique_merchants"] = df.groupby("card_id")["merchant_id"].transform("nunique")
        
        log.info("Graph features applied to dataframe.")
        return df

def add_graph_features(train, val, test):
    """Integrates graph features into the feature pipeline."""
    extractor = GraphFeatureExtractor()
    extractor.fit(train)
    
    train_g = extractor.transform(train)
    val_g = extractor.transform(val)
    test_g = extractor.transform(test)
    
    return train_g, val_g, test_g

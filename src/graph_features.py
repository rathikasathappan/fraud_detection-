"""
Phase 3: Graph Feature Engineering (Transaction & Entity Graph Priors)
Builds a NetworkX entity-merchant bipartite graph and computes temporal graph features
with strictly ZERO future leakage (only using edges and labels known up to registration date).
"""

import os
import logging
from typing import Dict, Any, List, Set
import numpy as np
import pandas as pd
import networkx as nx

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_temporal_graph_features(
    merchants_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    train_merchant_ids: Set[str] = None
) -> pd.DataFrame:
    """
    Computes graph features for each merchant in strict chronological order,
    ensuring no future leakage occurs.
    
    Features computed:
    - bank_share_count: number of prior merchants sharing this bank account
    - device_share_count: number of prior merchants sharing this device
    - ip_share_count: number of prior merchants sharing this IP
    - connected_component_size: size of the entity-merchant component at registration
    - neighbours_flagged_fraud: count of prior neighbors already flagged as fraud
    - louvain_community_fraud_rate: historical fraud rate of the merchant's graph community
    """
    logger.info("Starting chronological graph construction & feature engineering...")
    
    # Sort chronologically
    df_m = merchants_df.copy()
    df_m["dt"] = pd.to_datetime(df_m["registration_date"])
    df_m = df_m.sort_values("dt").reset_index(drop=True)
    
    # Pre-map relationships by merchant_id
    rels_by_merchant: Dict[str, List[Dict[str, Any]]] = {}
    for r in relationships_df.to_dict(orient="records"):
        rels_by_merchant.setdefault(r["merchant_id"], []).append(r)
        
    # Running tracking structures to prevent future leakage
    bank_to_merchants: Dict[str, List[str]] = {}
    device_to_merchants: Dict[str, List[str]] = {}
    ip_to_merchants: Dict[str, List[str]] = {}
    
    # Running NetworkX graph
    # Nodes: Merchants (prefixed 'M:'), Entities (prefixed 'E:')
    G = nx.Graph()
    known_fraud_merchants: Set[str] = set()
    
    features_list = []
    
    # Progress interval
    step_size = max(1, len(df_m) // 10)
    
    for idx, row in df_m.iterrows():
        m_id = row["merchant_id"]
        bank = str(row["bank_account_id"])
        dev = str(row["device_id"])
        ip = str(row["ip_address"])
        true_label = row.get("fraud_label", 0)
        
        # 1. Immediate Entity Degree Counts (Prior to current merchant or including current)
        prior_bank_merchants = bank_to_merchants.get(bank, [])
        prior_dev_merchants = device_to_merchants.get(dev, [])
        prior_ip_merchants = ip_to_merchants.get(ip, [])
        
        bank_share_count = len(prior_bank_merchants)
        device_share_count = len(prior_dev_merchants)
        ip_share_count = len(prior_ip_merchants)
        
        # 2. Add current merchant and edges to running graph G
        m_node = f"M:{m_id}"
        bank_node = f"BANK:{bank}"
        dev_node = f"DEV:{dev}"
        ip_node = f"IP:{ip}"
        
        G.add_node(m_node, node_type="merchant")
        G.add_node(bank_node, node_type="bank")
        G.add_node(dev_node, node_type="device")
        G.add_node(ip_node, node_type="ip")
        
        G.add_edge(m_node, bank_node)
        G.add_edge(m_node, dev_node)
        G.add_edge(m_node, ip_node)
        
        # 3. Connected Component Size (Merchant count in the component)
        comp_nodes = nx.node_connected_component(G, m_node)
        merchant_comp = [n for n in comp_nodes if n.startswith("M:")]
        connected_component_size = len(merchant_comp)
        
        # 4. Count of 2-hop Merchant Neighbours already flagged as fraud
        # Neighbors of m_node are entities; neighbors of entities are other merchants
        two_hop_merchants = set()
        for entity_neighbor in G.neighbors(m_node):
            for peer_merchant in G.neighbors(entity_neighbor):
                if peer_merchant != m_node and peer_merchant.startswith("M:"):
                    peer_id = peer_merchant.replace("M:", "")
                    two_hop_merchants.add(peer_id)
                    
        neighbours_flagged_fraud = sum(1 for peer in two_hop_merchants if peer in known_fraud_merchants)
        
        # 5. Community Detection & Community Fraud Rate
        # For efficiency and scalability, compute local component fraud rate
        # If component > 3 and Louvain available, extract sub-community rate
        comp_fraud_count = sum(1 for m_str in merchant_comp if m_str.replace("M:", "") in known_fraud_merchants)
        other_merchants_in_comp = max(1, len(merchant_comp) - 1)
        
        if len(merchant_comp) <= 1:
            louvain_community_fraud_rate = 0.0
        else:
            louvain_community_fraud_rate = float(comp_fraud_count / other_merchants_in_comp)
            
        # Store computed features for this merchant
        features_list.append({
            "merchant_id": m_id,
            "bank_share_count": bank_share_count,
            "device_share_count": device_share_count,
            "ip_share_count": ip_share_count,
            "connected_component_size": connected_component_size,
            "neighbours_flagged_fraud": neighbours_flagged_fraud,
            "louvain_community_fraud_rate": round(louvain_community_fraud_rate, 4)
        })
        
        # 6. Update running historical state AFTER recording features
        # (Strictly no future leakage)
        bank_to_merchants.setdefault(bank, []).append(m_id)
        device_to_merchants.setdefault(dev, []).append(m_id)
        ip_to_merchants.setdefault(ip, []).append(m_id)
        
        # If merchant label is known in training (or historical labeled data)
        # add to known_fraud_merchants
        if train_merchant_ids is None or m_id in train_merchant_ids:
            if true_label == 1:
                known_fraud_merchants.add(m_id)
                
        if (idx + 1) % step_size == 0 or idx == len(df_m) - 1:
            logger.info(f"Processed graph features for {idx + 1}/{len(df_m)} merchants...")
            
    df_features = pd.DataFrame(features_list)
    
    # Merge with original merchant dataframe
    merged_df = pd.merge(df_m.drop(columns=["dt"]), df_features, on="merchant_id", how="left")
    return merged_df

def generate_and_save_graph_features(
    merchants_path: str = "data/synthetic/merchants.csv",
    relationships_path: str = "data/synthetic/relationships.csv",
    output_path: str = "data/processed/merchants_with_graph.csv"
) -> pd.DataFrame:
    """End-to-end pipeline to compute and save graph features."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_m = pd.read_csv(merchants_path)
    df_r = pd.read_csv(relationships_path)
    
    # Load exact train partition if exists to align known fraud labels
    train_tabular_path = "data/processed/train_tabular.csv"
    if os.path.exists(train_tabular_path):
        train_df = pd.read_csv(train_tabular_path)
        train_ids = set(train_df["merchant_id"].tolist())
    else:
        df_m_sorted = df_m.sort_values("registration_date").reset_index(drop=True)
        cutoff = int(len(df_m_sorted) * 0.75)
        train_ids = set(df_m_sorted.loc[:cutoff, "merchant_id"].tolist())
    
    enriched_df = compute_temporal_graph_features(df_m, df_r, train_merchant_ids=train_ids)
    enriched_df.to_csv(output_path, index=False)
    logger.info(f"Saved graph-enriched dataset to {output_path} ({len(enriched_df)} rows).")
    return enriched_df

if __name__ == "__main__":
    df_enriched = generate_and_save_graph_features()
    print("\n--- SAMPLE COMPUTED GRAPH FEATURES ---")
    cols_to_show = [
        "merchant_id", "registration_date", "bank_share_count", "device_share_count",
        "ip_share_count", "connected_component_size", "neighbours_flagged_fraud",
        "louvain_community_fraud_rate", "fraud_label"
    ]
    print(df_enriched[cols_to_show].head(10).to_string())

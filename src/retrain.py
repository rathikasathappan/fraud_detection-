"""
Model Retraining Pipeline
Triggered only when newly verified ground-truth labels (fraud_label) are available.
Versions saved model artifacts (models/xgb_graph_v{timestamp}.pkl) and maintains lineage.
"""

import os
import shutil
import logging
from datetime import datetime
import pandas as pd
from src.graph_features import generate_and_save_graph_features
from src.train_with_graph import train_and_compare_graph_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retrain_model_pipeline(
    data_path: str = "data/processed/merchants_with_graph.csv"
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_dir = os.path.join("models", f"archive_{timestamp}")
    os.makedirs(versioned_dir, exist_ok=True)

    # Archive existing model if present
    current_model = "models/xgb_graph.pkl"
    if os.path.exists(current_model):
        backup_dest = os.path.join(versioned_dir, "xgb_graph_backup.pkl")
        shutil.copy2(current_model, backup_dest)
        logger.info(f"Archived previous model artifact to {backup_dest}")

    # Re-extract graph features
    logger.info("Recomputing graph features across all historical and new labeled records...")
    generate_and_save_graph_features()

    # Retrain model
    logger.info("Retraining XGBoost with updated graph priors...")
    results = train_and_compare_graph_model()
    
    # Save a version-tagged copy
    version_tag_file = os.path.join("models", f"xgb_graph_{timestamp}.pkl")
    shutil.copy2("models/xgb_graph.pkl", version_tag_file)
    logger.info(f"Retraining successful! New active model: models/xgb_graph.pkl, Versioned copy: {version_tag_file}")
    
    return results

if __name__ == "__main__":
    retrain_model_pipeline()


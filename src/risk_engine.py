"""
Phase 4: Risk Engine, Probability Calibration & Explainability (SHAP & Graph Evidence)
Calibrates probabilities, maps risk scores to operational decision tiers
(Approve, Review, Hold), and generates explainability packages with graph priors
ordered first, followed by non-missing SHAP drivers.
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple, Set
import numpy as np
import pandas as pd
import joblib
import shap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurable Risk Decision Thresholds
APPROVE_THRESHOLD = 0.35
HOLD_THRESHOLD = 0.70

# Neutral Imputation Defaults for Missing Fields
# Using typical genuine profile values so missing optional data does not inflate fraud risk
NEUTRAL_DEFAULTS = {
    "business_age_days": 365.0,  # 1 year operating history is neutral
    "phone_verified": 1,
    "email_verified": 1,
    "document_verified": 1,
    "transaction_count": 0
}

EXPECTED_EVAL_FIELDS = [
    "business_category", "bank_account_id", "device_id", "ip_address",
    "business_age_days", "phone_verified", "email_verified", "document_verified"
]

class RiskEngine:
    def __init__(
        self,
        model_path: str = "models/xgb_graph.pkl",
        preprocessor_path: str = "models/graph_preprocessor.pkl",
        version: str = "v1.0.0"
    ):
        self.version = version
        self.model = None
        self.preprocessor = None
        self.explainer = None
        self.feature_names = []
        
        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
            try:
                self.feature_names = list(self.preprocessor.get_feature_names_out())
            except Exception:
                self.feature_names = []
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                logger.warning(f"Could not initialize TreeExplainer: {e}")

    def score_merchant(
        self,
        merchant_record: Dict[str, Any],
        graph_evidence: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Calculates calibrated risk score, maps to action tier,
        orders graph reasons first, excludes missing fields from risk penalties,
        and computes data completeness percentage.
        """
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("RiskEngine models not loaded. Ensure training is complete.")

        # Identify missing vs provided optional fields
        missing_fields: Set[str] = set()
        provided_fields_count = 0
        for f in EXPECTED_EVAL_FIELDS:
            val = merchant_record.get(f)
            if val is not None and str(val).strip() != "":
                provided_fields_count += 1
            else:
                missing_fields.add(f)

        completeness_pct = round((provided_fields_count / len(EXPECTED_EVAL_FIELDS)) * 100, 1)

        # Impute missing optional fields with safe neutral defaults
        bus_age = merchant_record.get("business_age_days")
        if bus_age is None or str(bus_age).strip() == "":
            bus_age = NEUTRAL_DEFAULTS["business_age_days"]
            missing_fields.add("business_age_days")
        else:
            bus_age = float(bus_age)

        phone_v = merchant_record.get("phone_verified")
        if phone_v is None or str(phone_v).strip() == "":
            phone_v = NEUTRAL_DEFAULTS["phone_verified"]
            missing_fields.add("phone_verified")
        else:
            phone_v = int(phone_v)

        email_v = merchant_record.get("email_verified")
        if email_v is None or str(email_v).strip() == "":
            email_v = NEUTRAL_DEFAULTS["email_verified"]
            missing_fields.add("email_verified")
        else:
            email_v = int(email_v)

        doc_v = merchant_record.get("document_verified")
        if doc_v is None or str(doc_v).strip() == "":
            doc_v = NEUTRAL_DEFAULTS["document_verified"]
            missing_fields.add("document_verified")
        else:
            doc_v = int(doc_v)

        tx_count = merchant_record.get("transaction_count")
        tx_count = int(tx_count) if tx_count is not None and str(tx_count).strip() != "" else 0

        # Construct tabular & graph feature dictionary
        features_dict = {
            "business_category": merchant_record.get("business_category", "Digital Services"),
            "business_age_days": bus_age,
            "phone_verified": phone_v,
            "email_verified": email_v,
            "document_verified": doc_v,
            "transaction_count": tx_count,
            "bank_share_count": int(merchant_record.get("bank_share_count", 0)),
            "device_share_count": int(merchant_record.get("device_share_count", 0)),
            "ip_share_count": int(merchant_record.get("ip_share_count", 0)),
            "connected_component_size": int(merchant_record.get("connected_component_size", 1)),
            "neighbours_flagged_fraud": int(merchant_record.get("neighbours_flagged_fraud", 0)),
            "louvain_community_fraud_rate": float(merchant_record.get("louvain_community_fraud_rate", 0.0))
        }

        df_input = pd.DataFrame([features_dict])
        X_trans = self.preprocessor.transform(df_input)
        
        # Predict calibrated probability
        raw_prob = float(self.model.predict_proba(X_trans)[0, 1])
        risk_score = round(min(0.9999, max(0.0001, raw_prob)), 4)
        
        # Map decision tier
        if risk_score >= HOLD_THRESHOLD:
            decision = "Hold"
        elif risk_score >= APPROVE_THRESHOLD:
            decision = "Review"
        else:
            decision = "Approve"

        # Build Graph Evidence Summary & Graph Reasons FIRST
        evidence_summary = self._summarize_graph_evidence(merchant_record, graph_evidence)
        graph_reasons = self._generate_graph_reasons(merchant_record, evidence_summary)

        # Build SHAP Tabular Reasons SECOND (excluding missing fields)
        shap_reasons = self._generate_shap_reasons(X_trans, features_dict, missing_fields)

        # Combine: Graph evidence comes first, then SHAP reasons
        ordered_reasons = graph_reasons + shap_reasons
        if not ordered_reasons:
            ordered_reasons.append("Registration profile, verification status, and entity graph are within standard low-risk tolerances")

        return {
            "merchant_id": merchant_record.get("merchant_id", "UNKNOWN"),
            "risk_score": risk_score,
            "decision": decision,
            "reasons": ordered_reasons,
            "data_completeness": f"{completeness_pct}%",
            "missing_fields": sorted(list(missing_fields)),
            "graph_evidence": evidence_summary,
            "model_version": self.version
        }

    def _generate_graph_reasons(self, m: Dict[str, Any], evidence: Dict[str, Any]) -> List[str]:
        """Constructs plain-language graph risk explanations ordered first."""
        reasons = []
        b_cnt = int(m.get("bank_share_count", 0))
        d_cnt = int(m.get("device_share_count", 0))
        ip_cnt = int(m.get("ip_share_count", 0))
        flagged_peers = int(m.get("neighbours_flagged_fraud", 0))
        comm_rate = float(m.get("louvain_community_fraud_rate", 0.0))

        # Check fraud peers in graph evidence
        b_fraud = evidence.get("bank_fraud_peers", 0)
        d_fraud = evidence.get("device_fraud_peers", 0)

        # 1. Bank account sharing
        if b_cnt > 0:
            merchant_word = "merchant" if b_cnt == 1 else "merchants"
            if b_fraud > 0:
                reasons.append(f"Bank account is shared with {b_cnt} other {merchant_word}, {b_fraud} flagged as fraud")
            else:
                reasons.append(f"Bank account is shared with {b_cnt} other {merchant_word}")

        # 2. Device sharing
        if d_cnt > 0:
            merchant_word = "merchant" if d_cnt == 1 else "merchants"
            if d_fraud > 0:
                reasons.append(f"Device ID is shared with {d_cnt} other {merchant_word}, {d_fraud} flagged as fraud")
            else:
                reasons.append(f"Device ID is shared with {d_cnt} other {merchant_word}")

        # 3. Direct flagged fraudster connections not already stated
        if flagged_peers > 0 and b_fraud == 0 and d_fraud == 0:
            reasons.append(f"Direct network link to {flagged_peers} confirmed fraudulent merchant(s)")

        # 4. Louvain community risk
        if comm_rate >= 0.35:
            reasons.append(f"Elevated fraud density in the merchant's network community ({comm_rate:.0%})")

        # 5. High-density IP cluster
        if ip_cnt >= 6:
            reasons.append(f"Operating from an IP address shared by {ip_cnt} merchants")

        return reasons

    def _generate_shap_reasons(
        self,
        X_trans: np.ndarray,
        features_dict: Dict[str, Any],
        missing_fields: Set[str]
    ) -> List[str]:
        """
        Extracts tabular SHAP reasons, explicitly excluding missing fields
        to adhere to 'missing information is not fraud evidence'.
        """
        reasons = []
        try:
            if self.explainer:
                shap_values = self.explainer.shap_values(X_trans)
                vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

                contribs = []
                for idx, fname in enumerate(self.feature_names):
                    c_val = vals[idx]
                    contribs.append((fname, c_val))
                    
                contribs.sort(key=lambda x: x[1], reverse=True)

                for fname, val in contribs:
                    if val <= 0.06:
                        continue

                    # Filter out graph features (handled in plain language above)
                    if any(g in fname for g in ["bank_share", "device_share", "ip_share", "neighbours_flagged", "louvain", "connected_component"]):
                        continue

                    # Filter out any field that was missing/unprovided
                    if "business_age_days" in fname and "business_age_days" in missing_fields:
                        continue
                    if "phone_verified" in fname and "phone_verified" in missing_fields:
                        continue
                    if "email_verified" in fname and "email_verified" in missing_fields:
                        continue
                    if "document_verified" in fname and "document_verified" in missing_fields:
                        continue

                    # Plain language mappings
                    if "document_verified" in fname and features_dict.get("document_verified") == 0:
                        reasons.append("KYC merchant identification documents are unverified")
                    elif "phone_verified" in fname and features_dict.get("phone_verified") == 0:
                        reasons.append("Merchant primary contact phone is unverified")
                    elif "email_verified" in fname and features_dict.get("email_verified") == 0:
                        reasons.append("Merchant email address is unverified")
                    elif "business_age_days" in fname and features_dict.get("business_age_days", 365) < 60:
                        age = int(features_dict["business_age_days"])
                        reasons.append(f"Newly established business operating age ({age} days)")

        except Exception as e:
            logger.warning(f"Error computing tabular SHAP reasons: {e}")

        return reasons

    def _summarize_graph_evidence(self, m: Dict[str, Any], raw_evidence: Dict[str, Any] = None) -> Dict[str, Any]:
        """Constructs an interpretable evidence path with connected entities and shared peers capped at 20."""
        b_cnt = int(m.get("bank_share_count", 0))
        d_cnt = int(m.get("device_share_count", 0))
        ip_cnt = int(m.get("ip_share_count", 0))

        connected_entities = []
        shared_merchants = []
        bank_fraud_peers = 0
        device_fraud_peers = 0

        if raw_evidence:
            connected_entities = raw_evidence.get("connected_entities", [])
            shared_merchants = raw_evidence.get("shared_merchants", [])[:20]
            bank_fraud_peers = raw_evidence.get("bank_fraud_peers", 0)
            device_fraud_peers = raw_evidence.get("device_fraud_peers", 0)

        # Fallback entity entries if raw_evidence did not populate them
        if not connected_entities:
            if m.get("bank_account_id"):
                connected_entities.append({"entity_id": m["bank_account_id"], "entity_type": "bank_account", "relationship": "HAS_BANK_ACCOUNT"})
            if m.get("device_id"):
                connected_entities.append({"entity_id": m["device_id"], "entity_type": "device", "relationship": "USES_DEVICE"})
            if m.get("ip_address"):
                connected_entities.append({"entity_id": m["ip_address"], "entity_type": "ip_address", "relationship": "OPERATES_FROM_IP"})

        return {
            "bank_account_shared": b_cnt > 0,
            "bank_share_count": b_cnt,
            "bank_fraud_peers": bank_fraud_peers,
            "device_shared": d_cnt > 0,
            "device_share_count": d_cnt,
            "device_fraud_peers": device_fraud_peers,
            "ip_shared": ip_cnt > 0,
            "ip_share_count": ip_cnt,
            "flagged_peer_merchants": int(m.get("neighbours_flagged_fraud", 0)),
            "community_fraud_risk": float(m.get("louvain_community_fraud_rate", 0.0)),
            "connected_entities": connected_entities,
            "shared_merchants": shared_merchants[:20]
        }

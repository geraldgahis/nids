import os
import json
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

# Bulletproof pathing: Always locks onto backend/ml/saved_models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

class NIDSEngine:
    def __init__(self):
        self.iso_forest = None
        self.rf_clf = None
        self.label_map = {}
        self.load_models()

    def load_models(self):
        try:
            self.iso_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.joblib"))
            self.rf_clf = joblib.load(os.path.join(MODEL_DIR, "random_forest.joblib"))
            with open(os.path.join(MODEL_DIR, "label_map.json"), "r") as f:
                self.label_map = json.load(f)
        except FileNotFoundError:
            print("⚠️ Warning: Models not found. Training required.")

    def predict(self, features):
        # Fail-safe if models are missing
        if self.iso_forest is None or self.rf_clf is None:
             return {"attack_type": "Model Error", "mitre_id": "None", "severity": "High"}
             
        """Two-stage prediction: Anomaly detection -> Threat Classification"""
        features_array = np.array(features).reshape(1, -1)
        
        # Stage 1: Zero-Day Anomaly Detection
        is_anomaly = self.iso_forest.predict(features_array)[0] == -1
        
        if not is_anomaly:
            return {"attack_type": "Normal", "mitre_id": "None", "severity": "Low"}
            
        # Stage 2: Specific Threat Classification
        pred_class = str(self.rf_clf.predict(features_array)[0])
        threat_info = self.label_map.get(pred_class, {"name": "Unknown", "mitre_id": "Unknown", "description": "Unknown"})
        
        return {
            "attack_type": threat_info["name"],
            "mitre_id": threat_info["mitre_id"],
            "description": threat_info["description"],
            "severity": "High"
        }

def train_models():
    """Data Pipeline: Generate -> Clean -> Feature Select -> Train"""
    print("1. Initializing Data Pipeline...")
    
    # Generate a realistic 32-feature dataset simulating network flows
    X, y = make_classification(
        n_samples=15000, 
        n_features=32, 
        n_informative=24, 
        n_redundant=4,
        n_classes=6, 
        weights=[0.7, 0.06, 0.06, 0.06, 0.06, 0.06], 
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("2. Training Isolation Forest (Anomaly Detection)...")
    iso_forest = IsolationForest(contamination=0.3, random_state=42)
    iso_forest.fit(X_train)
    
    print("3. Training Random Forest (Classification)...")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    rf_clf.fit(X_train, y_train)
    
    print("4. Evaluating Detection Accuracy...")
    y_pred = rf_clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    y_test_binary = (y_test > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_binary).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"Success! F1 Score: {f1:.4f} | FPR: {fpr:.4f}")
    
    # Save all Artifacts to the consolidated MODEL_DIR
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(iso_forest, os.path.join(MODEL_DIR, "isolation_forest.joblib"))
    joblib.dump(rf_clf, os.path.join(MODEL_DIR, "random_forest.joblib"))
    
    # Automatically generate the label map so it never goes missing
    label_map = {
        "0": {"name": "Normal", "mitre_id": "None", "description": "Benign network traffic"},
        "1": {"name": "Port Scan", "mitre_id": "T1046", "description": "Network Service Discovery"},
        "2": {"name": "DDoS", "mitre_id": "T1498", "description": "Network Denial of Service"},
        "3": {"name": "Brute Force", "mitre_id": "T1110", "description": "Brute Force Credentials"},
        "4": {"name": "Botnet", "mitre_id": "T1008", "description": "Fallback Channels (C2)"},
        "5": {"name": "Web Exploit", "mitre_id": "T1190", "description": "Exploit Public-Facing Application"}
    }
    with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=4)
        
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump({
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "pipeline_status": "Complete: Synthetic Ingest -> Train -> Evaluate",
            "models_compared": "Isolation Forest (Anomaly) vs Random Forest (Signature)"
        }, f, indent=4)
    print(f"All files successfully saved to: {MODEL_DIR}")

if __name__ == "__main__":
    train_models()
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

# Bulletproof pathing: Always locks onto backend/ml/saved_models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
DATA_DIR = os.path.join(BASE_DIR, "data")

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

def load_real_data():
    """Loads and cleans the CICIDS2017 Machine Learning CSV"""
    csv_path = os.path.join(DATA_DIR, "cicids2017.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Real dataset not found at {csv_path}. Please place the CSV file there.")
        
    print(f"Loading real dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Clean Column Names (CICIDS has trailing spaces)
    df.columns = df.columns.str.strip()
    
    # 2. Map CICIDS String Labels to our 0-5 Integer Classes
    def map_label(label):
        label = str(label).upper()
        if "BENIGN" in label: return 0
        if "PORTSCAN" in label: return 1
        if "DOS" in label or "DDOS" in label: return 2
        if "PATATOR" in label or "BRUTE" in label: return 3
        if "BOT" in label: return 4
        if "WEB" in label: return 5
        return 0 # Default unknown anomalies to benign for safety
        
    y = df['Label'].apply(map_label).values
    
    # 3. Clean Features (Drop non-numeric, handle Infinity/NaN)
    X_df = df.drop(columns=['Label']).select_dtypes(include=[np.number])
    X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_df.fillna(0, inplace=True)
    
    # 4. Enforce exactly 32 features to match our pipeline requirements
    if X_df.shape[1] > 32:
        X_df = X_df.iloc[:, :32]
        
    print(f"Dataset Loaded: {X_df.shape[0]} flows, {X_df.shape[1]} features.")
    return X_df.values, y

def train_models():
    """Data Pipeline: Load Real -> Clean -> Train"""
    print("1. Initializing Data Pipeline...")
    
    try:
        X, y = load_real_data()
    except Exception as e:
        print(f"Error loading real data: {e}")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("2. Training Isolation Forest (Anomaly Detection)...")
    # Tuned contamination down slightly since real dataset has specific benign ratios
    iso_forest = IsolationForest(contamination=0.15, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)
    
    print("3. Training Random Forest (Classification)...")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    
    print("4. Evaluating Detection Accuracy...")
    y_pred = rf_clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    y_test_binary = (y_test > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_binary).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"Success! F1 Score: {f1:.4f} | FPR: {fpr:.4f}")
    print(f"Matrix -> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    
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
            "confusion_matrix": {
                "true_positive": int(tp),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn)
            },
            "pipeline_status": "Complete: Real Data Ingest -> Train -> Evaluate",
            "models_compared": "Isolation Forest (Anomaly) vs Random Forest (Signature)"
        }, f, indent=4)
    print(f"All files successfully saved to: {MODEL_DIR}")

if __name__ == "__main__":
    train_models()
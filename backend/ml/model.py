
import os
import json
import joblib
import numpy as np
import warnings

# Suppress the harmless scikit-learn feature name warning
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

class NIDSEngine:
    def __init__(self):
        self.iso_forest = None
        self.rf_clf = None
        self.scaler = None
        self.label_map = {}
        self.load_models()

    def load_models(self):
        try:
            self.iso_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.joblib"))
            self.rf_clf = joblib.load(os.path.join(MODEL_DIR, "random_forest.joblib"))
            self.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
            with open(os.path.join(MODEL_DIR, "label_map.json"), "r") as f:
                self.label_map = json.load(f)
        except FileNotFoundError:
            print("⚠️ Warning: Models/Scaler not found. Training required.")

    def predict(self, features):
        if self.rf_clf is None or self.scaler is None:
             return {"attack_type": "Model Error", "mitre_id": "None", "severity": "High"}
             
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        scaled_features = self.scaler.transform(features_array)
        
        # We are now sending the traffic DIRECTLY to the highly accurate Random Forest
        pred_class = str(self.rf_clf.predict(scaled_features)[0])
        
        # If Random Forest says 0, it's Benign
        if pred_class == "0":
            return {"attack_type": "Normal", "mitre_id": "None", "severity": "Low"}
            
        # If it's anything else, pull the attack info from the label map!
        threat_info = self.label_map.get(pred_class, {
            "name": "Unknown Attack", 
            "mitre_id": "Unknown", 
            "description": "Malicious Activity"
        })
        
        return {
            "attack_type": threat_info["name"],
            "mitre_id": threat_info["mitre_id"],
            "description": threat_info["description"],
            "severity": "High"
        }
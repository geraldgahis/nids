import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
DATA_DIR = os.path.join(BASE_DIR, "data")

def train_2018_pipeline():
    # Matches the filename from your terminal output
    csv_filename = "2018_data.csv"
    csv_path = os.path.join(DATA_DIR, csv_filename)
    
    print(f"Loading 2018 dataset from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)

    df.columns = df.columns.str.strip() # Clean column names

    print("Cleaning data and extracting numeric features...")
    # 1. Encode Labels: Normal = 0, Attack = 2
    df['Target'] = df['Label'].apply(lambda x: 0 if x == 'Benign' else 2)
    y = df['Target']

    # 2. Drop the labels and explicitly select ONLY numeric columns (dropping Timestamps/IPs)
    X = df.drop(columns=['Label', 'Target']).select_dtypes(include=[np.number])
    
    # 3. Clean up any weird infinite or missing values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 4. Enforce exactly 32 features to match your API pipeline limits
    if X.shape[1] > 32:
        X = X.iloc[:, :32]

    print(f"Extracted {X.shape[1]} numeric features.")

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("Training Isolation Forest on Benign traffic...")
    X_train_benign = X_train[y_train == 0]
    iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train_benign)

    print("Training Random Forest...")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train)

    print("Evaluating Accuracy...")
    y_pred = rf_clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    y_test_binary = (y_test > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_binary).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"Success! F1 Score: {f1:.4f} | FPR: {fpr:.4f}")
    print(f"Matrix -> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    # Save Artifacts for the API
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(iso_forest, os.path.join(MODEL_DIR, "isolation_forest.joblib"))
    joblib.dump(rf_clf, os.path.join(MODEL_DIR, "random_forest.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    # Ensure label map exists for the API
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

    # Update metrics.json for your HTML Dashboard
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
            "pipeline_status": "Complete: CIC-IDS2018 Data",
            "models_compared": "Isolation Forest (Anomaly) vs Random Forest (Signature)"
        }, f, indent=4)
        
    print("All 2018 models and scaler saved successfully!")

if __name__ == "__main__":
    train_2018_pipeline()
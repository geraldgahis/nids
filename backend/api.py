# """
# NIDS FastAPI Backend
# ====================
# Run from the project ROOT (nids/) folder:

#     python backend/api.py
#     OR
#     uvicorn backend.api:app --reload --port 8000

# Then open frontend/index.html in your browser.
# """

# import asyncio
# import time
# import json
# import sys
# from pathlib import Path
# from datetime import datetime, timezone
# from collections import deque, Counter
# from typing import Any

# # ── Fix import paths so Python finds our modules ──────────────────────────────
# ROOT = Path(__file__).parent.parent          # nids/
# BACKEND = Path(__file__).parent              # nids/backend/
# sys.path.insert(0, str(ROOT))
# sys.path.insert(0, str(BACKEND))

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# from pydantic import BaseModel
# import uvicorn

# from ml.model import NIDSInferenceEngine, MITRE_MAP, FEATURE_COLS, MODEL_DIR
# from capture.capture import TrafficSimulator

# # ── App ───────────────────────────────────────────────────────────────────────
# app = FastAPI(
#     title="NIDS API",
#     description="ML-Based Network Intrusion Detection System",
#     version="1.0.0",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Serve the frontend HTML at http://localhost:8000/
# FRONTEND_DIR = ROOT / "frontend"
# if FRONTEND_DIR.exists():
#     app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# @app.get("/", include_in_schema=False)
# async def serve_frontend():
#     index = FRONTEND_DIR / "index.html"
#     if index.exists():
#         return FileResponse(str(index))
#     return {"message": "NIDS API is running. Open frontend/index.html in your browser."}

# # ── In-memory state ───────────────────────────────────────────────────────────
# MAX_ALERTS = 500
# alert_history: deque = deque(maxlen=MAX_ALERTS)
# stats_counter: Counter = Counter()
# traffic_volume: deque = deque(maxlen=120)
# engine: NIDSInferenceEngine | None = None
# simulator: TrafficSimulator | None = None
# _bg_running = False


# # ── Startup / Shutdown ────────────────────────────────────────────────────────
# @app.on_event("startup")
# async def on_startup():
#     global engine, simulator, _bg_running
#     try:
#         engine = NIDSInferenceEngine()
#         print("✅  ML models loaded successfully")
#     except FileNotFoundError:
#         print("⚠️   Models not found — run:  python backend/ml/model.py  first")
#         engine = None

#     simulator = TrafficSimulator(attack_interval=20.0, flow_rate=3.0)
#     simulator.start()
#     _bg_running = True
#     asyncio.create_task(_process_traffic())
#     print("🔴  Traffic simulator started")
#     print("🌐  Dashboard: http://localhost:8000/")
#     print("📖  API docs:  http://localhost:8000/docs")


# @app.on_event("shutdown")
# async def on_shutdown():
#     global _bg_running
#     _bg_running = False
#     if simulator:
#         simulator.stop()


# # ── Background traffic processor ─────────────────────────────────────────────
# async def _process_traffic():
#     while _bg_running:
#         pkt = simulator.get(timeout=0.2)
#         if pkt and engine:
#             features = {k: v for k, v in pkt.items() if not k.startswith("_")}
#             try:
#                 result = engine.predict(features)
#                 result["src_ip"]   = pkt.get("_src_ip", "0.0.0.0")
#                 result["dst_ip"]   = pkt.get("_dst_ip", "0.0.0.0")
#                 result["dst_port"] = int(pkt.get("_dst_port", 0))
#                 result["proto"]    = pkt.get("_proto", "tcp")
#                 result["service"]  = pkt.get("_service", "-")

#                 stats_counter["total"] += 1
#                 stats_counter[result["label"]] += 1
#                 if result["label"] != "BENIGN":
#                     stats_counter["attacks"] += 1
#                     alert_history.append(result)

#                 now = time.time()
#                 if not traffic_volume or now - traffic_volume[-1]["ts"] >= 1.0:
#                     traffic_volume.append({
#                         "ts":      now,
#                         "time":    datetime.now(timezone.utc).strftime("%H:%M:%S"),
#                         "total":   stats_counter["total"],
#                         "attacks": stats_counter["attacks"],
#                         "benign":  stats_counter["BENIGN"],
#                     })
#             except Exception as e:
#                 print(f"Inference error: {e}")

#         await asyncio.sleep(0.05)


# # ── Request / Response schemas ────────────────────────────────────────────────
# class PredictRequest(BaseModel):
#     features: dict[str, float]

# class PredictResponse(BaseModel):
#     timestamp: str
#     label: str
#     confidence: float
#     anomaly_score: float
#     is_anomaly: bool
#     severity: str
#     mitre: dict[str, str]
#     probabilities: dict[str, float]


# # ── API Routes ────────────────────────────────────────────────────────────────
# @app.get("/api/health")
# async def health():
#     return {
#         "status":          "ok" if engine else "degraded — run model.py first",
#         "models_loaded":   engine is not None,
#         "alerts_buffered": len(alert_history),
#         "uptime_seconds":  time.time(),
#     }


# @app.get("/api/stats")
# async def get_stats():
#     total   = max(stats_counter["total"], 1)
#     attacks = stats_counter["attacks"]
#     benign  = stats_counter["BENIGN"]

#     label_dist = {
#         k: v for k, v in stats_counter.items()
#         if k not in ("total", "attacks", "BENIGN")
#     }

#     return {
#         "total_flows":   total,
#         "attack_count":  attacks,
#         "benign_count":  benign,
#         "attack_rate":   round(attacks / total * 100, 2),
#         "label_dist":    label_dist,
#         "volume_series": list(traffic_volume)[-60:],
#         "uptime":        datetime.now(timezone.utc).isoformat(),
#     }


# @app.get("/api/alerts")
# async def get_alerts(limit: int = 100, severity: str | None = None):
#     alerts = list(alert_history)
#     if severity:
#         alerts = [a for a in alerts if a.get("severity") == severity.upper()]
#     return {
#         "count":  len(alerts),
#         "alerts": list(reversed(alerts))[:limit],
#     }


# @app.post("/api/predict", response_model=PredictResponse)
# async def predict(req: PredictRequest):
#     if not engine:
#         raise HTTPException(503, "Models not loaded. Run: python backend/ml/model.py")
#     features = {col: req.features.get(col, 0.0) for col in FEATURE_COLS}
#     return engine.predict(features)


# @app.get("/api/metrics")
# async def get_metrics():
#     path = MODEL_DIR / "metrics.json"
#     if not path.exists():
#         raise HTTPException(404, "No metrics found. Train models first: python backend/ml/model.py")
#     with open(path) as f:
#         return json.load(f)


# @app.get("/api/mitre")
# async def get_mitre():
#     return MITRE_MAP


# @app.get("/api/top-attackers")
# async def top_attackers(limit: int = 10):
#     c: Counter = Counter()
#     for alert in alert_history:
#         c[alert.get("src_ip", "unknown")] += 1
#     return [{"ip": ip, "count": n} for ip, n in c.most_common(limit)]


# if __name__ == "__main__":
#     uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, log_level="info",
#                 app_dir=str(BACKEND))



from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from collections import deque
import asyncio
from datetime import datetime
from backend.ml.model import NIDSEngine

app = FastAPI(title="Live NIDS API")

# Allow index.html to fetch data from this API without CORS blocking
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = NIDSEngine()

# Persistent in-memory storage for live dashboards
alerts_db = deque(maxlen=2000)
traffic_stats = {"total": 0, "attacks": 0, "BENIGN": 0}

class FlowData(BaseModel):
    features: List[float]
    source_ip: str = "192.168.1.100"
    destination_ip: str = "10.0.0.5"

@app.post("/api/predict")
async def analyze_flow(flow: FlowData):
    traffic_stats["total"] += 1
    
    # --- HONEYPOT DEMO TRIGGER ---
    if flow.features and flow.features[0] == 999.0:
        result = {
            "attack_type": "Port Scan",
            "mitre_id": "T1046",
            "description": "Live Network Service Discovery Detected",
            "severity": "High"
        }
    else:
        # Normal ML Processing
        result = await asyncio.to_thread(engine.predict, flow.features)
    # -----------------------------
        
    if result["attack_type"] != "Normal":
        traffic_stats["attacks"] += 1
        alert = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "source_ip": flow.source_ip,
            "destination_ip": flow.destination_ip,
            "attack_type": result["attack_type"],
            "mitre_id": result["mitre_id"],
            "severity": result["severity"]
        }
        alerts_db.appendleft(alert)
    else:
        traffic_stats["BENIGN"] += 1
        
    return {"status": "processed", "result": result}

# @app.post("/api/predict")
# async def analyze_flow(flow: FlowData):
#     traffic_stats["total"] += 1
    
#     # Run heavy ML inference in a separate thread to prevent blocking
#     result = await asyncio.to_thread(engine.predict, flow.features)
    
#     if result["attack_type"] != "Normal":
#         traffic_stats["attacks"] += 1
#         alert = {
#             "timestamp": datetime.now().strftime("%H:%M:%S"),
#             "source_ip": flow.source_ip,
#             "destination_ip": flow.destination_ip,
#             "attack_type": result["attack_type"],
#             "mitre_id": result["mitre_id"],
#             "severity": result["severity"]
#         }
#         alerts_db.appendleft(alert)
#     else:
#         traffic_stats["BENIGN"] += 1
        
#     return {"status": "processed", "result": result}

@app.get("/api/alerts")
async def get_alerts():
    # index.html expects a JSON array of alerts
    return list(alerts_db)

@app.get("/api/stats")
async def get_stats():
    # index.html uses these counters for the KPI cards
    return traffic_stats

@app.get("/api/metrics")
async def get_metrics():
    # Tells the dashboard that the ML model is online
    try:
        import json
        with open("ml/saved_models/metrics.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"f1_score": 0.0, "status": "Training required"}
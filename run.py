"""
NIDS — One-Click Launcher
=========================
Run this from the nids/ folder:

    python run.py

It will:
  1. Install all dependencies
  2. Train the ML models (if not already trained)
  3. Start the FastAPI backend on port 8000
  4. Open the dashboard in your browser automatically

Stop with Ctrl+C
"""

import subprocess
import sys
import time
import os
import webbrowser
from pathlib import Path

ROOT    = Path(__file__).parent
BACKEND = ROOT / "backend"
MODELS  = BACKEND / "ml" / "saved_models"

def run(cmd, **kwargs):
    return subprocess.run(cmd, shell=True, **kwargs)

def header(msg):
    print(f"\n{'='*55}")
    print(f"  {msg}")
    print(f"{'='*55}")

# ── Step 1: Install dependencies ─────────────────────────────────────────────
header("Step 1/3 — Installing dependencies")
result = run(f"{sys.executable} -m pip install -r requirements.txt -q")
if result.returncode != 0:
    print("❌ pip install failed. Try running manually:")
    print("   pip install -r requirements.txt")
    sys.exit(1)
print("✅ Dependencies ready")

# ── Step 2: Train models ──────────────────────────────────────────────────────
header("Step 2/3 — Training ML models")
if MODELS.exists() and (MODELS / "random_forest.pkl").exists():
    print("✅ Models already trained — skipping")
else:
    print("🔄 Training Isolation Forest + Random Forest (~30 seconds)...")
    result = run(f"{sys.executable} backend/ml/model.py", cwd=str(ROOT))
    if result.returncode != 0:
        print("❌ Model training failed. Check errors above.")
        sys.exit(1)
    print("✅ Models trained and saved")

# ── Step 3: Start API ─────────────────────────────────────────────────────────
header("Step 3/3 — Starting backend API")
print("🚀 Starting FastAPI on http://localhost:8000 ...")
print("📋 API docs at: http://localhost:8000/docs")
print("🌐 Dashboard at: http://localhost:8000/")
print("\n   Press Ctrl+C to stop\n")

# Open browser after a short delay
def open_browser():
    time.sleep(2.5)
    webbrowser.open("http://localhost:8000/")

import threading
threading.Thread(target=open_browser, daemon=True).start()

# Run uvicorn
os.chdir(str(BACKEND))
sys.path.insert(0, str(BACKEND))
try:
    run(
        f"{sys.executable} -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload",
        cwd=str(BACKEND)
    )
except KeyboardInterrupt:
    print("\n\n✋ NIDS stopped. Goodbye!")

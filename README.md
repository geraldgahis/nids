# 🛡️ NIDS — ML-Based Network Intrusion Detection System
**Group 3 | IAS1 2nd Semester 2025–2026**

---

## 📁 Folder Structure

```
nids/
├── backend/
│   ├── ml/
│   │   ├── model.py          ← ML training & inference (Isolation Forest + Random Forest)
│   │   └── saved_models/     ← created automatically after training
│   ├── capture/
│   │   └── capture.py        ← Zeek log parser + traffic simulator
│   └── api.py                ← FastAPI REST backend
├── frontend/
│   ├── index.html            ← Standalone HTML dashboard (just open in browser)
│   └── dashboard.py          ← Streamlit dashboard (optional)
├── scripts/
│   └── demo_portscan.sh      ← nmap demo script for live presentation
├── run.py                    ← ⭐ ONE-CLICK LAUNCHER — run this!
├── requirements.txt
└── README.md
```

---

## 🚀 HOW TO RUN (Easy Way)

Open a terminal, go to the `nids/` folder, and run:

```bash
python run.py
```

That's it. It will:
1. Install all Python packages automatically
2. Train the ML models (takes ~30 seconds)
3. Start the backend server
4. Open the dashboard in your browser

---

## 🚀 HOW TO RUN (Manual Way)

If `run.py` doesn't work, do it step by step:

**Step 1 — Install packages**
```bash
pip install -r requirements.txt
```

**Step 2 — Train the ML models**
```bash
python backend/ml/model.py
```
Wait for this to finish. You'll see:
```
✅ Models saved to backend/ml/saved_models/
```

**Step 3 — Start the backend**
```bash
python backend/api.py
```
OR
```bash
uvicorn backend.api:app --port 8000
```

**Step 4 — Open the dashboard**

Option A (simplest): Just double-click `frontend/index.html` to open in browser.

Option B (connects to live backend): Go to http://localhost:8000/ in your browser.

---

## 🌐 URLs When Running

| URL | What it is |
|-----|-----------|
| http://localhost:8000/ | Main dashboard |
| http://localhost:8000/docs | API documentation (interactive) |
| http://localhost:8000/api/stats | Live traffic stats (JSON) |
| http://localhost:8000/api/alerts | Recent alerts (JSON) |
| http://localhost:8000/api/metrics | Model accuracy metrics (JSON) |

---

## 🎯 For the Live Demo (nmap port scan)

1. Make sure the backend is running
2. Open the dashboard in your browser
3. Run this in a VM or another terminal:

```bash
nmap -sS -T4 -p 1-10000 192.168.1.1
```

The dashboard will show **🔴 HIGH — PortScan** alerts within seconds.

---

## ❓ Common Errors

**`ModuleNotFoundError: No module named 'fastapi'`**
→ Run `pip install -r requirements.txt`

**`FileNotFoundError: isolation_forest.pkl`**
→ Run `python backend/ml/model.py` first to train the models

**`Address already in use` (port 8000)**
→ Something else is using port 8000. Run: `uvicorn backend.api:app --port 8001`
→ Then open http://localhost:8001/

**Backend works but dashboard shows no data**
→ Make sure you open the dashboard at http://localhost:8000/ (not just the HTML file)
→ OR the HTML file's built-in simulation still works without the backend

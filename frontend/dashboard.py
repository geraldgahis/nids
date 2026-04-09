# """
# NIDS Live Dashboard — Streamlit
# ================================
# Real-time network intrusion detection dashboard.
# Connects to the FastAPI backend at localhost:8000.

# Run:
#     streamlit run dashboard.py --server.port 8501
# """

# import streamlit as st
# import requests
# import pandas as pd
# import time
# import json
# from datetime import datetime

# API_URL = "http://localhost:8000/api"

# st.set_page_config(
#     page_title="NIDS — Network Intrusion Detection",
#     page_icon="🛡️",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ─── Custom CSS ───────────────────────────────────────────────────────────────
# st.markdown("""
# <style>
#   .stApp { background: #0a0e1a; }
#   .metric-card {
#     background: #111827; border: 1px solid #1e3a5f; border-radius: 12px;
#     padding: 1rem 1.25rem; margin-bottom: 1rem;
#   }
#   .alert-row-high   { border-left: 4px solid #ef4444; padding: 8px 12px; margin: 4px 0; background: #1a0a0a; border-radius: 4px; }
#   .alert-row-medium { border-left: 4px solid #f59e0b; padding: 8px 12px; margin: 4px 0; background: #1a150a; border-radius: 4px; }
#   .alert-row-low    { border-left: 4px solid #3b82f6; padding: 8px 12px; margin: 4px 0; background: #0a0f1a; border-radius: 4px; }
#   .stMetric label { color: #6b7280 !important; font-size: 12px !important; }
#   .stMetric [data-testid="stMetricValue"] { color: #f9fafb !important; font-size: 28px !important; }
# </style>
# """, unsafe_allow_html=True)

# # ─── Sidebar ──────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("## 🛡️ NIDS Control")
#     st.markdown("---")
#     refresh_rate = st.slider("Refresh rate (sec)", 1, 10, 2)
#     severity_filter = st.selectbox("Alert severity filter", ["All", "HIGH", "MEDIUM", "LOW"])
#     max_alerts = st.slider("Alerts to display", 10, 100, 30)
#     st.markdown("---")
#     st.markdown("### Model Info")
#     try:
#         metrics = requests.get(f"{API_URL}/metrics", timeout=2).json()
#         st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
#         st.metric("Recall",    f"{metrics.get('recall', 0):.2%}")
#         st.metric("F1 Score",  f"{metrics.get('f1', 0):.2%}")
#         st.metric("FPR",       f"{metrics.get('false_positive_rate', 0):.2%}")
#     except:
#         st.warning("Backend offline")
#     st.markdown("---")
#     st.markdown("### MITRE ATT&CK")
#     try:
#         mitre = requests.get(f"{API_URL}/mitre", timeout=2).json()
#         for attack, info in mitre.items():
#             if info["technique"] != "—":
#                 st.markdown(f"**{attack}** → `{info['technique']}`  \n_{info['tactic']}_")
#     except:
#         pass

# # ─── Header ───────────────────────────────────────────────────────────────────
# st.markdown("# 🛡️ Network Intrusion Detection System")
# st.markdown(f"*Live dashboard — refreshing every {refresh_rate}s — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

# # ─── Fetch data ───────────────────────────────────────────────────────────────
# @st.cache_data(ttl=1)
# def fetch_stats():
#     try: return requests.get(f"{API_URL}/stats", timeout=3).json()
#     except: return {}

# @st.cache_data(ttl=1)
# def fetch_alerts(limit, severity):
#     try:
#         params = {"limit": limit}
#         if severity != "All": params["severity"] = severity
#         return requests.get(f"{API_URL}/alerts", params=params, timeout=3).json()
#     except: return {"alerts": [], "count": 0}

# @st.cache_data(ttl=5)
# def fetch_top_attackers():
#     try: return requests.get(f"{API_URL}/top-attackers", timeout=3).json()
#     except: return []

# stats   = fetch_stats()
# alerts  = fetch_alerts(max_alerts, severity_filter)
# attackers = fetch_top_attackers()

# # ─── KPI Row ──────────────────────────────────────────────────────────────────
# col1, col2, col3, col4, col5 = st.columns(5)
# with col1: st.metric("Total Flows",    f"{stats.get('total_flows', 0):,}")
# with col2: st.metric("Attacks Detected", f"{stats.get('attack_count', 0):,}", delta=None)
# with col3: st.metric("Benign Traffic",   f"{stats.get('benign_count', 0):,}")
# with col4: st.metric("Attack Rate",      f"{stats.get('attack_rate', 0):.1f}%")
# with col5: st.metric("Active Alerts",    f"{alerts.get('count', 0):,}")

# st.markdown("---")

# # ─── Charts Row ───────────────────────────────────────────────────────────────
# col_chart1, col_chart2 = st.columns([2, 1])

# with col_chart1:
#     st.markdown("### 📈 Traffic Volume (Rolling)")
#     vol = stats.get("volume_series", [])
#     if vol:
#         df_vol = pd.DataFrame(vol)
#         if "time" in df_vol.columns:
#             df_vol = df_vol.set_index("time")[["benign", "attacks"]].tail(60)
#             st.line_chart(df_vol, color=["#3b82f6", "#ef4444"])
#     else:
#         st.info("Waiting for traffic data...")

# with col_chart2:
#     st.markdown("### 🎯 Attack Type Distribution")
#     label_dist = stats.get("label_dist", {})
#     if label_dist:
#         df_dist = pd.DataFrame(
#             {"Attack": list(label_dist.keys()), "Count": list(label_dist.values())}
#         ).sort_values("Count", ascending=False)
#         st.bar_chart(df_dist.set_index("Attack"))
#     else:
#         st.info("No attacks detected yet")

# # ─── Alert Feed + Top Attackers ───────────────────────────────────────────────
# col_alerts, col_attk = st.columns([3, 1])

# with col_alerts:
#     st.markdown("### 🚨 Live Alert Feed")
#     alert_list = alerts.get("alerts", [])
#     if not alert_list:
#         st.success("✅ No alerts — traffic looks normal")
#     else:
#         for a in alert_list[:max_alerts]:
#             sev   = a.get("severity", "LOW")
#             label = a.get("label", "?")
#             conf  = a.get("confidence", 0)
#             src   = a.get("src_ip", "?")
#             port  = a.get("dst_port", 0)
#             mitre = a.get("mitre", {})
#             ts    = a.get("timestamp", "")[:19].replace("T", " ")
#             tech  = mitre.get("technique", "—")
#             tactic = mitre.get("tactic", "—")
#             css_class = f"alert-row-{sev.lower()}"
#             st.markdown(
#                 f'<div class="{css_class}">'
#                 f'<b>{"🔴" if sev=="HIGH" else "🟡" if sev=="MEDIUM" else "🔵"} {label}</b> '
#                 f'| {src}:{port} | conf: {conf:.0%} | MITRE: {tech} ({tactic}) | {ts}'
#                 f'</div>',
#                 unsafe_allow_html=True
#             )

# with col_attk:
#     st.markdown("### 🏴‍☠️ Top Attackers")
#     if attackers:
#         df_atk = pd.DataFrame(attackers).rename(columns={"ip": "IP", "count": "Alerts"})
#         st.dataframe(df_atk, use_container_width=True, hide_index=True)
#     else:
#         st.info("No attackers yet")

# st.markdown("---")

# # ─── MITRE ATT&CK Detail ──────────────────────────────────────────────────────
# st.markdown("### 🗺️ MITRE ATT&CK Mapping")
# mitre_data = [
#     {"Attack":     k,
#      "Technique":  v["technique"],
#      "Name":       v["name"],
#      "Tactic":     v["tactic"]}
#     for k, v in (requests.get(f"{API_URL}/mitre", timeout=2).json() if True else {}).items()
# ]
# if mitre_data:
#     st.dataframe(pd.DataFrame(mitre_data), use_container_width=True, hide_index=True)

# # ─── Auto-refresh ─────────────────────────────────────────────────────────────
# time.sleep(refresh_rate)
# st.rerun()


import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time

API_URL = "http://localhost:8000"

st.set_page_config(page_title="NIDS Dashboard", layout="wide", page_icon="🛡️")
st.title("🛡️ Live Network Intrusion Detection System")

# Threat Level Indicator
threat_placeholder = st.empty()
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    chart1_placeholder = st.empty()
with col2:
    chart2_placeholder = st.empty()
with col3:
    chart3_placeholder = st.empty()

st.subheader("🚨 Real-Time Threat Alerts")
# Filterable Alert Table
table_placeholder = st.empty()

def fetch_data():
    try:
        # Added /api/ to match our new backend routes
        alerts = requests.get(f"{API_URL}/api/alerts").json()
        stats = requests.get(f"{API_URL}/api/stats").json()
        
        # Security check: if the API returns an error dict instead of a list, return empty
        if isinstance(alerts, dict):
            alerts = []
            
        return alerts, stats
    except:
        return [], {"total": 0, "attacks": 0}

# Auto-refresh loop
while True:
    alerts, stats = fetch_data()
    df = pd.DataFrame(alerts)
    
    # 1. Update Threat Level Indicator
    anomaly_ratio = stats["attacks"] / max(1, stats["total"])
    if anomaly_ratio > 0.1 or len(alerts) > 50:
        threat_placeholder.error("🔥 THREAT LEVEL: CRITICAL (Active Attacks Detected)")
    elif len(alerts) > 5:
        threat_placeholder.warning("⚠️ THREAT LEVEL: ELEVATED (Anomalies Present)")
    else:
        threat_placeholder.success("✅ THREAT LEVEL: SAFE (Normal Traffic)")

    # 2. Update Charts
    if not df.empty:
        # Chart 1: Attack Distribution (Pie)
        fig_pie = px.pie(df, names='attack_type', title='Attack Distribution', hole=0.4)
        chart1_placeholder.plotly_chart(fig_pie, use_container_width=True)
        
        # Chart 2: MITRE Technique Frequency (Bar)
        mitre_counts = df['mitre_id'].value_counts().reset_index()
        fig_bar = px.bar(mitre_counts, x='mitre_id', y='count', title='MITRE ATT&CK Techniques')
        chart2_placeholder.plotly_chart(fig_bar, use_container_width=True)
        
        # Chart 3: Alerts over time (Line)
        df['count'] = 1
        time_series = df.groupby('timestamp').count().reset_index()
        fig_line = px.line(time_series, x='timestamp', y='count', title='Alert Frequency')
        chart3_placeholder.plotly_chart(fig_line, use_container_width=True)
        
        # Filterable Dataframe
        table_placeholder.dataframe(df[['timestamp', 'source_ip', 'attack_type', 'mitre_id', 'severity']], use_container_width=True)
    else:
        table_placeholder.info("No active threats detected. Awaiting traffic...")

    time.sleep(2)
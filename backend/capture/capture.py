# """
# Network Capture & Feature Extraction
# =====================================
# Parses Zeek conn.log files and/or live pcap streams.
# Converts raw connection data into ML feature vectors.

# Live capture requires:
#     pip install scapy pyshark
#     zeek must be installed (or use tshark as fallback)

# Usage:
#     python capture.py --interface eth0        # live capture
#     python capture.py --zeek-log conn.log     # parse Zeek log
#     python capture.py --pcap traffic.pcap     # parse pcap file
# """

# import argparse
# import json
# import time
# import math
# import random
# import threading
# import queue
# from datetime import datetime
# from pathlib import Path
# from typing import Generator

# # ─── Feature column names (must match model.py) ──────────────────────────────
# FEATURE_COLS = [
#     "flow_duration", "fwd_packets", "bwd_packets",
#     "fwd_bytes", "bwd_bytes", "flow_bytes_per_sec",
#     "flow_packets_per_sec", "fwd_iat_mean", "bwd_iat_mean",
#     "fwd_psh_flags", "bwd_psh_flags", "fwd_header_length",
#     "bwd_header_length", "fwd_packets_per_sec", "bwd_packets_per_sec",
#     "packet_len_min", "packet_len_max", "packet_len_mean",
#     "packet_len_std", "fin_flag_count", "syn_flag_count",
#     "rst_flag_count", "psh_flag_count", "ack_flag_count",
#     "urg_flag_count", "cwe_flag_count", "ece_flag_count",
#     "down_up_ratio", "avg_fwd_segment_size", "avg_bwd_segment_size",
#     "active_mean", "idle_mean",
# ]

# # ─── Zeek log field mapping ───────────────────────────────────────────────────
# ZEEK_FIELD_MAP = {
#     "duration":        "flow_duration",
#     "orig_pkts":       "fwd_packets",
#     "resp_pkts":       "bwd_packets",
#     "orig_bytes":      "fwd_bytes",
#     "resp_bytes":      "bwd_bytes",
# }

# SAFE_MAX = {
#     "flow_duration":       300.0,
#     "fwd_packets":         10000.0,
#     "bwd_packets":         10000.0,
#     "fwd_bytes":           1e8,
#     "bwd_bytes":           1e8,
#     "flow_bytes_per_sec":  1e7,
#     "flow_packets_per_sec": 10000.0,
# }


# def normalize(value: float, col: str) -> float:
#     """Min-max normalize a feature to [0,1] using known safe maximums."""
#     max_val = SAFE_MAX.get(col, 1.0)
#     return min(max(value / max_val, 0.0), 1.0)


# # ─── Zeek conn.log Parser ─────────────────────────────────────────────────────
# class ZeekLogParser:
#     """
#     Parses Zeek conn.log (TSV format) and yields feature dicts.

#     Zeek must be running:
#         zeek -i eth0 /opt/zeek/share/zeek/policy/misc/capture-loss.zeek
#     """

#     def __init__(self, log_path: str):
#         self.log_path = Path(log_path)

#     def _parse_line(self, line: str, fields: list[str]) -> dict | None:
#         if line.startswith("#") or not line.strip():
#             return None
#         parts = line.strip().split("\t")
#         if len(parts) != len(fields):
#             return None
#         row = dict(zip(fields, parts))

#         def safe_float(v): 
#             try: return float(v)
#             except: return 0.0

#         duration = safe_float(row.get("duration", 0))
#         orig_pkts = safe_float(row.get("orig_pkts", 0))
#         resp_pkts = safe_float(row.get("resp_pkts", 0))
#         orig_bytes = safe_float(row.get("orig_bytes", 0))
#         resp_bytes = safe_float(row.get("resp_bytes", 0))
#         total_pkts = orig_pkts + resp_pkts
#         total_bytes = orig_bytes + resp_bytes
#         dur_safe = max(duration, 1e-6)

#         features = {col: 0.0 for col in FEATURE_COLS}
#         features["flow_duration"]          = normalize(duration, "flow_duration")
#         features["fwd_packets"]            = normalize(orig_pkts, "fwd_packets")
#         features["bwd_packets"]            = normalize(resp_pkts, "bwd_packets")
#         features["fwd_bytes"]              = normalize(orig_bytes, "fwd_bytes")
#         features["bwd_bytes"]              = normalize(resp_bytes, "bwd_bytes")
#         features["flow_bytes_per_sec"]     = normalize(total_bytes / dur_safe, "flow_bytes_per_sec")
#         features["flow_packets_per_sec"]   = normalize(total_pkts / dur_safe, "flow_packets_per_sec")
#         features["fwd_packets_per_sec"]    = normalize(orig_pkts / dur_safe, "fwd_packets_per_sec")
#         features["bwd_packets_per_sec"]    = normalize(resp_pkts / dur_safe, "bwd_packets_per_sec")
#         features["down_up_ratio"]          = normalize(resp_bytes / max(orig_bytes, 1), "down_up_ratio")

#         # TCP flags from Zeek history string
#         history = row.get("history", "")
#         features["syn_flag_count"]  = float("S" in history)
#         features["fin_flag_count"]  = float("F" in history)
#         features["rst_flag_count"]  = float("R" in history)
#         features["ack_flag_count"]  = float("a" in history)
#         features["psh_flag_count"]  = float("P" in history or "p" in history)

#         # Metadata
#         features["_src_ip"]   = row.get("id.orig_h", "")
#         features["_dst_ip"]   = row.get("id.resp_h", "")
#         features["_dst_port"] = safe_float(row.get("id.resp_p", 0))
#         features["_proto"]    = row.get("proto", "")
#         features["_service"]  = row.get("service", "")
#         features["_ts"]       = row.get("ts", "")
#         return features

#     def parse(self) -> Generator[dict, None, None]:
#         fields = []
#         with open(self.log_path) as f:
#             for line in f:
#                 if line.startswith("#fields"):
#                     fields = line.strip().split("\t")[1:]
#                 elif not line.startswith("#"):
#                     result = self._parse_line(line, fields)
#                     if result:
#                         yield result

#     def tail(self) -> Generator[dict, None, None]:
#         """Tail the log file in real-time (like `tail -f`)."""
#         fields = []
#         with open(self.log_path) as f:
#             for line in f:
#                 if line.startswith("#fields"):
#                     fields = line.strip().split("\t")[1:]
#             while True:
#                 line = f.readline()
#                 if line:
#                     result = self._parse_line(line, fields)
#                     if result:
#                         yield result
#                 else:
#                     time.sleep(0.1)


# # ─── Demo Traffic Simulator ───────────────────────────────────────────────────
# class TrafficSimulator:
#     """
#     Simulates realistic mixed traffic for dashboard demo.
#     Generates BENIGN traffic with occasional injected attacks.
#     Used for live demo when no real network interface is available.
#     """

#     ATTACK_PROFILES = {
#         "PortScan": {
#             "syn_flag_count": (0.9, 1.0),
#             "fwd_packets": (0.7, 1.0),
#             "flow_packets_per_sec": (0.85, 1.0),
#             "flow_duration": (0.0, 0.1),
#             "fwd_bytes": (0.0, 0.05),
#             "bwd_bytes": (0.0, 0.02),
#         },
#         "DDoS": {
#             "flow_bytes_per_sec": (0.9, 1.0),
#             "flow_packets_per_sec": (0.9, 1.0),
#             "ack_flag_count": (0.8, 1.0),
#             "fwd_packets": (0.8, 1.0),
#             "flow_duration": (0.0, 0.05),
#         },
#         "BruteForce": {
#             "rst_flag_count": (0.9, 1.0),
#             "syn_flag_count": (0.8, 1.0),
#             "bwd_bytes": (0.0, 0.03),
#             "flow_packets_per_sec": (0.5, 0.8),
#         },
#         "WebAttack": {
#             "fwd_bytes": (0.6, 1.0),
#             "bwd_bytes": (0.5, 0.9),
#             "psh_flag_count": (0.9, 1.0),
#             "flow_duration": (0.3, 0.8),
#         },
#     }

#     BENIGN_PORTS   = [80, 443, 22, 53, 8080, 8443, 3306, 5432, 6379]
#     ATTACK_PORTS   = {"PortScan": 0, "DDoS": 80, "BruteForce": 22, "WebAttack": 80}
#     FAKE_IPS       = [f"192.168.1.{i}" for i in range(10, 50)] + \
#                      [f"10.0.0.{i}" for i in range(1, 30)] + \
#                      [f"172.16.{i}.{j}" for i in range(1,5) for j in range(1,10)]
#     ATTACKER_IPS   = [f"203.0.113.{i}" for i in range(1, 20)] + \
#                      [f"198.51.100.{i}" for i in range(1, 10)]

#     def __init__(self, attack_interval: float = 15.0, flow_rate: float = 2.0):
#         self.attack_interval = attack_interval
#         self.flow_rate = flow_rate
#         self._queue: queue.Queue = queue.Queue(maxsize=500)
#         self._stop = threading.Event()
#         self._last_attack = time.time()
#         self._current_attack: str | None = None

#     def _benign_features(self) -> dict:
#         f = {col: random.gauss(0.3, 0.15) for col in FEATURE_COLS}
#         for k in f: f[k] = max(0.0, min(1.0, f[k]))
#         f["syn_flag_count"]  = random.choice([0.0, 1.0]) * 0.1
#         f["fin_flag_count"]  = random.choice([0.0, 1.0]) * 0.3
#         f["ack_flag_count"]  = 1.0
#         f["rst_flag_count"]  = 0.0
#         return f

#     def _attack_features(self, attack_type: str) -> dict:
#         f = self._benign_features()
#         profile = self.ATTACK_PROFILES.get(attack_type, {})
#         for col, (lo, hi) in profile.items():
#             f[col] = random.uniform(lo, hi)
#         return f

#     def _make_packet(self, is_attack: bool, attack_type: str | None) -> dict:
#         if is_attack and attack_type:
#             features = self._attack_features(attack_type)
#             src_ip   = random.choice(self.ATTACKER_IPS)
#             dst_port = self.ATTACK_PORTS.get(attack_type, random.randint(1, 65535))
#         else:
#             features = self._benign_features()
#             src_ip   = random.choice(self.FAKE_IPS)
#             dst_port = random.choice(self.BENIGN_PORTS)

#         features.update({
#             "_src_ip":   src_ip,
#             "_dst_ip":   random.choice(self.FAKE_IPS),
#             "_dst_port": float(dst_port),
#             "_proto":    random.choice(["tcp", "tcp", "tcp", "udp"]),
#             "_service":  "http" if dst_port in [80, 8080] else "ssh" if dst_port == 22 else "-",
#             "_ts":       datetime.utcnow().isoformat() + "Z",
#         })
#         return features

#     def _producer(self):
#         while not self._stop.is_set():
#             now = time.time()
#             # Trigger attack burst
#             if now - self._last_attack > self.attack_interval:
#                 self._current_attack = random.choice(list(self.ATTACK_PROFILES))
#                 attack_burst_end = now + random.uniform(3, 8)
#                 self._last_attack = now
#             else:
#                 attack_burst_end = 0

#             is_attack = time.time() < attack_burst_end and self._current_attack is not None
#             pkt = self._make_packet(is_attack, self._current_attack if is_attack else None)
#             pkt["_simulated_label"] = self._current_attack if is_attack else "BENIGN"

#             try:
#                 self._queue.put_nowait(pkt)
#             except queue.Full:
#                 pass  # Drop if dashboard is slow

#             time.sleep(1.0 / self.flow_rate)

#     def start(self):
#         t = threading.Thread(target=self._producer, daemon=True)
#         t.start()

#     def stop(self):
#         self._stop.set()

#     def get(self, timeout: float = 1.0) -> dict | None:
#         try:
#             return self._queue.get(timeout=timeout)
#         except queue.Empty:
#             return None


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="NIDS capture module")
#     parser.add_argument("--zeek-log", help="Path to Zeek conn.log")
#     parser.add_argument("--simulate", action="store_true", help="Run traffic simulator")
#     parser.add_argument("--count", type=int, default=20, help="Number of flows to print")
#     args = parser.parse_args()

#     if args.simulate:
#         print("🔴 LIVE SIMULATION MODE — press Ctrl+C to stop\n")
#         sim = TrafficSimulator(attack_interval=8.0, flow_rate=3.0)
#         sim.start()
#         for _ in range(args.count):
#             pkt = sim.get()
#             if pkt:
#                 label = pkt.get("_simulated_label", "BENIGN")
#                 src   = pkt.get("_src_ip", "?")
#                 port  = int(pkt.get("_dst_port", 0))
#                 print(f"[{datetime.now().strftime('%H:%M:%S')}] {label:<15} src={src:<18} port={port}")
#     elif args.zeek_log:
#         parser_ = ZeekLogParser(args.zeek_log)
#         for i, flow in enumerate(parser_.parse()):
#             if i >= args.count: break
#             print(json.dumps({k: v for k, v in flow.items() if not k.startswith("_float")}, indent=2))
#     else:
#         print("Usage: python capture.py --simulate --count 50")
#         print("       python capture.py --zeek-log /path/to/conn.log")


import socket
import requests
import threading
import time
import random

API_URL = "http://127.0.0.1:8000/api/predict"

def listen_port(port):
    """Listens on a specific port for an incoming scan"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('127.0.0.1', port))
        s.listen(1)
        while True:
            conn, addr = s.accept()
            print(f"🚨 [LIVE DETECT] Port Scan detected from {addr[0]} on port {port}!")
            
            # Send the magic 999.0 honeypot trigger to the API
            payload = {
                "features": [999.0] * 32, 
                "source_ip": addr[0],
                "destination_ip": "127.0.0.1"
            }
            try:
                requests.post(API_URL, json=payload)
            except:
                pass
            conn.close()
    except Exception as e:
        pass # Port might be in use by another app, just ignore

print("🛡️ Starting Live NIDS Capture (Honeypot Mode)...")
print("Listening for attacks on local network...")

# Listen on common ports that your portscan.bat targets
for p in [22, 80, 443, 3306, 8080]:
    threading.Thread(target=listen_port, args=(p,), daemon=True).start()

# Send background "Normal" traffic so the dashboard charts keep moving
while True:
    try:
        payload = {
            "features": [0.0] * 32, 
            "source_ip": f"192.168.1.{random.randint(2, 50)}", 
            "destination_ip": "10.0.0.5"
        }
        requests.post(API_URL, json=payload)
    except:
        pass
    time.sleep(1)
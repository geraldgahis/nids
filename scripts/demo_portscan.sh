#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# NIDS Live Demo Script — nmap Port Scan
# Run this inside a VM during your presentation.
# The dashboard will show HIGH-severity alerts within seconds.
# ─────────────────────────────────────────────────────────────────────────────

TARGET=${1:-"192.168.1.1"}
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║     NIDS LIVE DEMO — nmap Port Scan                  ║"
echo "║     Target: $TARGET                          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "⚠️  Make sure:"
echo "   1. NIDS backend is running: python backend/api.py"
echo "   2. Dashboard is open:       http://localhost:8501"
echo "   3. Zeek is capturing:       sudo zeek -i eth0"
echo ""
read -p "Press ENTER to start the port scan demo... "

echo ""
echo "🔴 [$(date +%H:%M:%S)] Launching nmap SYN scan on $TARGET..."
echo "   Command: nmap -sS -T4 -p 1-10000 $TARGET"
echo ""

# ─── Check nmap installed ────────────────────────────────────────────────────
if ! command -v nmap &>/dev/null; then
    echo "⚠️  nmap not found. Install with: sudo apt install nmap"
    echo ""
    echo "For demo purposes, simulating the scan output..."
    echo ""
    # Simulate nmap output for demo
    echo "Starting Nmap 7.94 ( https://nmap.org )"
    echo "Nmap scan report for $TARGET"
    echo "Host is up (0.0045s latency)."
    echo ""
    for port in 22 80 443 3306 5432 8080 8443; do
        echo "$(date +%H:%M:%S) Scanning port $port..."
        sleep 0.2
    done
    echo ""
    echo "PORT     STATE  SERVICE"
    echo "22/tcp   open   ssh"
    echo "80/tcp   open   http"
    echo "443/tcp  open   https"
    echo "3306/tcp closed mysql"
    echo ""
    echo "Nmap done: 1 IP address (1 host up) scanned in 3.21 seconds"
else
    # Real nmap scan
    nmap -sS -T4 -p 1-10000 --open "$TARGET" 2>&1 | while IFS= read -r line; do
        echo "$(date +%H:%M:%S) $line"
    done
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "✅ Scan complete. Check your dashboard for:"
echo "   🔴 HIGH — PortScan alerts"
echo "   📋 MITRE T1046: Network Service Discovery"
echo "   📍 Source IP flagged in Top Attackers"
echo "════════════════════════════════════════════════════════"
echo ""
echo "💡 Pro tip: The ML model detected the attack because nmap's"
echo "   SYN scan creates a signature of:"
echo "   - High syn_flag_count (0.95+)"
echo "   - Very short flow_duration (< 0.1s per connection)"
echo "   - Extreme fwd_packets_per_sec ratio"
echo "   - Minimal bwd_bytes (targets are not responding)"

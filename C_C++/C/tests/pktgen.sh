#!/bin/bash
# pktgen flood script
# Usage: ./pktgen.sh [interface] [dst_ip] [dst_mac] [duration_seconds] [pkt_size]

# make sure that pktgen kernel module is loaded with modprobe pktgen"
# ─── CONFIG (edit or pass as args) ───────────────────────────────────────────
IFACE="${1:-enp0s20f0u5u1u2}"
DST_IP="${2:-10.0.0.20}"
DST_MAC="${3:-2c:cf:67:f3:1b:7f}"
DURATION="${4:-180}"
PKT_SIZE="${5:-64}"
THREAD_NUM="${6:-3}"
THREAD="kpktgend_0"
PGDEV="/proc/net/pktgen"
# ─────────────────────────────────────────────────────────────────────────────

# colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; exit 1; }

# ─── CLEANUP on exit / Ctrl+C ────────────────────────────────────────────────
cleanup() {
  echo ""
  warn "Stopping pktgen..."
  echo "stop" > "$PGDEV/pgctrl" 2>/dev/null
  echo "rem_device_all" > "$PGDEV/$THREAD" 2>/dev/null
  log "Cleanup done."

  # Print final stats
  echo ""
  echo -e "${YELLOW}─── Final interface stats ───────────────────${NC}"
  ip -s link show "$IFACE"
  echo -e "${YELLOW}────────────────────────────────────────────${NC}"
}
trap cleanup EXIT INT TERM

# ─── CHECKS ──────────────────────────────────────────────────────────────────
[[ $EUID -ne 0 ]] && err "Run as root (sudo $0)"

ip link show "$IFACE" &>/dev/null || err "Interface '$IFACE' not found"

if [[ ! -d "$PGDEV" ]]; then
  log "Loading pktgen kernel module..."
  modprobe pktgen || err "Failed to load pktgen module"
fi

# ─── CONFIGURE ───────────────────────────────────────────────────────────────
log "Configuring pktgen on $IFACE..."
log "  Destination IP  : $DST_IP"
log "  Destination MAC : $DST_MAC"
log "  Packet size     : $PKT_SIZE bytes"
log "  Duration        : $DURATION seconds"

# Assign interface to thread
echo "rem_device_all"      > "$PGDEV/$THREAD"
echo "add_device $IFACE"   > "$PGDEV/$THREAD"

# Configure the device
pgset() { echo "$1" > "$PGDEV/$IFACE"; }

# Disable random size flag if you set it
echo "flag !TXSIZE_RND" > /proc/net/pktgen/$IFACE

# Disable random size flag if you set it

# Use multiple threads — one per CPU core
for i in $(seq 0 $THREAD_NUM); do  # adjust 3 to your core count - 1
    pgset "dst $DST_IP"
    pgset "dst_mac $DST_MAC"
    pgset "pkt_size $PKT_SIZE"
    pgset "count 0"          # 0 = infinite (we use sleep + stop instead)
    pgset "delay 0"          # no delay between packets = max speed
    #pgset "flag TXSIZE_RND"  # randomize TX size slightly (optional, more realistic)
done

# ─── START ───────────────────────────────────────────────────────────────────
echo ""
log "Starting flood for ${DURATION}s... (Ctrl+C to stop early)"
echo ""

echo "start" > "$PGDEV/pgctrl" &

# ─── LIVE STATS LOOP ─────────────────────────────────────────────────────────
START=$(date +%s)
while true; do
  NOW=$(date +%s)
  ELAPSED=$(( NOW - START ))
  REMAINING=$(( DURATION - ELAPSED ))

  [[ $ELAPSED -ge $DURATION ]] && break

  # Read pktgen counters
  PKTS=$(grep "^pkts-sofar"   "$PGDEV/$IFACE" 2>/dev/null | awk '{print $2}')
  ERRS=$(grep "^errors"       "$PGDEV/$IFACE" 2>/dev/null | awk '{print $2}')
  # Read kernel drop counters
  DROPS=$(ip -s link show "$IFACE" 2>/dev/null \
          | awk '/RX:/{getline; print $4}')

  printf "\r  ⏱  %3ds remaining | pkts sent: %-12s | tx errors: %-8s | rx dropped: %-8s" \
    "$REMAINING" "${PKTS:-0}" "${ERRS:-0}" "${DROPS:-0}"

  sleep 1
done

echo ""
log "Duration reached."
# cleanup() fires automatically via trap
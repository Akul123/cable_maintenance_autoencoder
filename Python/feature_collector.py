import subprocess
import argparse
import time
import csv
from pathlib import Path
from collections import deque
from statistics import mean
from glob import glob

DEFAULT_INTERVAL = 60.0
BITS_PER_BYTE = 8.0

BASE_FIELDS = [
    "timestamp", "iface", "carrier", "speed_mbps", "temp_c",
]

FEATURE_FIELDS = [
    "frame_err_ppm",
    "length_err_ppm",
    "rx_err_ppm",
    "speed_change_count_10m",
    "speed_is_downgraded",
    # optional if available NIC:
    # "pair_error_imbalance",
    "has_phy_stats",
    #rpi5
    "phy_receive_errors_rate",
    "phy_idle_errors_rate",
    "phy_serdes_ber_errors_rate",
    "phy_false_carrier_sense_errors_rate",
    "phy_local_rcvr_nok_rate",
    "phy_remote_rcv_nok_rate",
    # flextra
    "phy_receive_errors_copper",
    "phy_idle_errors",
    "link_flap",
    "time_since_last_link_down",
    "fcs_rate",
    "fcs_per_million_pkts",
    "rx_error_rate",
    "tx_error_rate",
    "bad_octets_rate",
    "fragment_rate",
    "jabber_rate",
    "oversize_rate",
    "undersize_rate",
    "port_discards_rate",
    "host_rx_drop_rate",
    "host_tx_drop_rate",
    "tx_dropped_rate",
    "host_rx_crc_rate",
    "rx_bps",
    "tx_bps",
    "pps",
    "utilization",
    "mean_fcs_per_million",
    "max_fcs_per_million",
    "mean_phy_rxerr",
    "max_phy_rxerr",
    "mean_discards",
    "max_discards",
    "flaps_10m",
    "flaps_1h",
    "mean_utilization",
    "mean_temp",
    "temp_delta_10m",
    "temp_slope_10m",
]

# Canonical metric -> possible ethtool key names across drivers
KEY_ALIASES = {
    "rx_bytes": ["rx_bytes", "rx_octets", "q0_rx_bytes"],
    "tx_bytes": ["tx_bytes", "tx_octets", "q0_tx_bytes"],
    "rx_packets": ["rx_packets", "rx_frames", "q0_rx_packets"],
    "tx_packets": ["tx_packets", "tx_frames", "q0_tx_packets"],

    "rx_errors": ["rx_errors"],
    "tx_errors": ["tx_errors"],

    "tx_dropped": ["tx_dropped", "q0_tx_dropped"],
    "rx_dropped": ["rx_dropped", "q0_rx_dropped"],

    "rx_crc_errors": ["rx_crc_errors", "rx_frame_check_sequence_errors"],
    "rx_frame_errors": ["rx_frame_errors", "rx_alignment_errors"],
    "rx_align_errors": ["rx_align_errors", "rx_alignment_errors"],
    "rx_length_errors": ["rx_length_errors", "rx_length_field_frame_errors"],
    "rx_missed_errors": ["rx_missed_errors", "rx_overruns", "rx_resource_errors"],
    "rx_no_buffer_count": ["rx_no_buffer_count", "rx_resource_errors"],

    "tx_carrier_errors": ["tx_carrier_errors", "tx_carrier_sense_errors"],
    "tx_timeout_count": ["tx_timeout_count"],
    "rx_dma_failed": ["rx_dma_failed"],
    "tx_dma_failed": ["tx_dma_failed"],
    "alloc_rx_buff_failed": ["alloc_rx_buff_failed"],

    "corr_ecc_errors": ["corr_ecc_errors"],
    "uncorr_ecc_errors": ["uncorr_ecc_errors"],

    # PHY
    "phy_receive_errors": ["phy_receive_errors", "phy_receive_errors_copper"],
    "phy_serdes_ber_errors": ["phy_serdes_ber_errors"],
    "phy_false_carrier_sense_errors": ["phy_false_carrier_sense_errors"],
    "phy_local_rcvr_nok": ["phy_local_rcvr_nok"],
    "phy_remote_rcv_nok": ["phy_remote_rcv_nok"],

    # DSA/switch-style ingress counters
    "in_bad_octets": ["in_bad_octets", "rx_symbol_errors"],
    "in_fragments": ["in_fragments", "rx_undersized_frames"],
    "in_jabber": ["in_jabber", "rx_jabbers"],
    "in_oversize": ["in_oversize", "rx_oversize_frames"],
    "in_undersize": ["in_undersize", "rx_undersized_frames"],
    "in_discards": ["in_discards", "rx_resource_errors", "rx_dropped", "q0_rx_dropped"],

    # Optional explicit fcs alias
    "in_fcs_error": ["in_fcs_error", "rx_crc_errors", "rx_frame_check_sequence_errors"],
}

def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return ""

def to_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default

def read_sys(iface, name, default=""):
    p = Path(f"/sys/class/net/{iface}/{name}")
    try:
        return p.read_text().strip()
    except Exception:
        return default

def read_first_float(path):
    try:
        return float(Path(path).read_text().strip())
    except Exception:
        return None

def read_host_temp_c():
    temps = []

    # Generic thermal zones
    for p in glob("/sys/class/thermal/thermal_zone*/temp"):
        v = read_first_float(p)
        if v is not None:
            temps.append(v / 1000.0 if v > 200 else v)  # milli Celsius -> Celsius

    # hwmon sensors (CPU/chipset/NIC if exposed)
    for p in glob("/sys/class/hwmon/hwmon*/temp*_input"):
        v = read_first_float(p)
        if v is not None:
            temps.append(v / 1000.0 if v > 200 else v) # milli Celsius -> Celsius

    if not temps:
        return ""
    # Conservative choice for risk monitoring
    return round(max(temps), 2)

def parse_ethtool_kv(text):
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        if line.lower().startswith("nic statistics"):
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        try:
            out[k] = int(v)
        except ValueError:
            continue
    return out

def ethtool_stats(iface):
    out = run_cmd(["ethtool", "-S", iface])
    return parse_ethtool_kv(out) if out else {}

def ethtool_phy_stats(iface):
    out = run_cmd(["ethtool", "--phy-statistics", iface])
    return parse_ethtool_kv(out) if out else {}

def ethtool_speed_mbps(iface, default=0):
    try:
        txt = run_cmd(["ethtool", iface])
        for line in txt.splitlines():
            s = line.strip()
            if s.startswith("Speed:"):
                raw = s.split(":", 1)[1].strip().replace("Mb/s", "")
                return int(raw) if raw.isdigit() else default
    except Exception:
        pass
    return default

def pick(stats: dict, *keys, default=None):
    for key in keys:
        for alias in KEY_ALIASES.get(key, [key]):
            if alias in stats:
                return stats[alias]
    return default

def as_num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def nonneg_delta(curr, prev):
    c = as_num(curr)
    p = as_num(prev)
    if c is None or p is None:
        return None
    d = c - p
    if d < 0:
        return None
    return d

def safe_rate(curr, prev, dt_sec):
    if dt_sec is None or dt_sec <= 0:
        return None
    c = as_num(curr)
    p = as_num(prev)
    if c is None or p is None:
        return None
    d = c - p
    if d < 0:   # counter reset/rollover
        return None
    return d / dt_sec

def first_present(stats, keys):
    for k in keys:
        v = stats.get(k)
        if v is not None:
            return v
    return None

def sum_present(stats, keys):
    vals = []
    for k in keys:
        v = stats.get(k)
        if v is not None:
            vals.append(v)
    return sum(vals) if vals else None

def get_rx_errors(port_stats):
    # Prefer direct counter if driver exposes it
    direct = first_present(port_stats, ["rx_errors"])
    if direct is not None:
        return direct

    # Fallback: derive from detailed RX error counters
    return sum_present(port_stats, [
        "rx_frame_check_sequence_errors",   # CRC/FCS
        "rx_alignment_errors",
        "rx_length_field_frame_errors",
        "rx_symbol_errors",
        "rx_overruns",
        "rx_resource_errors",
    ])

def get_tx_errors(port_stats):
    # Prefer direct counter if present
    direct = first_present(port_stats, ["tx_errors"])
    if direct is not None:
        return direct

    # Fallback: derive from detailed TX error counters
    return sum_present(port_stats, [
        "tx_underrun",
        "tx_single_collision_frames",
        "tx_multiple_collision_frames",
        "tx_excessive_collisions",
        "tx_late_collisions",
        "tx_carrier_sense_errors",
    ])

class Rolling:
    def __init__(self, seconds):
        self.seconds = seconds
        self.buf = deque()  # (ts, value)

    def push(self, ts, value):
        self.buf.append((ts, value))
        cutoff = ts - self.seconds
        while self.buf and self.buf[0][0] < cutoff:
            self.buf.popleft()

    def values(self):
        return [v for _, v in self.buf if v is not None]

    def sum(self):
        vals = self.values()
        return sum(vals) if vals else 0.0

    def avg(self):
        vals = self.values()
        return mean(vals) if vals else None

    def max(self):
        vals = self.values()
        return max(vals) if vals else None

    def first(self):
        vals = self.values()
        return vals[0] if vals else None

class CableFeatureState:
    def __init__(self):
        self.prev_ts = None
        self.prev_phy = {}
        self.prev_port = {}
        self.prev_host = {}
        self.prev_carrier = None
        self.last_link_down_ts = None
        self.prev_speed_mbps = None

        # 10m windows
        self.w_speed_change_10m = Rolling(600)
        self.w_fcs_ppm_10m = Rolling(600)
        self.w_phy_rxerr_10m = Rolling(600)
        self.w_discards_10m = Rolling(600)
        self.w_util_10m = Rolling(600)
        self.w_temp_10m = Rolling(600)
        self.w_link_flap_10m = Rolling(600)

        # 1h window
        self.w_link_flap_1h = Rolling(3600)

def compute_cable_features(
    state: CableFeatureState,
    ts: float,
    phy_stats: dict,
    port_stats: dict,
    host_stats: dict,
    carrier: int,
    speed_mbps: int,
    temp_c: float | None
):
    feats = {}
    dt_sec = 0 if state.prev_ts is None else (ts - state.prev_ts) # delta time

    # ---------- PHY ----------
    has_phy = 1 if phy_stats else 0
    feats["has_phy_stats"] = has_phy

    phy_rx = pick(phy_stats, "phy_receive_errors")

    phy_serdes_ber_errors = pick(phy_stats, "phy_serdes_ber_errors")
    phy_false_carrier_sense_errors = pick(phy_stats, "phy_false_carrier_sense_errors")
    phy_local_rcvr_nok = pick(phy_stats, "phy_local_rcvr_nok")
    phy_remote_rcv_nok = pick(phy_stats, "phy_remote_rcv_nok")
    prev_phy_rx = pick(state.prev_phy, "phy_receive_errors_copper", "phy_receive_errors")
    prev_phy_serdes_ber_errors = pick(state.prev_phy, "phy_serdes_ber_errors")
    prev_phy_false_carrier_sense_errors = pick(state.prev_phy, "phy_false_carrier_sense_errors")
    prev_phy_local_rcvr_nok = pick(state.prev_phy, "phy_local_rcvr_nok")
    prev_phy_remote_rcv_nok = pick(state.prev_phy, "phy_remote_rcv_nok")

    feats["phy_receive_errors_rate"] = safe_rate(phy_rx, prev_phy_rx, dt_sec)
    feats["phy_serdes_ber_errors_rate"] = safe_rate(phy_serdes_ber_errors, prev_phy_serdes_ber_errors, dt_sec)
    feats["phy_false_carrier_sense_errors_rate"] = safe_rate(phy_false_carrier_sense_errors, prev_phy_false_carrier_sense_errors, dt_sec)
    feats["phy_local_rcvr_nok_rate"] = safe_rate(phy_local_rcvr_nok, prev_phy_local_rcvr_nok, dt_sec)
    feats["phy_remote_rcv_nok_rate"] = safe_rate(phy_remote_rcv_nok, prev_phy_remote_rcv_nok, dt_sec)

    # ---------- Link stability ----------
    link_flap = 0
    if state.prev_carrier is not None and carrier != state.prev_carrier:
        link_flap = 1
    feats["link_flap"] = link_flap

    if carrier == 0:
        state.last_link_down_ts = ts

    feats["time_since_last_link_down"] = (
        0.0 if state.last_link_down_ts is None else (ts - state.last_link_down_ts)
    )

    # ---------- Port MAC integrity ----------
    in_fcs = pick(port_stats, "in_fcs_error", "rx_crc_errors", "rx_frame_check_sequence_errors")
    in_rx_err = get_rx_errors(port_stats)
    out_tx_err = get_tx_errors(port_stats)
    in_bad_octets = pick(port_stats, "in_bad_octets")
    in_frag = pick(port_stats, "in_fragments")
    in_jabber = pick(port_stats, "in_jabber")
    in_oversize = pick(port_stats, "in_oversize")
    in_undersize = pick(port_stats, "in_undersize")
    in_discards = pick(port_stats, "in_discards")
    rx_packets = pick(port_stats, "rx_packets")

    p_in_fcs = pick(state.prev_port, "in_fcs_error", "rx_crc_errors", "rx_frame_check_sequence_errors")
    p_in_rx_err = get_rx_errors(state.prev_port)
    p_out_tx_err = get_tx_errors(state.prev_port)
    p_in_bad_octets = pick(state.prev_port, "in_bad_octets")
    p_in_frag = pick(state.prev_port, "in_fragments")
    p_in_jabber = pick(state.prev_port, "in_jabber")
    p_in_oversize = pick(state.prev_port, "in_oversize")
    p_in_undersize = pick(state.prev_port, "in_undersize")
    p_in_discards = pick(state.prev_port, "in_discards")
    p_rx_packets = pick(state.prev_port, "rx_packets")

    # frame/length/rx ppm (per million RX packets)
    in_frame_err = pick(port_stats, "rx_frame_errors", "rx_alignment_errors")
    in_len_err = pick(port_stats, "rx_length_errors", "rx_length_field_frame_errors")
    p_in_frame_err = pick(state.prev_port, "rx_frame_errors", "rx_alignment_errors")
    p_in_len_err = pick(state.prev_port, "rx_length_errors", "rx_length_field_frame_errors")

    frame_delta = nonneg_delta(in_frame_err, p_in_frame_err)
    len_delta = nonneg_delta(in_len_err, p_in_len_err)
    rx_err_delta = nonneg_delta(in_rx_err, p_in_rx_err)
    rx_pkts_delta = nonneg_delta(rx_packets, p_rx_packets)

    speed_changed = 0
    speed_downgraded = 0
    if state.prev_speed_mbps is not None and speed_mbps is not None:
        speed_changed = int(speed_mbps != state.prev_speed_mbps)
        speed_downgraded = int(speed_mbps < state.prev_speed_mbps)

    state.w_speed_change_10m.push(ts, speed_changed)
    feats["speed_change_count_10m"] = int(state.w_speed_change_10m.sum())
    feats["speed_is_downgraded"] = speed_downgraded

    den = max(rx_pkts_delta, 1.0) if rx_pkts_delta is not None else None
    feats["frame_err_ppm"] = (1_000_000.0 * frame_delta / den) if (den is not None and frame_delta is not None) else None
    feats["length_err_ppm"] = (1_000_000.0 * len_delta / den) if (den is not None and len_delta is not None) else None
    feats["rx_err_ppm"] = (1_000_000.0 * rx_err_delta / den) if (den is not None and rx_err_delta is not None) else None

    feats["fcs_rate"] = safe_rate(in_fcs, p_in_fcs, dt_sec)
    if dt_sec > 0 and None not in (rx_packets, p_rx_packets, in_fcs, p_in_fcs):
        rx_pkts_delta = nonneg_delta(rx_packets, p_rx_packets)
        fcs_delta = nonneg_delta(in_fcs, p_in_fcs)

        if dt_sec > 0 and rx_pkts_delta is not None and fcs_delta is not None:
            feats["fcs_per_million_pkts"] = 1_000_000.0 * fcs_delta / max(rx_pkts_delta, 1.0)
        else:
            feats["fcs_per_million_pkts"] = None

    else:
        feats["fcs_per_million_pkts"] = None


    feats["rx_error_rate"] = safe_rate(in_rx_err, p_in_rx_err, dt_sec)
    feats["tx_error_rate"] = safe_rate(out_tx_err, p_out_tx_err, dt_sec)
    feats["bad_octets_rate"] = safe_rate(in_bad_octets, p_in_bad_octets, dt_sec)
    feats["fragment_rate"] = safe_rate(in_frag, p_in_frag, dt_sec)
    feats["jabber_rate"] = safe_rate(in_jabber, p_in_jabber, dt_sec)
    feats["oversize_rate"] = safe_rate(in_oversize, p_in_oversize, dt_sec)
    feats["undersize_rate"] = safe_rate(in_undersize, p_in_undersize, dt_sec)
    feats["port_discards_rate"] = safe_rate(in_discards, p_in_discards, dt_sec)

    # ---------- Host context ----------
    host_rx_drop = pick(host_stats, "IEEE_rx_drop", "rx_dropped")
    host_tx_drop = pick(host_stats, "IEEE_tx_drop", "tx_dropped")
    tx_dropped = pick(host_stats, "tx_dropped")
    host_rx_crc = pick(host_stats, "rx_crc_errors")

    p_host_rx_drop = pick(state.prev_host, "IEEE_rx_drop", "rx_dropped")
    p_host_tx_drop = pick(state.prev_host, "IEEE_tx_drop", "tx_dropped")
    p_tx_dropped = pick(state.prev_host, "tx_dropped")
    p_host_rx_crc = pick(state.prev_host, "rx_crc_errors")

    feats["host_rx_drop_rate"] = safe_rate(host_rx_drop, p_host_rx_drop, dt_sec)
    feats["host_tx_drop_rate"] = safe_rate(host_tx_drop, p_host_tx_drop, dt_sec)
    feats["tx_dropped_rate"] = safe_rate(tx_dropped, p_tx_dropped, dt_sec)
    feats["host_rx_crc_rate"] = safe_rate(host_rx_crc, p_host_rx_crc, dt_sec)

    # ---------- Traffic context ----------
    rx_bytes = pick(port_stats, "rx_bytes")
    tx_bytes = pick(port_stats, "tx_bytes")
    rx_packets = pick(port_stats, "rx_packets")
    tx_packets = pick(port_stats, "tx_packets")
    p_rx_bytes = pick(state.prev_port, "rx_bytes")
    p_tx_bytes = pick(state.prev_port, "tx_bytes")
    p_tx_packets = pick(state.prev_port, "tx_packets")

    rx_bps = None if (dt_sec <= 0 or None in (rx_bytes, p_rx_bytes)) else max(rx_bytes - p_rx_bytes, 0) * BITS_PER_BYTE / dt_sec # bytes per second
    tx_bps = None if (dt_sec <= 0 or None in (tx_bytes, p_tx_bytes)) else max(tx_bytes - p_tx_bytes, 0) * BITS_PER_BYTE / dt_sec # bytes per second
    pps = None if (dt_sec <= 0 or None in (rx_packets, p_rx_packets, tx_packets, p_tx_packets)) else (
        max(rx_packets - p_rx_packets, 0) + max(tx_packets - p_tx_packets, 0)
    ) / dt_sec

    util = None
    if rx_bps is not None and tx_bps is not None and speed_mbps > 0:
        util = (rx_bps + tx_bps) / (speed_mbps * 1_000_000.0)

    feats["rx_bps"] = rx_bps
    feats["tx_bps"] = tx_bps
    feats["pps"] = pps
    feats["utilization"] = util
    feats["temp_c"] = temp_c

    # ---------- Window updates ----------
    state.w_fcs_ppm_10m.push(ts, feats["fcs_per_million_pkts"])
    state.w_phy_rxerr_10m.push(ts, feats["phy_receive_errors_rate"])
    state.w_discards_10m.push(ts, feats["port_discards_rate"])
    state.w_util_10m.push(ts, feats["utilization"])
    state.w_temp_10m.push(ts, temp_c)
    state.w_link_flap_10m.push(ts, link_flap)
    state.w_link_flap_1h.push(ts, link_flap)

    feats["mean_fcs_per_million"] = state.w_fcs_ppm_10m.avg()
    feats["max_fcs_per_million"] = state.w_fcs_ppm_10m.max()
    feats["mean_phy_rxerr"] = state.w_phy_rxerr_10m.avg()
    feats["max_phy_rxerr"] = state.w_phy_rxerr_10m.max()
    feats["mean_discards"] = state.w_discards_10m.avg()
    feats["max_discards"] = state.w_discards_10m.max()
    feats["flaps_10m"] = int(state.w_link_flap_10m.sum())
    feats["flaps_1h"] = int(state.w_link_flap_1h.sum())
    feats["mean_utilization"] = state.w_util_10m.avg()
    feats["mean_temp"] = state.w_temp_10m.avg()

    t0 = state.w_temp_10m.first()
    feats["temp_delta_10m"] = None if (temp_c is None or t0 is None) else (temp_c - t0)
    feats["temp_slope_10m"] = None if feats["temp_delta_10m"] is None else feats["temp_delta_10m"] / 600.0

    # ---------- advance state ----------
    state.prev_ts = ts
    state.prev_phy = dict(phy_stats)
    state.prev_port = dict(port_stats)
    state.prev_host = dict(host_stats)
    state.prev_carrier = carrier
    state.prev_speed_mbps = speed_mbps

    return feats

state = CableFeatureState()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", required=True, help="e.g. enp0s31f6")
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    ap.add_argument("--output", default="collector.csv")
    ap.add_argument("--count", type=int, default=0, help="0 = run forever")
    args = ap.parse_args()

    fields = BASE_FIELDS + FEATURE_FIELDS
    samples = 0
    if args.iface.startswith("wl"):
        print("Warning: wlan interface usually does not support --phy-statistics; PHY features will be empty.")

    print(f"Collecting features on interface: {args.iface}, interval: {args.interval}, output file: {args.output}")
    with open(args.output, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if f.tell() == 0:
            w.writeheader()

        # loop:
        while True:
            ts = time.time()
            phy = ethtool_phy_stats(args.iface)      # e.g. eth0, XP1...
            port = ethtool_stats(args.iface)         # same port
            host = ethtool_stats(args.iface)         # masterif in DSA
            speed_mbps = ethtool_speed_mbps(args.iface)
            carrier = to_int(read_sys(args.iface, "carrier", "0"), 0)  # replace with /sys/class/net/<iface>/carrier
            temp_c = read_host_temp_c()

            features = compute_cable_features(
                state, ts, phy, port, host, carrier, speed_mbps, temp_c
            )
            row = {
                "timestamp": ts,
                "iface": args.iface,
                "carrier": carrier,
                "speed_mbps": speed_mbps,
                "temp_c": temp_c,
                **features,
            }
            w.writerow(row)
            f.flush()   # force write to disk each iteration

            samples += 1
            if args.count > 0 and samples >= args.count:
                break
            time.sleep(args.interval)

if __name__ == "__main__":
    main()

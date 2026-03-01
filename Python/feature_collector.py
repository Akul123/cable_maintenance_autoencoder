#!/usr/bin/env python3
# Use it:
# python3 cable_collector.py --iface enp0s31f6 --interval 10 --output enp0s31f6_metrics.csv
# python3 cable_collector.py --iface enp0s31f6 --interval 10 --count 360 --output enp0s31f6_metrics.csv

import argparse
import csv
import datetime as dt
import subprocess
import time
from pathlib import Path
from glob import glob

DEFAULT_INTERVAL = 60.0

STAT_KEYS = [
    "rx_packets", "tx_packets", "rx_bytes", "tx_bytes",
    "rx_errors", "tx_errors", "tx_dropped", "collisions",
    "rx_crc_errors", "rx_frame_errors", "rx_align_errors",
    "rx_length_errors", "rx_long_length_errors", "rx_short_length_errors",
    "rx_over_errors", "rx_missed_errors", "rx_no_buffer_count",
    "tx_carrier_errors", "tx_aborted_errors", "tx_fifo_errors",
    "tx_window_errors", "tx_abort_late_coll", "tx_timeout_count",
    "tx_restart_queue", "rx_dma_failed", "tx_dma_failed",
    "alloc_rx_buff_failed", "corr_ecc_errors", "uncorr_ecc_errors",
]

RATE_KEYS = [
    "rx_crc_errors", "rx_frame_errors", "rx_align_errors",
    "rx_missed_errors", "rx_no_buffer_count",
    "tx_carrier_errors", "tx_timeout_count",
    "rx_errors", "tx_errors", "tx_dropped",
    "corr_ecc_errors", "uncorr_ecc_errors",
]

def run(cmd):
    return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)

def read_sys(iface, name, default=""):
    p = Path(f"/sys/class/net/{iface}/{name}")
    try:
        return p.read_text().strip()
    except Exception:
        return default

def to_int(v, default=0):
    try:
        return int(v)
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

def ethtool_stats(iface):
    out = run(["ethtool", "-S", iface])
    stats = {}
    for line in out.splitlines():
        line = line.strip()
        if ":" not in line or line.startswith("NIC statistics"):
            continue
        k, v = line.split(":", 1)
        stats[k.strip()] = to_int(v.strip(), 0)
    return stats

def ethtool_link(iface):
    speed, duplex = "", ""
    try:
        out = run(["ethtool", iface])
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("Speed:"):
                speed = s.split(":", 1)[1].strip().replace("Mb/s", "")
            elif s.startswith("Duplex:"):
                duplex = s.split(":", 1)[1].strip().lower()
    except Exception:
        pass
    if not speed:
        speed = read_sys(iface, "speed", "")
    if not duplex:
        duplex = read_sys(iface, "duplex", "").lower()
    return to_int(speed, 0), duplex or "unknown"

def collect_once(iface):
    stats = ethtool_stats(iface)
    row = {k: stats.get(k, 0) for k in STAT_KEYS}
    speed, duplex = ethtool_link(iface)
    row.update({
        "timestamp": dt.datetime.now().isoformat(timespec="seconds") + "Z",
        "interface": iface,
        "operstate": read_sys(iface, "operstate", "unknown"),
        "carrier": to_int(read_sys(iface, "carrier", "0"), 0),
        "speed_mbps": speed,
        "duplex": duplex,
        "temp_c": read_host_temp_c(),
    })
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", required=True, help="e.g. enp0s31f6")
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    ap.add_argument("--output", default="collector.csv")
    ap.add_argument("--count", type=int, default=0, help="0 = run forever")
    args = ap.parse_args()

    base_fields = ["timestamp", "interface", "operstate", "carrier", "speed_mbps", "duplex", "temp_c", "link_flap_event"]
    raw_fields = STAT_KEYS
    rate_fields = [f"{k}_rate" for k in RATE_KEYS]
    fields = base_fields + raw_fields + rate_fields + ["temp_rate_c_per_sec"]

    prev = None
    prev_ts = None
    samples = 0

    print(f"Collecting features on interface: {args.iface}, interval: {args.interval}, output file: {args.output}")

    with open(args.output, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if f.tell() == 0:
            w.writeheader()

        while True:
            now = time.time()
            row = collect_once(args.iface)

            # link flap event
            row["link_flap_event"] = 0
            if prev is not None and row["carrier"] != prev["carrier"]:
                row["link_flap_event"] = 1

            # rate features (delta/sec)
            dt_sec = (now - prev_ts) if prev_ts else 0.0
            for k in RATE_KEYS:
                rk = f"{k}_rate"
                if prev is None or dt_sec <= 0:
                    row[rk] = ""
                else:
                    d = row[k] - prev[k]
                    if d < 0:  # counter reset/rollover
                        row[rk] = ""
                    else:
                        row[rk] = round(d / dt_sec, 6)

            # temperature rate (delta C / sec)
            if prev is None or dt_sec <= 0 or row["temp_c"] == "" or prev["temp_c"] == "":
                row["temp_rate_c_per_sec"] = ""
            else:
                row["temp_rate_c_per_sec"] = round((row["temp_c"] - prev["temp_c"]) / dt_sec, 6)

            w.writerow(row)
            f.flush()

            prev, prev_ts = row, now
            samples += 1
            if args.count > 0 and samples >= args.count:
                break
            time.sleep(args.interval)

if __name__ == "__main__":
    main()

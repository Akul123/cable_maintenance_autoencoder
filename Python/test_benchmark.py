import numpy as np
import re
from ai_edge_litert.interpreter import Interpreter
from time import perf_counter_ns
import os
from configs import CableMaintenanceConfig as Config
import pandas as pd
import psutil
import json
import tensorflow as tf

DEBUG = 1
WARMUP_RUNS = 100
BENCHMARK_RUNS = 18000

batch_size = 1 # can be 1 for single sample
proc = psutil.Process(os.getpid())

feature_cols: tuple = (
    "frame_err_ppm",
    "length_err_ppm",
    "speed_change_count_10m",
    "speed_is_downgraded",
    "rx_err_ppm",
    "phy_receive_errors_rate",    # rpi5
    #"phy_idle_errors_rate",
    "phy_serdes_ber_errors_rate", # rpi5
    "fcs_per_million_pkts",
    "rx_error_rate",
    "host_rx_crc_rate",
    "tx_dropped_rate",
    "utilization",
    "flaps_10m",
    "temp_slope_10m", #if it is not temp from NIC [/sys/class/hwmon/hwmonX/temp1_input] drop this feature
)

def read_header(csv_path: str):
    with tf.io.gfile.GFile(csv_path, "r") as f:
        header = f.readline().strip().split(",")
    return header

def load_csv_features(csv_path: str) -> tf.Tensor:
    header = read_header(csv_path)
    col_to_idx = {c: i for i, c in enumerate(header)}
    idx = [col_to_idx[c] for c in feature_cols]
    print(f"Indexes from dataset { idx }")

    ds = tf.data.TextLineDataset(csv_path).skip(1)

    def parse_line(line):
        parts = tf.strings.split(line, ",")
        vals = tf.gather(parts, idx)
        vals = tf.where(vals == "", tf.fill(tf.shape(vals), "nan"), vals)
        x = tf.strings.to_number(vals, out_type=tf.float32)  # NaN for missing
        miss = tf.cast(tf.math.is_nan(x), tf.float32)
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        if Config.add_missing_indicators:
            x = tf.concat([x, miss], axis=0)
        return x

    x_list = list(ds.map(parse_line))
    if not x_list:
        raise ValueError(f"No rows found in {csv_path}")
    X = tf.stack(x_list, axis=0)
    return X

def split_train_val(X: tf.Tensor):
    n = tf.shape(X)[0]
    n_val = tf.cast(tf.round(tf.cast(n, tf.float32) * Config.val_ratio), tf.int32)
    idx = tf.random.shuffle(tf.range(n), seed=Config.seed)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return tf.gather(X, tr_idx), tf.gather(X, val_idx)

def load_interpreter(path):
    itp = Interpreter(model_path=path)
    itp.allocate_tensors()
    return itp

# before loading model
rss_before = proc.memory_info().rss
cpu_before = proc.cpu_times()

name = os.path.basename(Config.MODEL_TO_TEST)
print(f"\n=== {name} ===")
interpreter = load_interpreter(Config.MODEL_TO_TEST)

# after loading model
rss_after = proc.memory_info().rss
peak_rss = max(rss_before, rss_after)  # coarse in-process peak
cpu_after = proc.cpu_times()

print(f"--MODEL LOAD: CPU & MEMORY--")
print(f"\t RAM start: {rss_before / 1024 / 1024} MB")
print(f"\t RAM end: {rss_after / 1024 / 1024} MB")
print(f"\t RAM delta: {(rss_after - rss_before)/1024/1024} MB")
print(f"\t RAM peak: {peak_rss / 1024 / 1024} MB")
print(f"\t CPU userspace : {cpu_after.user - cpu_before.user} s")
print(f"\t CPU system: {cpu_after.system - cpu_before.system} s")

if DEBUG:
    for i, d in enumerate(interpreter.get_input_details()):
        print(f"IN  {i}: {d['name']} {d['shape']} {d['dtype']}")
    for i, d in enumerate(interpreter.get_output_details()):
        print(f"OUT {i}: {d['name']} {d['shape']} {d['dtype']}")

# -----------------------------
# Inspect input/output details
# -----------------------------
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

if DEBUG:
    print("Inputs:")
    for inp in input_details:
        print(inp)

    print("\nOutputs:")
    for out in output_details:
        print(out)

# Load data
# --------------------
X = load_csv_features(Config.train_csv)
X_train, X_val = split_train_val(X)

# -----------------------------
# Invoke model
# -----------------------------

# before inference
rss_before = proc.memory_info().rss
cpu_before = proc.cpu_times()

n_samples = X_val.shape[0]
n_eval_benchmark = min(BENCHMARK_RUNS, n_samples)
n_eval_warmup = min(WARMUP_RUNS, n_samples)  # avoid wrap mismatch for scoring
latency_warmup_us = np.empty(n_eval_warmup, dtype=np.float64)
latency_benchmark_us = np.empty(n_eval_benchmark, dtype=np.float64)
outputs = np.empty(n_eval_benchmark, dtype=object)

# Warmup
for i in range(n_eval_warmup):
    idx = i % n_eval_warmup

    # set tensors for this sample
    x = X_val[i].numpy().astype(input_details["dtype"], copy=False).reshape(1, -1)
    interpreter.set_tensor(input_details["index"], x)

    t0 = perf_counter_ns()
    interpreter.invoke()
    t1 = perf_counter_ns()
    latency_warmup_us[i] = (t1 - t0) / 1000.0

i=0
# BENCHMARK
  # avoid wrap mismatch for scoring
for i in range(n_eval_benchmark):
    idx = i % n_eval_benchmark  # sequential wrap-around

    # set tensors for this sample
    x = X_val[i].numpy().astype(input_details["dtype"], copy=False).reshape(1, -1)
    interpreter.set_tensor(input_details["index"], x)

    t0 = perf_counter_ns()
    interpreter.invoke()
    t1 = perf_counter_ns()
    latency_benchmark_us[i] = (t1 - t0) / 1000.0

    # get outputs
    outputs[i] = interpreter.get_tensor(output_details["index"])

# after inference
rss_after = proc.memory_info().rss
peak_rss = max(rss_before, rss_after)  # coarse in-process peak
cpu_after = proc.cpu_times()

print(f"--INFERENCE percentils--")
print(f"runs = {BENCHMARK_RUNS}, warmup = {WARMUP_RUNS} times:")
print(f"\t mean = {latency_benchmark_us.mean():.2f} us")
print(f"\t p50 = {np.percentile(latency_benchmark_us, 50):.2f} us")
print(f"\t p95 = {np.percentile(latency_benchmark_us, 95):.2f} us")
print(f"\t p99 = {np.percentile(latency_benchmark_us, 99):.2f} us")
print(f"\t min = {latency_benchmark_us.min():.2f} us")
print(f"\t max = {latency_benchmark_us.max():.2f} us")

print(f"--INFERENCE cpu & memory--")
print(f"\t RAM start: {rss_before / 1024 / 1024} MB")
print(f"\t RAM end: {rss_after / 1024 / 1024} MB")
print(f"\t RAM delta: {(rss_after - rss_before)/1024/1024} MB")
print(f"\t RAM peak: {peak_rss / 1024 / 1024} MB")
print(f"\t CPU userspace : {cpu_after.user - cpu_before.user} s")
print(f"\t CPU system: {cpu_after.system - cpu_before.system} s")

n_eval = min(BENCHMARK_RUNS, n_samples)  # avoid wrap mismatch for scoring
sample_outputs = outputs[:n_eval]        # each entry: list of 10 tensors, each (1, d)

#Load threshold
with tf.io.gfile.GFile(f"{Config.OUTPUT_JSON}", "r") as f:
    threshold = float(json.load(f)["threshold"])

# x shape: (1, n_features), y shape: (1, n_features)
# y = interpreter.get_tensor(output_details["index"])
# err = np.mean((x.astype(np.float32) - y.astype(np.float32)) ** 2, axis=1)[0]
# after running all n_eval inferences and filling outputs[i] with shape (1, n_features)
x_true = X_val[:n_eval].numpy().astype(np.float32)            # (n_eval, n_features)
y_pred = np.concatenate(outputs[:n_eval], axis=0).astype(np.float32)  # (n_eval, n_features)

# all inputs vs matching outputs (row-wise)
err = np.mean((x_true - y_pred) ** 2, axis=1)                 # (n_eval,)
anomalies = err > threshold

count = np.sum(anomalies > threshold)
print(f"Anomalies: {count}/{anomalies.size}")

# -----------------------------
# Read outputs
# -----------------------------
if DEBUG:
    outputs = interpreter.get_tensor(output_details["index"])

    # -----------------------------
    # Print output shapes
    # -----------------------------
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")
        print(out)  # optionally print values

print(f"================")
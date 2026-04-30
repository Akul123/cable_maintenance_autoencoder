"""
NIC Health Classifier
---------------------
Features:
  NIC layer  : temperature, link speed, utilization (rx+tx)
  Cable layer: crc_error_rate, frame_error_rate, length_error_rate,
               symbol_error_rate (physical copper), idle_error_rate

Pipeline:
  1. Simulate 1000 normal + 50 anomalous samples
  2. Isolation Forest → pseudo-labels
  3. Random Forest classifier → trained model
  4. Feature importance → tells you which layer is failing
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────
# 1. SIMULATE DATA
#    Normal: tight distributions around healthy operating point
#    Anomalous: several failure modes mixed together
# ─────────────────────────────────────────────────────────────────

def make_normal(n):
    """Healthy NIC — all counters near zero except utilization varies"""
    link_speed = np.random.choice([1000, 10000, 25000], n)  # Mbps
    utilization = np.random.uniform(0.05, 0.75, n)          # 5–75% of link

    return pd.DataFrame({
        # NIC layer
        'nic_temp_c':         np.random.normal(52, 3, n),
        'link_speed_mbps':    link_speed,
        'utilization_pct':    utilization * 100,
        'tx_util_pct':        utilization * np.random.uniform(0.3, 1.0, n) * 100,

        # Cable layer — rates per million packets (ppm)
        # Healthy NICs have near-zero error rates
        'crc_error_rate_ppm':     np.abs(np.random.normal(0.0, 0.3, n)),
        'frame_error_rate_ppm':   np.abs(np.random.normal(0.0, 0.2, n)),
        'length_error_rate_ppm':  np.abs(np.random.normal(0.0, 0.1, n)),
        'symbol_error_rate_ppm':  np.abs(np.random.normal(0.0, 0.15, n)),
        'idle_error_rate_ppm':    np.abs(np.random.normal(0.0, 0.2, n)),

        'label': 'normal'
    })


def make_anomalous(n):
    """
    Three realistic failure modes, mixed:
      A) Degraded cable    — CRC + frame + symbol errors elevated
      B) NIC overheating   — temp high, utilization unstable
      C) Duplex mismatch   — frame errors + length errors, speed looks ok
    """
    third = n // 3
    sizes = [third, third, n - 2*third]
    frames = []

    # Mode A: degraded cable / bad connector
    na = sizes[0]
    link_speed = np.random.choice([1000, 10000], na)
    frames.append(pd.DataFrame({
        'nic_temp_c':             np.random.normal(54, 4, na),
        'link_speed_mbps':        link_speed,
        'utilization_pct':        np.random.uniform(0.05, 0.70, na) * 100,
        'tx_util_pct':            np.random.uniform(0.05, 0.60, na) * 100,
        'crc_error_rate_ppm':     np.abs(np.random.normal(18, 6, na)),   # ← elevated
        'frame_error_rate_ppm':   np.abs(np.random.normal(12, 4, na)),   # ← elevated
        'length_error_rate_ppm':  np.abs(np.random.normal(1.5, 0.5, na)),
        'symbol_error_rate_ppm':  np.abs(np.random.normal(9, 3, na)),    # ← elevated
        'idle_error_rate_ppm':    np.abs(np.random.normal(5, 2, na)),    # ← elevated
        'label': 'cable_fault'
    }))

    # Mode B: NIC overheating
    nb = sizes[1]
    link_speed = np.random.choice([10000, 25000], nb)
    frames.append(pd.DataFrame({
        'nic_temp_c':             np.random.normal(88, 5, nb),           # ← hot
        'link_speed_mbps':        link_speed,
        'utilization_pct':        np.random.uniform(0.60, 0.95, nb) * 100,  # ← high load
        'tx_util_pct':            np.random.uniform(0.50, 0.90, nb) * 100,
        'crc_error_rate_ppm':     np.abs(np.random.normal(3, 2, nb)),    # mild
        'frame_error_rate_ppm':   np.abs(np.random.normal(1, 0.5, nb)),
        'length_error_rate_ppm':  np.abs(np.random.normal(0.3, 0.2, nb)),
        'symbol_error_rate_ppm':  np.abs(np.random.normal(2, 1, nb)),
        'idle_error_rate_ppm':    np.abs(np.random.normal(1, 0.5, nb)),
        'label': 'nic_overheat'
    }))

    # Mode C: duplex mismatch (one end auto, other forced half-duplex)
    nc = sizes[2]
    link_speed = np.random.choice([100, 1000], nc)
    frames.append(pd.DataFrame({
        'nic_temp_c':             np.random.normal(53, 3, nc),
        'link_speed_mbps':        link_speed,
        'utilization_pct':        np.random.uniform(0.05, 0.50, nc) * 100,
        'tx_util_pct':            np.random.uniform(0.05, 0.45, nc) * 100,
        'crc_error_rate_ppm':     np.abs(np.random.normal(5, 2, nc)),
        'frame_error_rate_ppm':   np.abs(np.random.normal(22, 7, nc)),   # ← very high
        'length_error_rate_ppm':  np.abs(np.random.normal(14, 5, nc)),   # ← very high
        'symbol_error_rate_ppm':  np.abs(np.random.normal(2, 1, nc)),
        'idle_error_rate_ppm':    np.abs(np.random.normal(8, 3, nc)),    # ← elevated
        'label': 'duplex_mismatch'
    }))

    return pd.concat(frames, ignore_index=True)


normal    = make_normal(1000)
anomalous = make_anomalous(50)

df = pd.concat([normal, anomalous], ignore_index=True).sample(
    frac=1, random_state=42).reset_index(drop=True)

true_labels  = df['label'].values
true_binary  = (df['label'] != 'normal').astype(int).values

print(f"Dataset: {len(df)} samples")
print(df['label'].value_counts().to_string())
print(f"\nAnomaly rate: {true_binary.mean():.1%}")


# ─────────────────────────────────────────────────────────────────
# 2. FEATURES
# ─────────────────────────────────────────────────────────────────

FEATURES = [
    # NIC layer
    'nic_temp_c',
    'link_speed_mbps',
    'utilization_pct',
    'tx_util_pct',

    # Cable layer — rates per million packets
    'crc_error_rate_ppm',
    'frame_error_rate_ppm',
    'length_error_rate_ppm',
    'symbol_error_rate_ppm',
    'idle_error_rate_ppm',

    # Derived ratios — often more stable than raw rates
    'cable_error_total',        # sum of all cable errors
    'frame_vs_crc_ratio',       # high = duplex mismatch signature
    'error_per_util',           # errors normalised by load
]

df['cable_error_total']  = (df['crc_error_rate_ppm'] +
                            df['frame_error_rate_ppm'] +
                            df['length_error_rate_ppm'] +
                            df['symbol_error_rate_ppm'] +
                            df['idle_error_rate_ppm'])

df['frame_vs_crc_ratio'] = (df['frame_error_rate_ppm'] /
                            (df['crc_error_rate_ppm'] + 0.01))

df['error_per_util']     = (df['cable_error_total'] /
                            (df['utilization_pct'] + 1))

X = df[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ─────────────────────────────────────────────────────────────────
# 3. STEP 1 — ISOLATION FOREST → PSEUDO-LABELS
# ─────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 1: Isolation Forest → pseudo-labels")
print("="*60)

iso = IsolationForest(
    n_estimators=300,
    contamination=0.05,     # ~5% anomalies expected
    random_state=42
)
iso.fit(X_scaled)
pseudo_binary = (iso.predict(X_scaled) == -1).astype(int)

agreement = (pseudo_binary == true_binary).mean()
print(f"Pseudo-label agreement with true labels: {agreement:.1%}")
print(f"Flagged as anomalous: {pseudo_binary.sum()} samples")

# ─────────────────────────────────────────────────────────────────
# 4. STEP 2 — TRAIN BINARY CLASSIFIER ON PSEUDO-LABELS
# ─────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 2: Binary classifier (normal vs anomaly)")
print("="*60)

X_tr, X_te, y_tr, y_te, ytrue_tr, ytrue_te = train_test_split(
    X_scaled, pseudo_binary, true_binary,
    test_size=0.25, random_state=42, stratify=pseudo_binary
)

rf_binary = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    random_state=42
)
rf_binary.fit(X_tr, y_tr)
pred_binary = rf_binary.predict(X_te)

print(classification_report(ytrue_te, pred_binary,
      target_names=['normal', 'anomaly']))


# ─────────────────────────────────────────────────────────────────
# 5. STEP 3 — MULTI-CLASS: WHICH KIND OF FAULT?
#    Train on pseudo-labels where anomalous class is known from
#    Isolation Forest, but we propagate the true fault type labels
#    for anomalous samples. In real life you'd cluster anomalies
#    or have engineers label the flagged events.
# ─────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 3: Multi-class fault classifier")
print("="*60)

rf_multi = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    random_state=42
)
_, X_te_m, _, ytrue_te_m = train_test_split(
    X_scaled, true_labels,
    test_size=0.25, random_state=42, stratify=true_binary
)
rf_multi.fit(X_scaled, true_labels)
pred_multi = rf_multi.predict(X_te_m)
print(classification_report(ytrue_te_m, pred_multi))


# ─────────────────────────────────────────────────────────────────
# 6. FEATURE IMPORTANCE — what actually matters?
# ─────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("FEATURE IMPORTANCES (multi-class model)")
print("="*60)

imp = pd.DataFrame({
    'feature':    FEATURES,
    'importance': rf_multi.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in imp.iterrows():
    bar = '█' * int(row.importance * 70)
    layer = ('[ NIC  ]' if row.feature in
             ['nic_temp_c','link_speed_mbps','utilization_pct','tx_util_pct']
             else '[ CABL ]' if '_ppm' in row.feature
             else '[DERIV ]')
    print(f"{layer} {row.feature:<28} {bar} {row.importance:.3f}")


# ─────────────────────────────────────────────────────────────────
# 7. INFERENCE EXAMPLES — what a real alert looks like
# ─────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("INFERENCE: example readings")
print("="*60)

def predict_reading(reading: dict, label: str):
    row = pd.DataFrame([reading])
    row['cable_error_total'] = (row['crc_error_rate_ppm'] +
                                row['frame_error_rate_ppm'] +
                                row['length_error_rate_ppm'] +
                                row['symbol_error_rate_ppm'] +
                                row['idle_error_rate_ppm'])
    row['frame_vs_crc_ratio'] = row['frame_error_rate_ppm'] / (row['crc_error_rate_ppm'] + 0.01)
    row['error_per_util']     = row['cable_error_total'] / (row['utilization_pct'] + 1)

    xs = scaler.transform(row[FEATURES])
    probs = rf_multi.predict_proba(xs)[0]
    classes = rf_multi.classes_

    top = sorted(zip(classes, probs), key=lambda x: -x[1])
    verdict = top[0][0]
    confidence = top[0][1]

    icon = '✅' if verdict == 'normal' else '⚠️ '
    print(f"\n  {icon}  [{label}]  → predicted: {verdict.upper()} ({confidence:.0%})")
    for cls, p in top:
        bar = '▓' * int(p * 30)
        print(f"         {cls:<20} {bar} {p:.1%}")

predict_reading({
    'nic_temp_c': 51, 'link_speed_mbps': 1000, 'utilization_pct': 35,
    'tx_util_pct': 20, 'crc_error_rate_ppm': 0.1, 'frame_error_rate_ppm': 0.05,
    'length_error_rate_ppm': 0.02, 'symbol_error_rate_ppm': 0.08,
    'idle_error_rate_ppm': 0.1
}, "Healthy server NIC")

predict_reading({
    'nic_temp_c': 53, 'link_speed_mbps': 1000, 'utilization_pct': 40,
    'tx_util_pct': 25, 'crc_error_rate_ppm': 21, 'frame_error_rate_ppm': 14,
    'length_error_rate_ppm': 1.8, 'symbol_error_rate_ppm': 11,
    'idle_error_rate_ppm': 6
}, "Bad patch cable")

predict_reading({
    'nic_temp_c': 91, 'link_speed_mbps': 25000, 'utilization_pct': 88,
    'tx_util_pct': 82, 'crc_error_rate_ppm': 2.5, 'frame_error_rate_ppm': 0.8,
    'length_error_rate_ppm': 0.2, 'symbol_error_rate_ppm': 1.9,
    'idle_error_rate_ppm': 0.9
}, "NIC overheating under load")

predict_reading({
    'nic_temp_c': 52, 'link_speed_mbps': 1000, 'utilization_pct': 28,
    'tx_util_pct': 15, 'crc_error_rate_ppm': 4.5, 'frame_error_rate_ppm': 25,
    'length_error_rate_ppm': 16, 'symbol_error_rate_ppm': 2,
    'idle_error_rate_ppm': 9
}, "Duplex mismatch (switch forced half)")
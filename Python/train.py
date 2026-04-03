#!/usr/bin/env python3
import argparse
from dataclasses import dataclass, asdict
from configs import CableMaintenanceConfig as Config
from CableMaintenanceAutoencoder import CableAutoencoder
import numpy as np
import json
import tensorflow as tf

# =========================
# Hyperparameters / Config
# =========================
@dataclass
class Config:
    # Data
    model_name=Config.model_name
    train_csv: str = Config.train_csv
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
        "phy_local_rcvr_nok_rate",
        "phy_remote_rcv_nok_rate",
        #"mean_fcs_per_million",
        #"max_fcs_per_million",
        "utilization",
        "flaps_10m",
        #"temp_slope_10m", #if it is not temp from NIC [/sys/class/hwmon/hwmonX/temp1_input] drop this feature
        "temp_delta_10m"
    )
    # Weighting (aligned with feature_cols order)
    # Example: emphasize error-related features
    feature_weights: tuple = (
        1.5, # 2.5,  # frame_err_ppm
        1.5, # 2.0,  # length_err_ppm
        1.0, # 2.0,  # speed_change_count_10m
        1.5, # 1.5,  # speed_is_downgraded
        1.5, # 3.0,  # rx_err_ppm
        1.5, # 3.0,  # phy_receive_errors_rate
        1.5, # 2.0,  # phy_serdes_ber_errors_rate
        1.5, # 4.0,  # fcs_per_million_pkts
        1.5, # 4.0,  # rx_error_rate
        1.5, # 3.0,  # phy_local_rcvr_nok_rate
        1.5, # 3.0,  # phy_remote_rcv_nok_rate
        #1.0, # 4.0,  # mean_fcs_per_million
        #1.5, # 4.0,  # max_fcs_per_million
        1.0, # 1.0,  # utilization
        1.0, # 3.0,  # flaps_10m
        1.0, # 1.0,  # temp_slope_10m
    )
    missing_indicator_weight: float = Config.missing_indicator_weight
    add_missing_indicators: bool = Config.add_missing_indicators

    # Split
    val_ratio: float = Config.val_ratio
    seed: int = Config.seed

    # Model
    enc1: int = Config.enc1
    enc2: int = Config.enc2
    enc3: int = Config.enc3
    bottleneck: int = Config.bottleneck
    dropout: float = Config.dropout
    learning_rate: float = Config.learning_rate

    # Train
    batch_size: int = Config.batch_size
    epochs: int = Config.epochs
    early_stop_patience: int = Config.early_stop_patience #10

    # Threshold
    threshold_quantile: float = Config.threshold_quantile
    threshold: float = 0
    # Export
    export_dir: str = Config.export_dir

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "score"], required=True)
    ap.add_argument("--csv", default=Config.train_csv)          #"./dataset/rpi5_eth0_5s_dataset.csv")
    ap.add_argument("--export-dir", default=Config.export_dir)  #"./output/cable_autoencoder_export")
    args = ap.parse_args()

    cfg = Config(train_csv=args.csv, export_dir=args.export_dir)
    cae = CableAutoencoder(cfg)

    if args.mode == "train":
        cae.train()
        cae.save()

        X = cae.load_csv_features(Config.train_csv)
        X_train, _ = cae.split_train_val(X)
        cae.convert2tflite_quant(X_train)
        print(f"Saved model and stats to: {cfg.export_dir}")
        print(f"Threshold: {float(cae.threshold.numpy()):.6f}")

        output = {
            "model_name": cae.model.name,
            "model_path": "/etc/cable_autoencoder/cable_autoencoder_quant_fp16.tflite",
            "vocabs_path": "",
            "enabled_acceleration": 1,
            "xnnpack_num_threads": 2,
            "fallback_num_threads": 2,
            "threshold": float(cae.threshold.numpy()),
            "iqr": cae.iqr.numpy().tolist(),
            "median": cae.median.numpy().tolist(),
            "eps": float(cae.eps)
        }

        with tf.io.gfile.GFile(f"{cfg.export_dir}/models/model_config.json", "w") as f:
            json.dump(output, f, indent=2)
    else:
        cae.load()
        X = cae.load_csv_features(args.csv)
        err, is_anom = cae.score_tensor(X)
        print("Errors:", err.numpy().tolist())
        print("Anomalies:", tf.cast(is_anom, tf.int32).numpy().tolist())

        for i in range(X.shape[0]):
            out = cae.score_one_with_memory(X[i])
            print(i, out)

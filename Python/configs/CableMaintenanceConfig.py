################################################
############                    ################
############  HYPER PARAMETERS  ################
############                    ################
################################################

# Training
model_name="cable_autoencoder"
train_csv: str = "/workdir/project/cable_autoencoder/Python/dataset/rpi5_eth0_60s_dataset.csv"
val_ratio: float = 0.2
seed: int = 42
enc1: int = 256
enc2: int = 128
enc3: int = 64
bottleneck: int = 16
dropout: float = 0.1
learning_rate: float = 1e-4
missing_indicator_weight: float = 0.5
add_missing_indicators: bool = False
batch_size: int = 1024
epochs: int = 100
early_stop_patience: int = 20 #10
threshold_quantile: float = 0.999  # 99.9%
export_dir: str = "output/cable_autoencoder_export"

# Testing
# MODEL_TO_TEST = '/workdir/project/cable_autoencoder/Python/output/cable_autoencoder_export/models/cable_autoencoder_quant_fp16.tflite'
MODEL_TO_TEST = '/workdir/project/cable_autoencoder/Python/output/cable_autoencoder_export/models/cable_autoencoder_quant_int8.tflite'
OUTPUT_JSON = '/workdir/project/cable_autoencoder/Python/output/cable_autoencoder_export/models/model_config.json'
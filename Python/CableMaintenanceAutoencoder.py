from configs import CableMaintenanceConfig as Config
import tensorflow as tf
import numpy as np
import os
import glob
from Memory import MemoryConfig, LongTermMemory
import json
from dataclasses import dataclass, asdict
from ai_edge_litert.interpreter import Interpreter
import matplotlib.pyplot as plt

def loss_plot(history, output_dir):
    plt.figure(figsize=(7,4))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    plt.savefig(f"{output_dir}/images/loss_curve.png", dpi=150)
    plt.close()

def score_distribution_plot(cable_autoencoder, X_val, threshold, output_dir):
    xhat_val = cable_autoencoder.model(X_val, training=False)
    #val_err = CableAutoencoder.reconstruction_error(X_val, xhat_val).numpy()
    val_err = cable_autoencoder.reconstruction_error_normalized(X_val, xhat_val).numpy()
    th = float(threshold.numpy())

    plt.figure(figsize=(7,4))
    plt.hist(val_err, bins=60, alpha=0.8, color="steelblue")
    plt.axvline(th, color="red", linestyle="--", label=f"threshold={th:.6f}")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Count")
    plt.title("Validation Score Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    plt.savefig(f"{output_dir}/images/score_distribution.png", dpi=150)
    plt.close()

def score_timeline_plot(cable_autoencoder, X_val, threshold, output_dir):
    xhat_val = cable_autoencoder.model(X_val, training=False)
    #val_err = CableAutoencoder.reconstruction_error(X_val, xhat_val).numpy()
    val_err = cable_autoencoder.reconstruction_error_normalized(X_val, xhat_val).numpy()
    th = float(threshold.numpy())

    plt.figure(figsize=(9,4))
    plt.plot(val_err, linewidth=1)
    plt.axhline(th, color="red", linestyle="--", label="threshold")
    plt.xlabel("Validation Sample Index")
    plt.ylabel("Reconstruction Error")
    plt.title("Validation Scores Over Samples")
    plt.legend()
    plt.tight_layout()
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    plt.savefig(f"{output_dir}/images/score_timeline.png", dpi=150)
    plt.close()

class CableAutoencoder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self.input_dim = None
        self.mem = LongTermMemory(MemoryConfig())

        # scaler tensors
        self.median = None
        self.iqr = None
        self.eps = tf.constant(1e-6, dtype=tf.float32) #constant to avoid division with 0

        # anomaly threshold tensor
        self.threshold = tf.Variable(0.0, dtype=tf.float32, trainable=False)

    def _read_header(self, csv_path: str):
        with tf.io.gfile.GFile(csv_path, "r") as f:
            header = f.readline().strip().split(",")
        return header

    def load_csv_features(self, csv_path: str) -> tf.Tensor:
        header = self._read_header(csv_path)
        col_to_idx = {c: i for i, c in enumerate(header)}
        idx = [col_to_idx[c] for c in self.cfg.feature_cols]
        print(f"Indexes from dataset { idx }")

        ds = tf.data.TextLineDataset(csv_path).skip(1)

        def parse_line(line):
            parts = tf.strings.split(line, ",")
            vals = tf.gather(parts, idx)
            vals = tf.where(vals == "", tf.fill(tf.shape(vals), "nan"), vals)
            x = tf.strings.to_number(vals, out_type=tf.float32)  # NaN for missing
            miss = tf.cast(tf.math.is_nan(x), tf.float32)
            x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
            if self.cfg.add_missing_indicators:
                x = tf.concat([x, miss], axis=0)
            return x

        x_list = list(ds.map(parse_line))
        if not x_list:
            raise ValueError(f"No rows found in {csv_path}")
        X = tf.stack(x_list, axis=0)
        return X

    @staticmethod
    def _quantile_axis0(x: tf.Tensor, q: float) -> tf.Tensor:
        x_sorted = tf.sort(x, axis=0)
        n = tf.shape(x_sorted)[0]
        idx = tf.cast(tf.round(q * tf.cast(n - 1, tf.float32)), tf.int32)
        return x_sorted[idx]

    def fit_scaler(self, X_train: tf.Tensor):
        n_num = len(self.cfg.feature_cols)
        X_num = X_train[:, :n_num]

        self.median = self._quantile_axis0(X_num, 0.5)
        q1 = self._quantile_axis0(X_num, 0.25)
        q3 = self._quantile_axis0(X_num, 0.75)
        iqr = q3 - q1
        #self.iqr = tf.where(iqr < self.eps, tf.ones_like(iqr), iqr)  # avoid blowups
        self.iqr = tf.where(iqr < 1e-3, tf.ones_like(iqr), iqr)  # avoid blowups

    def transform(self, X: tf.Tensor) -> tf.Tensor:
        return (X - self.median) / (self.iqr + self.eps)

    def split_train_val(self, X: tf.Tensor):
        n = tf.shape(X)[0]
        n_val = tf.cast(tf.round(tf.cast(n, tf.float32) * self.cfg.val_ratio), tf.int32)
        idx = tf.random.shuffle(tf.range(n), seed=self.cfg.seed)
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        return tf.gather(X, tr_idx), tf.gather(X, val_idx)

    def make_ds(self, X: tf.Tensor, training: bool):
        ds = tf.data.Dataset.from_tensor_slices((X, X))
        if training:
            buffer_size = tf.cast(tf.shape(X)[0], tf.int64)
            ds = ds.shuffle(buffer_size=buffer_size, seed=self.cfg.seed, reshuffle_each_iteration=True)
        return ds.batch(self.cfg.batch_size).prefetch(tf.data.AUTOTUNE)

    def build_model(self, input_dim: int):
        self.loss_weights = self._build_loss_weights()

        n_num = len(self.cfg.feature_cols)
        has_missing_ind = self.cfg.add_missing_indicators

        inp = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)

        x_num = inp[:, :n_num]
        if has_missing_ind:
            x_miss = inp[:, n_num:]

        median_c = tf.constant(self.median, dtype=tf.float32)  # shape [n_num]
        iqr_c = tf.constant(self.iqr, dtype=tf.float32)        # shape [n_num]

        x_num = (x_num - median_c) / (self.iqr + self.eps)
        x_num = tf.keras.ops.clip(x_num, -100.0, 100.0)

        x = tf.keras.ops.concatenate([x_num, x_miss], axis=1) if has_missing_ind else x_num

        x = tf.keras.layers.Dense(self.cfg.enc1, activation="relu")(x)
        x = tf.keras.layers.Dropout(self.cfg.dropout)(x)
        x = tf.keras.layers.Dense(self.cfg.enc2, activation="relu")(x)
        x = tf.keras.layers.Dense(self.cfg.enc3, activation="relu")(x)
        z = tf.keras.layers.Dense(self.cfg.bottleneck, activation="relu")(x)
        x = tf.keras.layers.Dense(self.cfg.enc3, activation="relu")(x)
        x = tf.keras.layers.Dense(self.cfg.enc2, activation="relu")(z)
        x = tf.keras.layers.Dense(self.cfg.enc1, activation="relu")(x)
        out = tf.keras.layers.Dense(input_dim)(x)

        self.model = tf.keras.Model(inp, out, name="cable_autoencoder")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.cfg.learning_rate),
            #loss="mse",
            loss=self.weighted_mse,
        )
        self.model.summary()

    @staticmethod
    def reconstruction_error(x, xhat):
        return tf.reduce_mean(tf.square(x - xhat), axis=1)

    def reconstruction_error_normalized(self, x, xhat):
        # clip values from -10 to 10
        x_norm = tf.clip_by_value(self.transform(x), -100.0, 100.0)
        xhat_norm = tf.clip_by_value(self.transform(xhat), -100.0, 100.0)
        return tf.reduce_mean(tf.square(x_norm - xhat_norm), axis=1)

    def calibrate_threshold(self, X_val_raw: tf.Tensor):
        xhat = self.model(X_val_raw, training=False)
        #err = self.reconstruction_error(X_val_raw, xhat)
        err = self.reconstruction_error_normalized(X_val_raw, xhat)
        th = self._quantile_axis0(tf.expand_dims(err, 1), self.cfg.threshold_quantile)[0]
        self.cfg.threshold = float(th.numpy())
        self.threshold.assign(th)

    def train(self):
        X = self.load_csv_features(self.cfg.train_csv)
        X_train, X_val = self.split_train_val(X)

        self.fit_scaler(X_train)  # fit on raw train
        self.input_dim = int(X_train.shape[1])
        self.build_model(self.input_dim)

        # feed RAW tensors; model normalizes internally
        train_ds = self.make_ds(X_train, training=True)
        val_ds = self.make_ds(X_val, training=False)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.cfg.early_stop_patience,
                restore_best_weights=True,
            )
        ]
        history = self.model.fit(train_ds, validation_data=val_ds, epochs=self.cfg.epochs, callbacks=callbacks, verbose=1)
        loss_plot(history, self.cfg.export_dir)
        self.calibrate_threshold(X_val)  # raw
        score_distribution_plot(self, X_val, self.threshold, self.cfg.export_dir)
        score_timeline_plot(self, X_val, self.threshold, self.cfg.export_dir)

    def score_tensor(self, X: tf.Tensor):
        # raw input; model handles normalization
        xhat = self.model(X, training=False)
        #err = self.reconstruction_error(X, xhat)
        err = self.reconstruction_error_normalized(X, xhat)
        is_anom = err > self.threshold
        return err, is_anom

    def save(self):
        save_path = self.cfg.export_dir + "/models"
        tf.io.gfile.makedirs(save_path)
        self.model.save(f"{save_path}/{self.cfg.model_name}.keras")

        ckpt = tf.train.Checkpoint(
            median=tf.Variable(self.median, trainable=False),
            iqr=tf.Variable(self.iqr, trainable=False),
            threshold=self.threshold,
        )
        ckpt.write(f"{save_path}/stats.ckpt")

        with tf.io.gfile.GFile(f"{save_path}/config_output.json", "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)

    # # save model to directory
    # def save_model(self.model, model_name, save_path):
    #     if not os.path.exists(save_path):
    #         # Create a new directory because it does not exist
    #         os.makedirs(save_path)
    #         print("The new directory is created!")
    #         self.model.save(self.cfg.export_dir+model_name+".keras")
    #     else:
    #         self.model.save(save_path+model_name+".keras")

    # convert it to .tflite
    def convert2tflite(self):
        if(not os.path.isdir(self.cfg.export_dir)):
            print("MISSING DIRECTORY: model directory doesnt exist, save the model first in the ", self.cfg.export_dir)

        model_path = os.path.join(self.cfg.export_dir, self.cfg.model_name + ".keras")
        tflite_file_path = os.path.join(self.cfg.export_dir, self.cfg.model_name + ".tflite")
        if(os.path.isfile(model_path) == 0):
            print("MISSING MODEL: model directory '", model_path, "' is empty ")

        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model input shape: ", model.input_shape)
        input_dim = model.input_shape[-1]

        # Convert directly from Keras model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = []  # no quantization for stability
        tflite_model = converter.convert()

        with open(tflite_file_path, 'wb') as f:
            f.write(tflite_model)

        print("TFLite model saved to:", tflite_file_path)

        # quantized model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_file_path = os.path.join(self.cfg.export_dir, self.cfg.model_name + '_quant_fp16.tflite')
        tflite_model = converter.convert()

        with open(tflite_file_path, "wb") as f:
            f.write(tflite_model)

        print("TFLite [quantized] model saved to:", tflite_file_path)

        # Quick verify
        interpreter = Interpreter(tflite_file_path)
        interpreter.allocate_tensors()
        print(f"TFLite input shape: {interpreter.get_input_details()[0]['shape']}")

        details = interpreter.get_input_details()[0]
        print("shape:", details["shape"])
        print("shape_signature:", details["shape_signature"])

        # Test batch using the model's fixed batch size
        input_details = interpreter.get_input_details()

        for details in input_details:
            shape = details["shape"].tolist()
            dtype = details["dtype"]

            if dtype == tf.int32:
                # Use safe indices to avoid out-of-bounds in Embedding/Gather
                value = tf.zeros(shape, dtype=tf.int32)
            else:
                value = np.random.rand(*shape).astype(dtype)

            interpreter.set_tensor(details["index"], value)

        interpreter.invoke()

        output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

        print(f"Test: {model.output_shape} -> {output.shape}")

        for i, d in enumerate(interpreter.get_output_details()):
            print(i, d["name"], d["shape"], d["dtype"])

    def convert2tflite_quant(self, train_inputs=None):
        output_path = os.path.join(self.cfg.export_dir, "models")
        tf.io.gfile.makedirs(output_path)

        if not os.path.isdir(output_path):
            print("MISSING DIRECTORY:", output_path)
            return

        keras_path = os.path.join(output_path, self.cfg.model_name + ".keras")
        if not os.path.isfile(keras_path):
            print("MISSING MODEL:", keras_path)
            return

        model = tf.keras.models.load_model(keras_path, compile=False)
        for idx, t in enumerate(model.inputs):
            print(idx, t.name, t.dtype)

        # 1) Float32 baseline
        fp32_path = os.path.join(output_path, self.cfg.model_name + "_fp32.tflite")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = []
        tflite_model = converter.convert()
        with open(fp32_path, "wb") as f:
            f.write(tflite_model)
        print("TFLite FP32 saved to:", fp32_path)

        # 2) FP16 quantized
        fp16_path = os.path.join(output_path, self.cfg.model_name + "_quant_fp16.tflite")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        with open(fp16_path, "wb") as f:
            f.write(tflite_model)
        print("TFLite FP16 saved to:", fp16_path)

        # 3) INT8 quantized (requires representative dataset)
        if train_inputs is not None:
            int8_path = os.path.join(output_path, self.cfg.model_name + "_quant_int8.tflite")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            rep_ds = make_representative_dataset(model, train_inputs)
            converter.representative_dataset = rep_ds
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

            # Keep I/O float first to avoid C-side changes:
            # (remove these two lines unless you want fully int8 I/O)
            # converter.inference_input_type = tf.int8
            # converter.inference_output_type = tf.int8

            tflite_model = converter.convert()
            with open(int8_path, "wb") as f:
                f.write(tflite_model)
            print("TFLite INT8 saved to:", int8_path)
        else:
            print("INT8 export skipped (no representative_data provided)")

    def load(self):
        self.model = tf.keras.models.load_model(f"{self.cfg.export_dir}/models/cable_autoencoder.keras", compile=False)
        self.input_dim = self.model.input_shape[-1]

        n_num = len(self.cfg.feature_cols)

        # load config
        with tf.io.gfile.GFile(f"{Config.OUTPUT_JSON}", "r") as f:
            data = json.load(f)

        self.threshold = tf.Variable(data["threshold"], dtype=tf.float32, trainable=False)
        self.iqr = tf.Variable(np.array(data["iqr"], dtype=np.float32), trainable=False)
        self.median = tf.Variable(np.array(data["median"], dtype=np.float32), trainable=False)

        ckpt = tf.train.Checkpoint(median=self.median, iqr=self.iqr, threshold=self.threshold)
        ckpt.restore(f"{self.cfg.export_dir}/models/stats.ckpt").expect_partial()

    def _build_loss_weights(self) -> tf.Tensor:
        w_num = tf.constant(self.cfg.feature_weights, dtype=tf.float32)
        if len(self.cfg.feature_weights) != len(self.cfg.feature_cols):
            raise ValueError("feature_weights must match feature_cols length")

        if self.cfg.add_missing_indicators:
            w_miss = tf.fill([len(self.cfg.feature_cols)], tf.constant(self.cfg.missing_indicator_weight, tf.float32))
            w = tf.concat([w_num, w_miss], axis=0)
        else:
            w = w_num
        return w

    def weighted_mse(self, y_true, y_pred):
        # shape: [batch, features]
        y_true_norm = tf.clip_by_value(self.transform(y_true), -100.0, 100.0)
        y_pred_norm = tf.clip_by_value(self.transform(y_pred), -100.0, 100.0)

        sq = tf.square(y_true_norm - y_pred_norm)
        return tf.reduce_mean(sq * self.loss_weights, axis=1)

    def score_one_with_memory(self, x_row: tf.Tensor, ts: float | None = None):
        # x_row shape: [features]
        x = tf.expand_dims(x_row, axis=0)
        xhat = self.model(x, training=False)
        # err = float(self.reconstruction_error(x, xhat)[0].numpy())
        err = float(self.reconstruction_error_normalized(x, xhat)[0].numpy())
        is_anom = err > float(self.threshold.numpy())
        mem = self.mem.update(err, is_anom, ts=ts)
        return {"error": err, "is_anomaly": is_anom, **mem}

# calib_data should contain REAL samples for each model input.
# Each value is an array with first dim = number of samples.
def make_representative_dataset(model, train_inputs, max_samples=300):
    # train_inputs is expected shape [num_samples, num_features]
    if isinstance(train_inputs, tf.Tensor):
        x = train_inputs.numpy().astype(np.float32)
    else:
        x = np.asarray(train_inputs, dtype=np.float32)

    n = min(max_samples, x.shape[0])

    def representative_dataset():
        for i in range(n):
            # Single-input model: yield one sample batch [1, num_features]
            yield [x[i:i+1]]

    return representative_dataset

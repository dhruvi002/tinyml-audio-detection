"""
Quantization pipeline for YAMNetClassifier checkpoint.
Exports to TFLite INT8 and measures compression + accuracy.

Usage:
    python src/quantize_transfer.py \
        --data_root ./data/ESC-50/ESC-50-master \
        --checkpoint_dir ./models/best_ckpt \
        --output_dir ./models
"""
import argparse, os, json, time
import numpy as np
import tensorflow as tf
from tensorflow import keras

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.train_transfer import (
        YAMNetClassifier, make_wav_dataset,
        run_epoch, MAX_SAMPLES, NUM_CLASSES
    )
except ImportError:
    from train_transfer import (
        YAMNetClassifier, make_wav_dataset,
        run_epoch, MAX_SAMPLES, NUM_CLASSES
    )


# ── Export to concrete SavedModel then TFLite ─────────────────────────────────
def export_tflite(model, output_path, representative_ds=None, quantize=False):
    """Export YAMNetClassifier to TFLite with frozen variables."""
    saved_dir = output_path.replace(".tflite", "_saved")

    @tf.function(input_signature=[tf.TensorSpec([1, MAX_SAMPLES], tf.float32)])
    def infer(wav):
        return model(wav, training=False)

    # Convert concrete function directly — avoids READ_VARIABLE ops
    concrete_func = infer.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    if quantize:
        # Dynamic range quantization — quantizes weights to INT8,
        # activations stay float32. Avoids calibration crash on YAMNet ops.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_bytes = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_bytes)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  → {output_path}  ({size_kb:.1f} KB)")
    return size_kb

def make_rep_dataset(val_ds, n=200):
    """Representative dataset generator for INT8 calibration."""
    samples = []
    for wavs, _ in val_ds:
        for w in wavs.numpy():
            samples.append(w[np.newaxis].astype(np.float32))
            if len(samples) >= n:
                break
        if len(samples) >= n:
            break
    def gen():
        for s in samples:
            yield [s]
    return gen


def evaluate_tflite(tflite_path, val_ds):
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    is_int8 = inp["dtype"] == np.int8
    correct, total, lats = 0, 0, []
    for wavs, labels in val_ds:
        for w, y in zip(wavs.numpy(), labels.numpy()):
            d = w[np.newaxis].astype(np.float32)
            if is_int8:
                s, z = inp["quantization"]
                d = (d / s + z).astype(np.int8)
            interp.set_tensor(inp["index"], d)
            t0 = time.perf_counter()
            interp.invoke()
            lats.append((time.perf_counter() - t0) * 1000)
            lg = interp.get_tensor(out["index"])
            if is_int8:
                s, z = out["quantization"]
                lg = (lg.astype(np.float32) - z) * s
            correct += int(np.argmax(lg) == np.argmax(y))
            total   += 1
    return {
        "accuracy":       correct / total,
        "avg_latency_ms": float(np.mean(lats)),
        "p95_latency_ms": float(np.percentile(lats, 95)),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      default="./data/ESC-50/ESC-50-master")
    p.add_argument("--checkpoint_dir", default="./models/best_ckpt")
    p.add_argument("--output_dir",     default="./models")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path  = os.path.join(args.data_root, "meta", "esc50.csv")
    audio_dir = os.path.join(args.data_root, "audio")

    print("Building val dataset …")
    val_ds = make_wav_dataset(csv_path, audio_dir, folds=[5],
                              batch_size=32, augment=False)

    print(f"\nLoading checkpoint from {args.checkpoint_dir} …")
    model = YAMNetClassifier(num_classes=NUM_CLASSES)
    model.load_weights(args.checkpoint_dir)

    # Baseline accuracy
    loss_fn = keras.losses.CategoricalCrossentropy()
    opt     = keras.optimizers.Adam(1e-5)
    _, acc_base = run_epoch(model, val_ds, opt, loss_fn,
                            train=False, finetune_yamnet=False)
    print(f"Baseline val accuracy: {acc_base*100:.2f}%")

    # 1. Float32 TFLite
    print("\n[1/3] Float32 TFLite …")
    f32_path = os.path.join(args.output_dir, "model_float32.tflite")
    size_f32 = export_tflite(model, f32_path, quantize=False)

    # 2. INT8 PTQ
    print("\n[2/3] Dynamic range quantisation (weights INT8) …")
    rep_gen  = make_rep_dataset(val_ds)
    int8_path = os.path.join(args.output_dir, "model_int8.tflite")
    size_int8 = export_tflite(model, int8_path,
                              representative_ds=rep_gen, quantize=True)

    # 3. Evaluate both
    print("\n[3/3] Evaluating TFLite models …")
    val_ds2    = make_wav_dataset(csv_path, audio_dir, folds=[5],
                                  batch_size=32, augment=False)
    f32_eval   = evaluate_tflite(f32_path,  val_ds2)
    val_ds3    = make_wav_dataset(csv_path, audio_dir, folds=[5],
                                  batch_size=32, augment=False)
    int8_eval  = evaluate_tflite(int8_path, val_ds3)

    compression = (1 - size_int8 / size_f32) * 100
    acc_drop    = (f32_eval["accuracy"] - int8_eval["accuracy"]) * 100

    results = {
        "baseline_keras_acc": float(acc_base),
        "float32_tflite": {
            "size_kb":        size_f32,
            "accuracy":       f32_eval["accuracy"],
            "avg_latency_ms": f32_eval["avg_latency_ms"],
        },
        "dynamic_range_quant": {
            "size_kb":        size_int8,
            "accuracy":       int8_eval["accuracy"],
            "compression_pct": compression,
            "acc_drop_pct":   acc_drop,
            "avg_latency_ms": int8_eval["avg_latency_ms"],
        },
    }

    out_path = os.path.join(args.output_dir, "quantization_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("  QUANTIZATION RESULTS")
    print("="*60)
    print(f"  Baseline (Keras)  : acc={acc_base*100:.2f}%")
    print(f"  Float32 TFLite    : {size_f32:.1f} KB   acc={f32_eval['accuracy']*100:.2f}%   lat={f32_eval['avg_latency_ms']:.1f}ms")
    print(f"  Dynamic range quant : {size_int8:.1f} KB   acc={int8_eval['accuracy']*100:.2f}%   lat={int8_eval['avg_latency_ms']:.1f}ms   ({compression:.1f}% smaller, Δacc={acc_drop:.2f}%)")
    print("="*60)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

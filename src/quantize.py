"""
Quantization & Pruning Pipeline
"""

import argparse
import os
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot

try:
    from src.dataset import make_dataset, waveform_to_mel, load_wav
except ImportError:
    from dataset import make_dataset, waveform_to_mel, load_wav

FOLD_VAL = [5]


def make_representative_dataset(csv_path, audio_dir, n_samples=200):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[df["fold"].isin(FOLD_VAL)].sample(n=min(n_samples, len(df)), random_state=0)
    def gen():
        for _, row in df.iterrows():
            path = os.path.join(audio_dir, row["filename"])
            wav  = load_wav(path).numpy()
            mel  = waveform_to_mel(tf.constant(wav)).numpy()
            yield [mel[np.newaxis].astype(np.float32)]
    return gen


def quantize_ptq(model, representative_dataset_gen, output_path, float16=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if float16:
        converter.target_spec.supported_types = [tf.float16]
    else:
        converter.representative_dataset    = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type      = tf.int8
        converter.inference_output_type     = tf.int8
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size = os.path.getsize(output_path)
    print(f"  TFLite model → {output_path}  ({size/1024:.1f} KB)")
    return size


def apply_pruning(model, train_ds, val_ds, target_sparsity=0.70, fine_tune_epochs=10):
    n_batches = len(list(train_ds))
    end_step  = n_batches * fine_tune_epochs

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=target_sparsity,
            begin_step=0,
            end_step=end_step,
        )
    }

    # Wrap the whole model at once — avoids weight count mismatch from clone_model
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    pruned_model.compile(
        optimizer=keras.optimizers.legacy.Adam(1e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir="./models/tb_prune"),
    ]

    print(f"\nFine-tuning with pruning (target sparsity={target_sparsity*100:.0f}%) …")
    pruned_model.fit(train_ds, validation_data=val_ds,
                     epochs=fine_tune_epochs, callbacks=callbacks, verbose=1)

    stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)
    print("Pruning wrappers stripped.")
    return stripped


def evaluate_tflite(tflite_path, val_ds):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    is_int8 = inp["dtype"] == np.int8

    correct, total = 0, 0
    latencies = []

    for batch_x, batch_y in val_ds:
        for x, y in zip(batch_x.numpy(), batch_y.numpy()):
            inp_data = x[np.newaxis].astype(np.float32)
            if is_int8:
                s, z = inp["quantization"]
                inp_data = (inp_data / s + z).astype(np.int8)
            interpreter.set_tensor(inp["index"], inp_data)
            t0 = time.perf_counter()
            interpreter.invoke()
            latencies.append((time.perf_counter() - t0) * 1000)
            logits = interpreter.get_tensor(out["index"])
            if is_int8:
                s, z = out["quantization"]
                logits = (logits.astype(np.float32) - z) * s
            correct += int(np.argmax(logits) == np.argmax(y))
            total   += 1

    return {
        "accuracy":       correct / total,
        "avg_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
    }


def run_quantization_pipeline(args):
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path  = os.path.join(args.data_root, "meta", "esc50.csv")
    audio_dir = os.path.join(args.data_root, "audio")

    print(f"Loading model from {args.model_path} …")
    model = keras.models.load_model(args.model_path)

    # 1. Baseline float32
    print("\n[1/4] Baseline float32 TFLite …")
    base_path = os.path.join(args.output_dir, "model_float32.tflite")
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    with open(base_path, "wb") as f:
        f.write(conv.convert())
    size_f32 = os.path.getsize(base_path)
    print(f"  Float32 → {size_f32/1024:.1f} KB")

    # 2. INT8 PTQ
    print("\n[2/4] INT8 post-training quantisation …")
    rep_gen   = make_representative_dataset(csv_path, audio_dir)
    int8_path = os.path.join(args.output_dir, "model_int8.tflite")
    size_int8 = quantize_ptq(model, rep_gen, int8_path)

    # 3. Pruning
    print("\n[3/4] Magnitude pruning …")
    train_ds = make_dataset(csv_path, audio_dir, folds=[1,2,3,4], batch_size=32, augment=True,  cache=False)
    val_ds   = make_dataset(csv_path, audio_dir, folds=[5],        batch_size=32, augment=False, cache=False)

    pruned_model = apply_pruning(model, train_ds, val_ds,
                                 target_sparsity=args.target_sparsity,
                                 fine_tune_epochs=args.prune_epochs)
    pruned_keras_path = os.path.join(args.output_dir, "model_pruned.keras")
    pruned_model.save(pruned_keras_path)

    rep_gen2         = make_representative_dataset(csv_path, audio_dir)
    pruned_int8_path = os.path.join(args.output_dir, "model_pruned_int8.tflite")
    size_pruned_int8 = quantize_ptq(pruned_model, rep_gen2, pruned_int8_path)

    # 4. Evaluate
    print("\n[4/4] Evaluating …")
    val_ds_eval = make_dataset(csv_path, audio_dir, folds=[5], batch_size=32, augment=False, cache=False)
    acc_base    = model.evaluate(val_ds_eval, verbose=0)[1]

    val_ds_eval = make_dataset(csv_path, audio_dir, folds=[5], batch_size=32, augment=False, cache=False)
    pruned_model.compile(optimizer=keras.optimizers.legacy.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    acc_pruned  = pruned_model.evaluate(val_ds_eval, verbose=0)[1]

    val_ds_eval      = make_dataset(csv_path, audio_dir, folds=[5], batch_size=32, augment=False, cache=False)
    int8_eval        = evaluate_tflite(int8_path,        val_ds_eval)
    val_ds_eval      = make_dataset(csv_path, audio_dir, folds=[5], batch_size=32, augment=False, cache=False)
    pruned_int8_eval = evaluate_tflite(pruned_int8_path, val_ds_eval)

    compression = (1 - size_pruned_int8 / size_f32) * 100
    acc_drop    = (acc_base - pruned_int8_eval["accuracy"]) * 100

    results = {
        "baseline":    {"size_kb": size_f32/1024,         "accuracy": acc_base},
        "int8_ptq":    {"size_kb": size_int8/1024,        "accuracy": int8_eval["accuracy"],
                        "compression": (1-size_int8/size_f32)*100,
                        "avg_latency_ms": int8_eval["avg_latency_ms"]},
        "pruned_int8": {"size_kb": size_pruned_int8/1024, "accuracy": pruned_int8_eval["accuracy"],
                        "compression": compression, "acc_drop_pct": acc_drop,
                        "avg_latency_ms": pruned_int8_eval["avg_latency_ms"]},
    }

    with open(os.path.join(args.output_dir, "quantization_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*55)
    print("  Results")
    print("="*55)
    print(f"  Float32   : {size_f32/1024:.1f} KB  |  acc={acc_base*100:.2f}%")
    print(f"  INT8 PTQ  : {size_int8/1024:.1f} KB  |  acc={int8_eval['accuracy']*100:.2f}%  ({(1-size_int8/size_f32)*100:.1f}% smaller)")
    print(f"  Pruned+INT8: {size_pruned_int8/1024:.1f} KB  |  acc={pruned_int8_eval['accuracy']*100:.2f}%  ({compression:.1f}% smaller, Δ={acc_drop:.2f}%)")
    print("="*55)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",      default="./models/best_model.keras")
    p.add_argument("--data_root",       default="./data/ESC-50/ESC-50-master")
    p.add_argument("--output_dir",      default="./models")
    p.add_argument("--target_sparsity", type=float, default=0.70)
    p.add_argument("--prune_epochs",    type=int,   default=10)
    return p.parse_args()


if __name__ == "__main__":
    run_quantization_pipeline(parse_args())

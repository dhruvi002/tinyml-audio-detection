"""
Continued fine-tuning from a saved YAMNet checkpoint.
Run this after train_transfer.py to push accuracy higher.

Usage:
    python src/finetune.py \
        --data_root ./data/ESC-50/ESC-50-master \
        --checkpoint_dir ./models/best_ckpt \
        --output_dir ./models \
        --lr 3e-5 \
        --epochs 30 \
        --patience 20
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
        train_phase, run_epoch, NUM_CLASSES
    )
except ImportError:
    from train_transfer import (
        YAMNetClassifier, make_wav_dataset,
        train_phase, run_epoch, NUM_CLASSES
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",       default="./data/ESC-50/ESC-50-master")
    p.add_argument("--checkpoint_dir",  default="./models/best_ckpt")
    p.add_argument("--output_dir",      default="./models")
    p.add_argument("--lr",              type=float, default=3e-5)
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--patience",        type=int,   default=20)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path  = os.path.join(args.data_root, "meta", "esc50.csv")
    audio_dir = os.path.join(args.data_root, "audio")

    print("Building datasets …")
    train_ds = make_wav_dataset(csv_path, audio_dir, folds=[1,2,3,4],
                                batch_size=args.batch_size, augment=True)
    val_ds   = make_wav_dataset(csv_path, audio_dir, folds=[5],
                                batch_size=args.batch_size, augment=False)
    print(f"Train: {len(list(train_ds))} batches | Val: {len(list(val_ds))} batches")

    print(f"\nLoading checkpoint from {args.checkpoint_dir} …")
    model = YAMNetClassifier(num_classes=NUM_CLASSES)
    model.load_weights(args.checkpoint_dir)

    # Verify starting accuracy
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    opt_eval = keras.optimizers.Adam(args.lr)
    _, start_acc = run_epoch(model, val_ds, opt_eval, loss_fn,
                             train=False, finetune_yamnet=False)
    print(f"Starting val accuracy: {start_acc*100:.2f}%")

    t0 = time.time()
    optimizer = keras.optimizers.Adam(args.lr)
    best_acc = train_phase(
        model, train_ds, val_ds, optimizer,
        epochs=args.epochs,
        output_dir=args.output_dir,
        phase_name="finetune_continued",
        finetune_yamnet=True,
        patience=args.patience,
    )
    elapsed = time.time() - t0

    print(f"\nFine-tuning complete in {elapsed/60:.1f} min")
    print(f"Start accuracy : {start_acc*100:.2f}%")
    print(f"Best accuracy  : {best_acc*100:.2f}%")
    print(f"Improvement    : +{(best_acc - start_acc)*100:.2f}pp")

    results_path = os.path.join(args.output_dir, "transfer_results.json")
    existing = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)

    existing.update({
        "finetune_start_acc":    float(start_acc),
        "finetune_best_acc":     float(best_acc),
        "finetune_lr":           args.lr,
        "finetune_epochs_run":   args.epochs,
        "finetune_time_min":     elapsed / 60,
    })
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()

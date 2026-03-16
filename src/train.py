import argparse, os, json, time
import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    from src.model   import build_ds_cnn, NUM_CLASSES, INPUT_SHAPE
    from src.dataset import make_dataset
except ImportError:
    from model   import build_ds_cnn, NUM_CLASSES, INPUT_SHAPE
    from dataset import make_dataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default="./data/ESC-50/ESC-50-master")
    p.add_argument("--epochs",     type=int,   default=120)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--output_dir", default="./models")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()

def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 60)
    print("  TinyML Audio — Improved Training Run")
    print("=" * 60)
    csv_path  = os.path.join(args.data_root, "meta", "esc50.csv")
    audio_dir = os.path.join(args.data_root, "audio")
    train_ds = make_dataset(csv_path, audio_dir, folds=[1,2,3,4], batch_size=args.batch_size, augment=True)
    val_ds   = make_dataset(csv_path, audio_dir, folds=[5],        batch_size=args.batch_size, augment=False)
    print(f"Train: {len(list(train_ds))} batches | Val: {len(list(val_ds))} batches")
    model = build_ds_cnn(width_multiplier=1.0)
    model.summary()
    n_batches    = len(list(train_ds))
    warmup_steps = n_batches * 10
    total_steps  = n_batches * args.epochs
    def lr_fn(epoch):
        step = epoch * n_batches
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return float(args.lr * 0.5 * (1.0 + np.cos(np.pi * progress))) + 1e-6
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(args.lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, "best_model.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=30,
            restore_best_weights=True, verbose=1, start_from_epoch=60),
        keras.callbacks.LearningRateScheduler(lr_fn, verbose=0),
        keras.callbacks.CSVLogger(os.path.join(args.output_dir, "history.csv")),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
    ]
    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.epochs, callbacks=callbacks, verbose=1)
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} min")
    val_loss, val_acc, val_top5 = model.evaluate(val_ds, verbose=0)
    print(f"Final Val Accuracy : {val_acc*100:.2f}%")
    print(f"Final Val Top-5    : {val_top5*100:.2f}%")
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, f, indent=2)

if __name__ == "__main__":
    main()

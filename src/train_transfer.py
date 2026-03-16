"""
YAMNet Transfer Learning with custom GradientTape training loop.
Fine-tunes YAMNet SavedModel weights directly — bypasses Keras trainable flag limitation.

Phases:
  1. Frozen YAMNet, train head only (Adam 1e-3, 15 epochs)
  2. Fine-tune all YAMNet + head (Adam 1e-5, 30 epochs)
"""
import argparse, os, json, time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

YAMNET_URL  = "https://tfhub.dev/google/yamnet/1"
TARGET_SR   = 16000
MAX_SAMPLES = TARGET_SR * 5
NUM_CLASSES = 50
AUTOTUNE    = tf.data.AUTOTUNE


# ── Dataset ───────────────────────────────────────────────────────────────────
def load_wav_16k(path):
    raw = tf.io.read_file(path)
    wav, _ = tf.audio.decode_wav(raw, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    orig_len   = tf.shape(wav)[0]
    target_len = tf.cast(tf.cast(orig_len, tf.float32) * TARGET_SR / 22050, tf.int32)
    wav = tf.image.resize(
        tf.reshape(wav, [1, -1, 1]), [1, target_len], method="bilinear"
    )[0, :, 0]
    wav = wav[:MAX_SAMPLES]
    pad = MAX_SAMPLES - tf.shape(wav)[0]
    wav = tf.pad(wav, [[0, pad]])
    wav.set_shape([MAX_SAMPLES])
    return wav


def make_wav_dataset(csv_path, audio_dir, folds,
                     batch_size=32, augment=True, shuffle_buf=2000):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[df["fold"].isin(folds)]
    paths  = [os.path.join(audio_dir, fn) for fn in df["filename"]]
    labels = df["target"].tolist()
    ds = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(paths),
        tf.data.Dataset.from_tensor_slices(labels),
    ))
    if augment:
        ds = ds.shuffle(shuffle_buf, reshuffle_each_iteration=True)

    def process(path, label):
        wav = load_wav_16k(path)
        if augment:
            shift_s = tf.cast(
                tf.random.uniform((), -0.2, 0.2) * tf.cast(tf.shape(wav)[0], tf.float32),
                tf.int32)
            wav = tf.cond(tf.random.uniform(()) > 0.5,
                          lambda: tf.roll(wav, shift_s, axis=0), lambda: wav)
            sp  = tf.reduce_mean(wav ** 2) + 1e-8
            wav = tf.cond(tf.random.uniform(()) > 0.5,
                          lambda: wav + tf.random.normal(
                              tf.shape(wav), stddev=tf.sqrt(sp / (10 ** (30.0/10.0)))),
                          lambda: wav)
        return wav, tf.one_hot(label, NUM_CLASSES)

    ds = ds.map(process, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ── Model ─────────────────────────────────────────────────────────────────────
class YAMNetClassifier(tf.Module):
    """
    Holds YAMNet (SavedModel) + a Keras head.
    Training loop accesses yamnet.trainable_variables directly,
    giving real gradient flow regardless of Keras trainable flags.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        print("Loading YAMNet from TF Hub …")
        self.yamnet = hub.load(YAMNET_URL)
        # Keras head
        self.head = keras.Sequential([
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="softmax"),
        ], name="head")
        # Build head by running a dummy forward pass
        dummy = tf.zeros([1, MAX_SAMPLES])
        _ = self(dummy, training=False)

    def embed(self, waveforms):
        """Run YAMNet on a batch, return attention-pooled embeddings (batch, 1024)."""
        def embed_one(wav):
            _, embeddings, _ = self.yamnet(wav)   # (n_frames, 1024)
            # Attention pooling
            w = tf.nn.softmax(
                tf.reduce_mean(
                    tf.matmul(embeddings, embeddings, transpose_b=True),
                    axis=-1),
                axis=-1)
            return tf.reduce_sum(embeddings * tf.expand_dims(w, -1), axis=0)
        return tf.map_fn(embed_one, waveforms, fn_output_signature=tf.float32)

    def __call__(self, waveforms, training=False):
        emb = self.embed(waveforms)
        return self.head(emb, training=training)

    @property
    def yamnet_variables(self):
        return list(self.yamnet._yamnet.trainable_variables)

    @property
    def head_variables(self):
        return self.head.trainable_variables

    def save_weights(self, path):
        """Save head weights + yamnet checkpoint."""
        os.makedirs(path, exist_ok=True)
        self.head.save_weights(os.path.join(path, "head.weights.h5"))
        ckpt = tf.train.Checkpoint(yamnet=self.yamnet._yamnet)
        ckpt.write(os.path.join(path, "yamnet_ckpt"))

    def load_weights(self, path):
        self.head.load_weights(os.path.join(path, "head.weights.h5"))
        ckpt = tf.train.Checkpoint(yamnet=self.yamnet._yamnet)
        ckpt.read(os.path.join(path, "yamnet_ckpt"))


# ── Training loop ─────────────────────────────────────────────────────────────
def run_epoch(model, dataset, optimizer, loss_fn,
              train=True, finetune_yamnet=False):
    total_loss, correct, total = 0.0, 0, 0
    for wavs, labels in dataset:
        watched = model.head_variables
        if finetune_yamnet:
            watched = watched + model.yamnet_variables

        with tf.GradientTape() as tape:
    
            preds = model(wavs, training=train)
            loss  = loss_fn(labels, preds)

        if train:
            grads = tape.gradient(loss, watched)
            optimizer.apply_gradients(zip(grads, watched))

        total_loss += float(loss)
        correct    += int(tf.reduce_sum(
            tf.cast(tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1)), tf.int32)))
        total      += int(tf.shape(wavs)[0])

    return total_loss / len(list(dataset)), correct / total


def train_phase(model, train_ds, val_ds, optimizer, epochs,
                output_dir, phase_name, finetune_yamnet=False,
                patience=12):
    best_val_acc  = 0.0
    best_ckpt_dir = os.path.join(output_dir, "best_ckpt")
    no_improve    = 0
    history       = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    print(f"\n{'='*55}")
    print(f"  {phase_name}  |  fine-tune YAMNet={finetune_yamnet}")
    print(f"{'='*55}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(
            model, train_ds, optimizer, loss_fn,
            train=True, finetune_yamnet=finetune_yamnet)
        va_loss, va_acc = run_epoch(
            model, val_ds, optimizer, loss_fn,
            train=False, finetune_yamnet=False)
        elapsed = time.time() - t0

        history["loss"].append(tr_loss)
        history["acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        improved = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            model.save_weights(best_ckpt_dir)
            improved = " ✓ saved"
            no_improve = 0
        else:
            no_improve += 1

        print(f"Epoch {epoch:3d}/{epochs} — "
              f"loss={tr_loss:.4f} acc={tr_acc*100:.1f}%  "
              f"val_loss={va_loss:.4f} val_acc={va_acc*100:.2f}%  "
              f"({elapsed:.0f}s){improved}")

        if no_improve >= patience:
            print(f"  Early stopping (no improvement for {patience} epochs)")
            break

    model.load_weights(best_ckpt_dir)
    print(f"  Best val_accuracy this phase: {best_val_acc*100:.2f}%")

    with open(os.path.join(output_dir, f"history_{phase_name}.json"), "w") as f:
        json.dump(history, f, indent=2)
    return best_val_acc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default="./data/ESC-50/ESC-50-master")
    p.add_argument("--output_dir", default="./models")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed",       type=int, default=42)
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

    model = YAMNetClassifier(num_classes=NUM_CLASSES)
    print(f"Head params: {sum(np.prod(v.shape) for v in model.head_variables):,}")
    print(f"YAMNet params: {sum(np.prod(v.shape) for v in model.yamnet_variables):,}")

    t0 = time.time()

    # Phase 1: head only, higher LR
    opt1 = keras.optimizers.Adam(1e-3)
    acc1 = train_phase(model, train_ds, val_ds, opt1,
                       epochs=15, output_dir=args.output_dir,
                       phase_name="phase1_frozen",
                       finetune_yamnet=False, patience=8)

    # Phase 2: full fine-tune, low LR
    opt2 = keras.optimizers.Adam(1e-5)
    acc2 = train_phase(model, train_ds, val_ds, opt2,
                       epochs=40, output_dir=args.output_dir,
                       phase_name="phase2_finetune",
                       finetune_yamnet=True, patience=15)

    elapsed = time.time() - t0
    print(f"\nTotal training time: {elapsed/60:.1f} min")
    print(f"Phase 1 best: {acc1*100:.2f}%  →  Phase 2 best: {acc2*100:.2f}%")

    # Final eval
    _, final_acc = run_epoch(model, val_ds, opt2,
                             keras.losses.CategoricalCrossentropy(),
                             train=False, finetune_yamnet=False)
    print(f"Final Val Accuracy: {final_acc*100:.2f}%")

    with open(os.path.join(args.output_dir, "transfer_results.json"), "w") as f:
        json.dump({"phase1_acc": float(acc1),
                   "phase2_acc": float(acc2),
                   "final_acc":  float(final_acc),
                   "training_time_min": elapsed/60}, f, indent=2)


if __name__ == "__main__":
    main()

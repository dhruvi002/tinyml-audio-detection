from model import N_FRAMES, N_MELS, N_FFT, HOP_LENGTH, MAX_SAMPLES, NUM_CLASSES, SAMPLE_RATE
import os
import numpy as np
import pandas as pd
import tensorflow as tf

AUTOTUNE     = tf.data.AUTOTUNE


def load_wav(path):
    raw = tf.io.read_file(path)
    wav, sr = tf.audio.decode_wav(raw, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    wav = wav[:MAX_SAMPLES]
    pad = MAX_SAMPLES - tf.shape(wav)[0]
    wav = tf.pad(wav, [[0, pad]])
    wav.set_shape([MAX_SAMPLES])
    return wav


def waveform_to_mel(wav):
    stfts = tf.signal.stft(wav, frame_length=N_FFT, frame_step=HOP_LENGTH)
    power = tf.abs(stfts) ** 2
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        N_MELS, N_FFT // 2 + 1, SAMPLE_RATE,
        lower_edge_hertz=20.0, upper_edge_hertz=8000.0)
    mel   = tf.tensordot(power, mel_w, 1)
    log_m = tf.math.log(mel + 1e-6)
    log_m = tf.transpose(log_m)
    log_m = tf.expand_dims(log_m, -1)
    mu, var = tf.nn.moments(log_m, axes=[0, 1, 2], keepdims=True)
    log_m   = (log_m - mu) / (tf.sqrt(var) + 1e-8)
    return log_m


def time_shift(wav, max_shift=0.2):
    shift = tf.random.uniform((), -max_shift, max_shift)
    shift_samples = tf.cast(shift * tf.cast(tf.shape(wav)[0], tf.float32), tf.int32)
    return tf.roll(wav, shift_samples, axis=0)


def add_noise(wav, snr_db=30.0):
    signal_power = tf.reduce_mean(wav ** 2) + 1e-8
    noise_power  = signal_power / (10 ** (snr_db / 10.0))
    noise = tf.random.normal(tf.shape(wav), stddev=tf.sqrt(noise_power))
    return wav + noise


def spec_augment(mel, freq_mask_param=8, time_mask_param=20, num_masks=2):
    shape = tf.shape(mel)
    F, T  = shape[0], shape[1]
    for _ in range(num_masks):
        f0 = tf.random.uniform((), 0, F - freq_mask_param, dtype=tf.int32)
        fw = tf.random.uniform((), 0, freq_mask_param,     dtype=tf.int32)
        mask_f = tf.concat([tf.ones([f0, T, 1]), tf.zeros([fw, T, 1]), tf.ones([F - f0 - fw, T, 1])], axis=0)
        mel = mel * tf.cast(mask_f, mel.dtype)
        t0 = tf.random.uniform((), 0, T - time_mask_param, dtype=tf.int32)
        tw = tf.random.uniform((), 0, time_mask_param,     dtype=tf.int32)
        mask_t = tf.concat([tf.ones([F, t0, 1]), tf.zeros([F, tw, 1]), tf.ones([F, T - t0 - tw, 1])], axis=1)
        mel = mel * tf.cast(mask_t, mel.dtype)
    return mel


def make_dataset(csv_path, audio_dir, folds, batch_size=32, augment=True, cache=True, shuffle_buf=2000):
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
        wav = load_wav(path)
        if augment:
            wav = tf.cond(tf.random.uniform(()) > 0.5, lambda: time_shift(wav), lambda: wav)
            wav = tf.cond(tf.random.uniform(()) > 0.5, lambda: add_noise(wav),  lambda: wav)
        mel = waveform_to_mel(wav)
        if augment:
            mel = spec_augment(mel)
        return mel, tf.one_hot(label, NUM_CLASSES)

    ds = ds.map(process, num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def download_esc50(dest="./data/ESC-50"):
    import urllib.request, zipfile
    url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, "esc50.zip")
    print("Downloading ESC-50 ...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    print(f"Extracted to {dest}")
    return os.path.join(dest, "ESC-50-master")

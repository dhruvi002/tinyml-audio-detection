import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

SAMPLE_RATE    = 22050
N_MELS         = 64
N_FFT          = 1024
HOP_LENGTH     = 512
MAX_DURATION   = 5.0
MAX_SAMPLES    = int(MAX_DURATION * SAMPLE_RATE)
NUM_CLASSES    = 50

# Compute actual frame count from STFT (no hardcoding)
def _get_n_frames():
    wav = tf.zeros([MAX_SAMPLES])
    stfts = tf.signal.stft(wav, frame_length=N_FFT, frame_step=HOP_LENGTH)
    return int(stfts.shape[0])

N_FRAMES    = _get_n_frames()
INPUT_SHAPE = (N_MELS, N_FRAMES, 1)


def ds_conv_block(x, filters, kernel_size=(3,3), stride=(1,1), use_bn=True, activation="relu6"):
    x = layers.DepthwiseConv2D(kernel_size, strides=stride, padding="same", use_bias=not use_bn)(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv2D(filters, (1,1), padding="same", use_bias=not use_bn)(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x


def squeeze_excite(x, ratio=8):
    c = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1,1,c))(se)
    se = layers.Conv2D(max(1, c//ratio), (1,1), activation="relu")(se)
    se = layers.Conv2D(c, (1,1), activation="sigmoid")(se)
    return layers.Multiply()([x, se])


def build_ds_cnn(input_shape=None, num_classes=NUM_CLASSES, width_multiplier=1.0):
    if input_shape is None:
        input_shape = INPUT_SHAPE
    inputs = keras.Input(shape=input_shape, name="mel_spectrogram")
    def _w(c): return max(1, int(c * width_multiplier))
    x = layers.Conv2D(_w(32), (3,3), strides=(2,2), padding="same", use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu6", name="stem_act")(x)
    cfg = [(_w(64),(3,3),(1,1)), (_w(128),(3,3),(2,2)), (_w(128),(3,3),(1,1)),
           (_w(256),(3,3),(2,2)), (_w(256),(3,3),(1,1)), (_w(512),(3,3),(2,2))]
    for i, (f,k,s) in enumerate(cfg):
        x = ds_conv_block(x, f, kernel_size=k, stride=s)
        if i % 2 == 1: x = squeeze_excite(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="logits")(x)
    return keras.Model(inputs, outputs, name="DS_CNN_Audio")


if __name__ == "__main__":
    print(f"N_FRAMES = {N_FRAMES}")
    print(f"INPUT_SHAPE = {INPUT_SHAPE}")
    model = build_ds_cnn()
    model.summary()
    print(f"Total params: {model.count_params():,}")

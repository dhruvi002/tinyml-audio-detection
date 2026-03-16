# TinyML Audio Event Detection on Edge Devices

**Stack**: YAMNet · TensorFlow Hub · TFLite · Custom GradientTape · ARM Cortex-M

Acoustic event detection across 50 classes using transfer learning from YAMNet (pretrained on AudioSet), fine-tuned on ESC-50 via a 3-phase gradual unfreezing strategy with a custom GradientTape training loop, then compressed via dynamic range quantization for edge deployment.

---

## Results

| Variant | Size | Accuracy | Notes |
|---|---|---|---|
| YAMNet baseline (frozen) | 15.2 MB | 48.75% | Head only, phase 1 |
| + Partial fine-tune | 15.2 MB | 71.00% | YAMNet unfrozen, phase 2 |
| + Continued fine-tune | 15.2 MB | 74.75% | Lower LR, phase 3 |
| Dynamic range INT8 | **4.0 MB** | **70.00%** | **73.6% smaller, 2.5× faster** |

**Inference latency**: 17.6ms (float32) → 6.9ms (INT8) on host CPU  
**Target hardware**: ARM Cortex-M4 @ 168 MHz, <10 mW estimated power

---

## Project Structure

```
tinyml-audio/
├── src/
│   ├── model.py              # DS-CNN architecture (depthwise separable CNN + SE)
│   ├── dataset.py            # ESC-50 loader, Mel spectrogram pipeline, augmentation
│   ├── train.py              # From-scratch DS-CNN training (baseline)
│   ├── train_transfer.py     # YAMNet transfer learning — 3-phase gradual unfreezing
│   ├── finetune.py           # Continued fine-tuning from checkpoint
│   ├── quantize.py           # DS-CNN quantization + pruning pipeline
│   └── quantize_transfer.py  # YAMNet TFLite export + dynamic range quantization
├── cpp/
│   ├── audio_inference.cc    # TFLite Micro C++ inference engine (ARM Cortex-M)
│   └── labels.h              # ESC-50 class labels
├── models/                   # Saved checkpoints + .tflite outputs
├── data/                     # ESC-50 dataset
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Environment setup (Python 3.11 required)

```bash
conda create -n tinyml-audio python=3.11
conda activate tinyml-audio
pip install -r requirements.txt
pip install -e .
```

### 2. Download ESC-50

```bash
python -c "from src.dataset import download_esc50; download_esc50('./data/ESC-50')"
```

### 3. Transfer learning (recommended)

```bash
export PYTHONPATH=$(pwd)

# Phase 1–2: frozen head → full fine-tune
python src/train_transfer.py \
  --data_root ./data/ESC-50/ESC-50-master \
  --output_dir ./models

# Phase 3: continued fine-tuning from best checkpoint
python src/finetune.py \
  --data_root      ./data/ESC-50/ESC-50-master \
  --checkpoint_dir ./models/best_ckpt \
  --output_dir     ./models \
  --lr 3e-5 \
  --epochs 30 \
  --patience 20
```

### 4. Quantize for deployment

```bash
python src/quantize_transfer.py \
  --data_root      ./data/ESC-50/ESC-50-master \
  --checkpoint_dir ./models/best_ckpt \
  --output_dir     ./models
```

---

## Transfer Learning Architecture

### Model

YAMNet backbone (3.7M params, pretrained on AudioSet 521-class dataset) + custom classification head:

```
Raw waveform (16kHz, 5s)
  └─ YAMNet backbone
      └─ Per-frame embeddings (N_frames × 1024)
      └─ Attention pooling → (1024,)
  └─ Dense 512 → ReLU → Dropout 0.4
  └─ Dense 256 → ReLU → Dropout 0.3
  └─ Dense 50  → Softmax
```

### Training Strategy

3-phase gradual unfreezing with a custom `tf.GradientTape` loop that directly tapes gradients through YAMNet's SavedModel variables — bypassing Keras's trainable flag limitation for hub-loaded models.

| Phase | Epochs | LR | YAMNet frozen | Best val acc |
|---|---|---|---|---|
| 1 — Head only | 15 | 1e-3 | Yes | 48.75% |
| 2 — Full fine-tune | 36 | 1e-5 | No | 71.00% |
| 3 — Continued | 30 | 3e-5 | No | 74.75% |

### Augmentation

Time shift (±20%), Gaussian noise (30 dB SNR), applied at waveform level before YAMNet's internal mel spectrogram computation.

---

## Compression Pipeline

**Dynamic range quantization** — weights quantized to INT8, activations remain float32. Chosen over full INT8 PTQ because YAMNet contains `tf.signal.stft` ops that TFLite's INT8 calibrator cannot handle.

| Metric | Float32 | INT8 (dynamic) |
|---|---|---|
| Size | 15.2 MB | 4.0 MB |
| Accuracy | 74.75% | 70.00% |
| Latency | 17.6 ms | 6.9 ms |
| Compression | — | 73.6% |

---

## C++ Deployment (TFLite Micro)

`cpp/audio_inference.cc` provides a complete ARM Cortex-M inference engine:

- Hann-windowed DFT → triangular Mel filterbank → INT8 input quantization
- TFLite Micro invoke with top-5 class decoding
- Power estimation via duty-cycle model (active: 8.5 mW, sleep: 3 µW)

Build for host simulation:
```bash
g++ -O2 -std=c++17 -I${TFLM_ROOT} cpp/audio_inference.cc ${TFLM_LIBS} -o build/audio_inference
```

Cross-compile for ARM Cortex-M4:
```bash
arm-none-eabi-g++ -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 -std=c++17 ...
```

---

## From-Scratch Baseline (DS-CNN)

A depthwise separable CNN trained from scratch on ESC-50 is also included (`src/train.py`) for comparison. It achieves ~48% val accuracy — demonstrating the value of AudioSet pretraining (+26.75pp).

| Model | Params | Val Accuracy | Size (INT8) |
|---|---|---|---|
| DS-CNN (scratch) | 385K | 48.0% | 474 KB |
| YAMNet + head (transfer) | 4.4M | 74.75% | 4.0 MB |

---
